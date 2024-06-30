import os
import logging

import trimesh
from torch import nn

logging.getLogger('PIL').setLevel(logging.WARNING)

import argparse

import imageio
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from net.dataset.scene_dataset import SceneDataset
from net.sdfmodel.fields import SDFNetwork, SingleVarianceNetwork, RenderingNetwork, NeRF
from net.sdfmodel.renderer import NeuSRenderer

TINY_NUMBER = 1e-6


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = SceneDataset(self.conf['dataset'])
        self.data_dir = self.dataset.instance_dir
        self.label_dir = os.path.join(self.data_dir, 'label.png')
        self.total_pixels = self.dataset.total_pixels
        self.cur_iter = 0

        # Training parameters
        self.end_epoch = self.conf.get_int('train.end_epoch')
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.train_dataloader = torch.utils.data.DataLoader(self.dataset,
                                                            batch_size=1,
                                                            shuffle=True,
                                                            collate_fn=self.dataset.collate_fn,
                                                            generator=torch.Generator(device='cuda')
                                                            )

        self.plot_dataloader = torch.utils.data.DataLoader(self.dataset,
                                                           batch_size=1,
                                                           shuffle=True,
                                                           collate_fn=self.dataset.collate_fn,
                                                           generator=torch.Generator(device='cuda')
                                                           )

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        self.start_epoch = 0
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)

        for epoch in range(self.start_epoch, self.end_epoch + 1):

            self.dataset.change_sampling_idx(self.batch_size)

            if self.cur_iter > self.end_iter:
                self.save_checkpoint()
                self.plot_to_disk()
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):

                if self.cur_iter % self.save_freq == 0:
                    self.save_checkpoint()

                if self.cur_iter % self.val_freq == 0:
                    self.plot_to_disk()

                if self.cur_iter % self.val_mesh_freq == 0:
                    self.validate_mesh()

                for key in model_input.keys():
                    model_input[key] = model_input[key].cuda()

                background_rgb = None
                if self.use_white_bkgd:
                    background_rgb = torch.ones([1, 3])

                render_out = self.renderer.render(model_input, training=True, background_rgb=background_rgb,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio())

                mask = model_input['object_mask'].squeeze(0).reshape(-1, 1)
                mask = (mask > 0.5).float()

                mask_sum = mask.sum() + 1e-5

                color_fine = render_out['color_fine']
                s_val = render_out['s_val']
                cdf_fine = render_out['cdf_fine']
                gradient_error = render_out['gradient_error']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']

                true_rgb = ground_truth['rgb'].squeeze(0).cuda()

                # Loss
                color_error = (color_fine - true_rgb) * mask
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

                eikonal_loss = gradient_error

                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

                loss = color_fine_loss + \
                       eikonal_loss * self.igr_weight + \
                       mask_loss * self.mask_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('Loss/loss', loss, self.cur_iter)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.cur_iter)
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.cur_iter)
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.cur_iter)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.cur_iter)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.cur_iter)
                self.writer.add_scalar('Statistics/psnr', psnr, self.cur_iter)

                if self.cur_iter % self.report_freq == 0:
                    print(self.base_exp_dir)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.cur_iter, loss,
                                                               self.optimizer.param_groups[0]['lr']))

                self.cur_iter += 1
                self.update_learning_rate()

            self.start_epoch += 1

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.cur_iter / self.anneal_end])

    def update_learning_rate(self):
        if self.cur_iter < self.warm_up_end:
            learning_factor = self.cur_iter / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.cur_iter - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.start_epoch
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.cur_iter)))

    def plot_to_disk(self):

        sampling_idx = self.dataset.sampling_idx
        self.dataset.change_sampling_idx(-1)
        indices, model_input, ground_truth = next(iter(self.plot_dataloader))

        for key in model_input.keys():
            model_input[key] = model_input[key].cuda()

        split = split_input(model_input, self.total_pixels)

        res = []
        for s in split:
            out = self.renderer.render(s, training=False,
                                       cos_anneal_ratio=self.get_cos_anneal_ratio())
            res.append({
                'normals': out['normals'].detach(),
                'color_fine': out['color_fine'].detach()
            })

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = merge_output(res, self.total_pixels, batch_size)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        plots_dir = os.path.join(self.base_exp_dir,
                                 'validations_fine')

        plot(
            model_outputs,
            ground_truth['rgb'],
            plots_dir,
            self.cur_iter,
            self.dataset.img_res,
        )

        self.dataset.sampling_idx = sampling_idx

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.array([1.01, 1.01, 1.01])
        bound_min = torch.tensor(self.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.cur_iter)))

        logging.info('End')


def split_input(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 256
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        split.append(data)
    return split


def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs


def plot(model_outputs, rgb_gt, path, iters, img_res):
    batch_size, num_samples, _ = rgb_gt.shape

    normal = model_outputs['normals']
    normal = normal.reshape(batch_size, num_samples, 3)
    render_rgb = model_outputs['color_fine']
    render_rgb = render_rgb.reshape(batch_size, num_samples, 3)

    normal = clip_img((normal + 1.) / 2.)
    render_rgb = clip_img(tonemap_img(render_rgb))
    ground_true = clip_img(tonemap_img(rgb_gt.cuda()))

    output_vs_gt = torch.cat((normal, render_rgb, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=1).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    print('saving render img to {0}/rendering_{1}.png'.format(path, iters))
    img.save('{0}/rendering_{1}.png'.format(path, iters))


tonemap_img = lambda x: torch.pow(x, 1. / 2.2)
clip_img = lambda x: torch.clamp(x, min=0., max=1.)


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
