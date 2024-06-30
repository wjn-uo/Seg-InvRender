import os
import logging

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
from net.sdfmodel.fields import SDFNetwork, RenderingNetwork
from net.sdfmodel.ray_tracing import RayTracing
from net.inverse_render.shadowrenderer import ShadowRenderer
from net.inverse_render.sg_envmap_material_cato import EnvmapMaterialNetwork
from net.inverse_render.loss import Loss

TINY_NUMBER = 1e-6


class Runner:
    def __init__(self, conf_path, case='CASE_NAME', is_continue=False):

        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        self.expname = case

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        self.sdf_exp_dir = self.conf['general.sdf_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = SceneDataset(self.conf['dataset'])
        self.data_dir = self.dataset.instance_dir
        self.total_pixels = self.dataset.total_pixels
        self.cur_iter = 0
        self.class_num = self.conf['dataset.class_num']

        # Training parameters
        self.end_epoch = self.conf.get_int('train.end_epoch')
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')

        self.is_continue = is_continue

        # Networks
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        self.envmap_material_network = EnvmapMaterialNetwork(**self.conf['model.envmap_material_network_cato']).to(
            self.device)
        self.ray_tracer = RayTracing(**self.conf['model.ray_tracer']).to(self.device)

        self.idr_optimizer = torch.optim.Adam(list(self.color_network.parameters()),
                                              lr=self.conf.get_float('train.idr_learning_rate'))
        self.idr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.idr_optimizer,
                                                                  self.conf.get_list('train.idr_sched_milestones',
                                                                                     default=[]),
                                                                  gamma=self.conf.get_float('train.idr_sched_factor',
                                                                                            default=0.0))

        self.sg_optimizer = torch.optim.Adam(list(self.envmap_material_network.parameters()),
                                             lr=self.conf.get_float('train.sg_learning_rate'))
        self.sg_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.sg_optimizer,
                                                                 self.conf.get_list('train.sg_sched_milestones',
                                                                                    default=[]),
                                                                 gamma=self.conf.get_float('train.sg_sched_factor',
                                                                                           default=0.0))

        self.loss = Loss(**self.conf['loss']).to(self.device)

        geo_model_name = None
        if True:
            model_list_raw = os.listdir(os.path.join(self.sdf_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth':
                    model_list.append(model_name)
            model_list.sort()
            geo_model_name = model_list[-1]

        if geo_model_name is not None:
            logging.info('Find checkpoint: {}'.format(geo_model_name))
            self.load_geo_checkpoint(geo_model_name)

        self.train_dataloader = torch.utils.data.DataLoader(self.dataset,
                                                            batch_size=1,
                                                            shuffle=True,
                                                            collate_fn=self.dataset.collate_fn,
                                                            generator=torch.Generator(device='cuda')
                                                            )

        self.plot_dataloader = torch.utils.data.DataLoader(self.dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           collate_fn=self.dataset.collate_fn,
                                                           generator=torch.Generator(device='cuda')
                                                           )

        self.n_batches = len(self.train_dataloader)

        self.renderer = ShadowRenderer(self.sdf_network,
                                       self.color_network,
                                       self.envmap_material_network,
                                       self.ray_tracer
                                       )

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
        mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)
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

                if self.cur_iter % self.val_freq == 0 and self.cur_iter != 0:
                    self.plot_to_disk()

                for key in model_input.keys():
                    model_input[key] = model_input[key].cuda()

                cato = ground_truth['label'].squeeze(0)[:, 0]
                cato = cato.long()
                cato_en = F.one_hot(cato, num_classes=self.class_num).cuda()

                render_out = self.renderer.render_core(model_input, cato_en)

                loss_output = self.loss(render_out, ground_truth, cato_en)

                loss = loss_output['loss']

                self.idr_optimizer.zero_grad()
                self.sg_optimizer.zero_grad()

                loss.backward()
                self.idr_optimizer.step()
                self.sg_optimizer.step()

                self.cur_iter += 1
                self.idr_scheduler.step()
                self.sg_scheduler.step()

                if self.cur_iter % self.report_freq == 0:
                    self.log(epoch, data_index, loss, loss_output, mse2psnr)

            self.start_epoch += 1

    def log(self, epoch, data_index, loss, loss_output, mse2psnr):
        print(
            '{} [{}/{}] ({}/{}): loss = {}, idr_rgb_loss = {}, sg_rgb_loss = {}, background_rgb_loss={},latent_smooth_loss = {}, cato_loss={}'
            ' idr_lr = {},sg_lr={},idr_psnr = {}, sg_psnr = {} '
            .format(self.expname, epoch, self.cur_iter, data_index, self.n_batches, loss.item(),
                    loss_output['idr_rgb_loss'].item(),
                    loss_output['sg_rgb_loss'].item(),
                    loss_output['background_rgb_loss'].item(),
                    loss_output['latent_smooth_loss'].item(),
                    loss_output['cato_loss'].item(),
                    self.idr_scheduler.get_lr()[0],
                    self.sg_scheduler.get_lr()[0],
                    mse2psnr(loss_output['idr_rgb_loss'].item()),
                    mse2psnr(loss_output['sg_rgb_loss'].item())))

        self.writer.add_scalar('idr_rgb_loss', loss_output['idr_rgb_loss'].item(), self.cur_iter)
        self.writer.add_scalar('idr_psnr', mse2psnr(loss_output['idr_rgb_loss'].item()), self.cur_iter)
        self.writer.add_scalar('sg_rgb_loss', loss_output['sg_rgb_loss'].item(), self.cur_iter)
        self.writer.add_scalar('sg_psnr', mse2psnr(loss_output['sg_rgb_loss'].item()), self.cur_iter)
        self.writer.add_scalar('latent_smooth_loss', loss_output['latent_smooth_loss'].item(), self.cur_iter)
        self.writer.add_scalar('background_rgb_loss', loss_output['background_rgb_loss'].item(), self.cur_iter)
        self.writer.add_scalar('cato_loss', loss_output['cato_loss'].item(), self.cur_iter)
        self.writer.add_scalar('idr_lrate', self.idr_scheduler.get_lr()[0], self.cur_iter)
        self.writer.add_scalar('sg_lrate', self.sg_scheduler.get_lr()[0], self.cur_iter)

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.envmap_material_network.load_state_dict(checkpoint['envmap_material_network'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.ray_tracer.load_state_dict(checkpoint['ray_tracer'])
        self.idr_optimizer.load_state_dict(checkpoint['idr_optimizer'])
        self.sg_optimizer.load_state_dict(checkpoint['sg_optimizer'])
        self.start_epoch = checkpoint['epoch']
        logging.info('End')

    def load_geo_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.sdf_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'envmap_material_network': self.envmap_material_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'ray_tracer': self.ray_tracer.state_dict(),
            'idr_optimizer': self.idr_optimizer.state_dict(),
            'sg_optimizer': self.sg_optimizer.state_dict(),
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

        cato = ground_truth['label'].squeeze(0)[:, 0]
        cato = cato.long()
        cato_en = F.one_hot(cato, num_classes=self.class_num).cuda().unsqueeze(0)

        model_input.update({"cato_en": cato_en})
        split = split_input(model_input, self.total_pixels)

        res = []
        for s in split:
            out = self.renderer.render_core(s, s['cato_en'])
            res.append({
                'normals': out['normals'].detach(),
                'object_mask': out['object_mask'].detach(),
                'roughness': out['roughness'].detach(),
                'diffuse_albedo': out['diffuse_albedo'].detach(),
                'indir_rgb': out['indir_rgb'].detach(),
                'idr_rgb': out['idr_rgb'].detach(),
                'sg_rgb': out['sg_rgb'].detach(),
                'vis_shadow': out['vis_shadow'].detach(),
                'sg_specular_rgb': out['sg_specular_rgb'].detach(),
                'sg_diffuse_rgb': out['sg_diffuse_rgb'].detach(),
            })

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = merge_output(res, self.total_pixels, batch_size)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        plots_dir = os.path.join(self.base_exp_dir,
                                 'validations_fine')

        plot_mat(
            model_outputs,
            ground_truth['rgb'],
            plots_dir,
            self.cur_iter,
            self.dataset.img_res,
        )

        self.dataset.sampling_idx = sampling_idx

        os.makedirs(os.path.join(self.base_exp_dir, 'envmap'), exist_ok=True)

        # log environment map
        lgtSGs = self.envmap_material_network.get_light()
        envmap = self.compute_envmap(lgtSGs=lgtSGs,
                                     H=256, W=512, upper_hemi=self.envmap_material_network.upper_hemi)
        envmap = envmap.cpu().numpy()
        im = np.power(envmap, 1. / 2.2)
        im = np.clip(im, 0., 1.)
        im = np.uint8(im * 255.)
        imageio.imwrite(os.path.join(self.base_exp_dir,
                                     'envmap', 'envmap_{}.png'.format(self.cur_iter)), im)

    def compute_envmap(self, lgtSGs, H, W, upper_hemi=False):
        # same convetion as blender
        if upper_hemi:
            phi, theta = torch.meshgrid([torch.linspace(0., np.pi / 2., H),
                                         torch.linspace(1.0 * np.pi, -1.0 * np.pi, W)])
        else:
            phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H),
                                         torch.linspace(1.0 * np.pi, -1.0 * np.pi, W)])
        viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi),
                                torch.sin(theta) * torch.sin(phi),
                                torch.cos(phi)], dim=-1)  # [H, W, 3]

        rgb = self.render_envmap_sg(lgtSGs, viewdirs)
        envmap = rgb.reshape((H, W, 3))
        return envmap

    def render_envmap_sg(self, lgtSGs, viewdirs):
        viewdirs = viewdirs.to(lgtSGs.device)
        viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

        # [M, 7] ---> [..., M, 7]
        dots_sh = list(viewdirs.shape[:-2])
        M = lgtSGs.shape[0]
        lgtSGs = lgtSGs.view([1, ] * len(dots_sh) + [M, 7]).expand(dots_sh + [M, 7])

        lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
        lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
        lgtSGMus = torch.abs(lgtSGs[..., -3:])
        # [..., M, 3]
        rgb = lgtSGMus * torch.exp(lgtSGLambdas * \
                                   (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
        rgb = torch.sum(rgb, dim=-2)  # [..., 3]
        return rgb


def split_input(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 1000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        data['cato_en'] = torch.index_select(model_input['cato_en'], 1, indx)
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


def plot_mat(model_outputs, rgb_gt, path, iters, img_res):
    ''' write inverse rendering result '''

    batch_size, num_samples, _ = rgb_gt.shape

    normal = model_outputs['normals']
    normal = normal.reshape(batch_size, num_samples, 3)

    specular_rgb = model_outputs['sg_specular_rgb']
    specular_rgb = specular_rgb.reshape(batch_size, num_samples, 3)

    diffuse_rgb = model_outputs['sg_diffuse_rgb']
    diffuse_rgb = diffuse_rgb.reshape(batch_size, num_samples, 3)

    sg_rgb = model_outputs['sg_rgb']
    sg_rgb = sg_rgb.reshape(batch_size, num_samples, 3)
    idr_rgb = model_outputs['idr_rgb']
    idr_rgb = idr_rgb.reshape(batch_size, num_samples, 3)

    indir_rgb = model_outputs['indir_rgb']
    indir_rgb = indir_rgb.reshape(batch_size, num_samples, 3)

    roughness = model_outputs['roughness'].reshape(batch_size, num_samples, 3)
    diffuse_albedo = model_outputs['diffuse_albedo'].reshape(batch_size, num_samples, 3)
    visibility = model_outputs['vis_shadow'].reshape(batch_size, num_samples, 3)

    # plot rendered images
    plot_materials(normal, rgb_gt,
                   visibility, diffuse_albedo, roughness, specular_rgb, diffuse_rgb,
                   indir_rgb, idr_rgb, sg_rgb, path, iters, img_res)


def plot_materials(normal, ground_true,
                   visibility, diffuse_albedo, roughness, specular_rgb, diffuse_rgb,
                   indir_rgb, idr_rgb, sg_rgb, path, iters, img_res):
    normal = clip_img((normal + 1.) / 2.)
    specular_rgb = clip_img(tonemap_img(specular_rgb))
    diffuse_rgb = clip_img(tonemap_img(diffuse_rgb))
    indir_rgb = clip_img(tonemap_img(indir_rgb))
    sg_rgb = clip_img(tonemap_img(sg_rgb))
    idr_rgb = clip_img(tonemap_img(idr_rgb))
    diffuse_albedo = clip_img(tonemap_img(diffuse_albedo))
    ground_true = clip_img(tonemap_img(ground_true.cuda()))

    output_vs_gt = torch.cat((normal, visibility, diffuse_albedo, roughness,
                              specular_rgb, diffuse_rgb, indir_rgb, idr_rgb, sg_rgb, ground_true), dim=0)
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
    runner = Runner(args.conf, args.case, args.is_continue)
    runner.train()
