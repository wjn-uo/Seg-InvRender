import logging
import os

import cv2

logging.getLogger('PIL').setLevel(logging.WARNING)

import argparse

import imageio
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from pyhocon import ConfigFactory
from net.dataset.scene_dataset import SceneDataset
from net.sdfmodel.fields import SDFNetwork
from net.sdfmodel.ray_tracing import RayTracing
from net.sdfmodel.surfpts import Surfpts
from net.utils import rend_util

TINY_NUMBER = 1e-6


def sec_hit(sdf_network, ray_tracer, label_pts, label_mask, cam_loc):
    label_mask = label_mask.squeeze(0)
    rays_o = label_pts[label_mask]
    rays_d = cam_loc - rays_o
    rays_d = F.normalize(rays_d, dim=1)

    object_mask = torch.ones(rays_o.shape[0]).bool()
    mask = torch.ones(rays_o.shape[0]).bool()
    chunk_size = 10000
    chunk_idxs_vis_compute = torch.split(torch.arange(rays_o.shape[0]), chunk_size)
    for chunk_idx in chunk_idxs_vis_compute:
        with torch.no_grad():
            _, surf_mask_intersect, _ = ray_tracer(
                sdf=lambda x: sdf_network(x)[:, :1], cam_loc=rays_o[chunk_idx],
                object_mask=object_mask[chunk_idx],
                ray_directions=rays_d[chunk_idx])  # n_rays*n_samples,
        mask[chunk_idx] = surf_mask_intersect
    return mask


class Runner:
    def __init__(self, conf_path, case='CASE_NAME'):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.sdf_exp_dir = self.conf['general.base_exp_dir']
        self.label_dir = self.conf['general.label_dir']
        os.makedirs(self.label_dir, exist_ok=True)
        self.dataset = SceneDataset(self.conf['dataset'])
        self.data_dir = self.dataset.instance_dir
        self.total_pixels = self.dataset.total_pixels

        # Networks
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.ray_tracer = RayTracing(**self.conf['model.ray_tracer']).to(self.device)

        self.surfpts = Surfpts(self.sdf_network,
                               self.ray_tracer
                               )

        if os.path.exists(self.sdf_exp_dir):
            model_list_raw = os.listdir(os.path.join(self.sdf_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth':
                    model_list.append(model_name)
            model_list.sort()
            geo_model_name = model_list[-1]
            logging.info('Find checkpoint: {}'.format(geo_model_name))
            self.load_geo_checkpoint(geo_model_name)
        else:
            print('No illum_model pretrain, please train it first!')
            exit(0)

        label_input_dir = os.path.join(self.conf['dataset.data_dir'], 'label_input')
        label_paths = os.listdir(label_input_dir)
        label_paths.sort(key=lambda x: int(x[0:-4]))

        label = []
        label_input = []
        if len(label_paths) > 0:
            for path in label_paths:
                label_png = imageio.imread(os.path.join(label_input_dir, path))[:, :, :3]
                label_png = np.float32(label_png).reshape(-1, 3)
                label.append(torch.from_numpy(label_png).float().to(self.device))
                label_input.append(self.dataset.input_at_image(int(path[0:-4])))

        for i in range(len(label)):
            for key in label_input[i].keys():
                label_input[i][key] = label_input[i][key].unsqueeze(0).cuda()

        self.label = torch.zeros(self.dataset.n_cameras, self.dataset.img_res[0], self.dataset.img_res[1], 3)
        for i in range(0, len(label)):
            for j in range(0, self.dataset.n_cameras):

                split = split_input(label_input[i], self.total_pixels)
                pts = []
                for s in split:
                    out = self.surfpts.render_core(s)
                    pts.append(out['points'])
                all_pts = torch.cat(pts, dim=0)
                mask = label_input[i]['object_mask'].squeeze(0)

                sec_mask = sec_hit(self.sdf_network, self.ray_tracer, all_pts, mask.bool(),
                                   self.dataset.pose_all[j][:3, 3].to(self.device))

                label_pts = all_pts[mask.bool().squeeze(0)][~sec_mask].unsqueeze(0)

                label_uv = rend_util.points2uv(label_pts, self.dataset.pose_all[j].to(self.device).unsqueeze(0),
                                               self.dataset.intrinsics_all[j].to(self.device).unsqueeze(0))
                label_uv = torch.round(label_uv.squeeze()).long()
                label_vu = torch.zeros_like(label_uv)
                label_vu[:, 0] = label_uv[:, 1]
                label_vu[:, 1] = label_uv[:, 0]

                if (torch.count_nonzero(~sec_mask)):
                    slices = [label_vu[:, k] for k in range(2)]
                    slices = slices + [Ellipsis]

                    self.label[j][slices] = label[i].float()[mask.bool().squeeze(0)][~sec_mask]

                wlabel = self.label[j]
                wlabel = wlabel.reshape(self.dataset.img_res[0], self.dataset.img_res[1], 3)
                wlabel = np.uint8(wlabel.cpu().numpy())

                os.makedirs(os.path.join(self.label_dir, '{}'.format(i)), exist_ok=True)

                imageio.imwrite(os.path.join(self.label_dir,
                                             '{}'.format(i), '{:0>3d}.png'.format(j)), wlabel)

        mask_dir = os.path.join(self.conf['dataset.data_dir'], 'mask')
        lable_list = os.listdir(self.label_dir)
        lable_list.sort(key=lambda x: int(x))
        mask_list = os.listdir(mask_dir)
        mask_list.sort(key=lambda x: int(x[0:-4]))
        lable = []
        mask = []
        for i, image_path in enumerate(mask_list):
            mask.append(cv.imread(os.path.join(mask_dir, image_path)))
        for i, label_path in enumerate(lable_list):
            lable_tmp = []
            img_list = os.listdir(os.path.join(self.label_dir, label_path))
            img_list.sort(key=lambda x: int(x[0:-4]))
            for j, image_path in enumerate(img_list):
                img = cv.imread(os.path.join(os.path.join(self.label_dir, label_path), image_path))
                lable_tmp.append(img)
            lable.append(lable_tmp)

        out_dir = os.path.join(self.data_dir, 'label')
        os.makedirs(out_dir, exist_ok=True)

        label_img = np.zeros_like(mask)
        for i in range(self.dataset.n_cameras):
            m = mask[i] > 0
            for j in range(len(label)):
                mm = lable[j][i] > 0
                label_img[i][m & mm] = lable[j][i][m & mm]

            # median_fliter
            blurred_image = cv2.medianBlur(label_img[i], ksize=3)

            cv.imwrite(os.path.join(out_dir, '{:0>3d}.png'.format(i)), blurred_image)

    def load_geo_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.sdf_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)

        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])


def split_input(model_input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    n_pixels = 20000
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        # data['encoder'] = torch.index_select(model_input['encoder'], 1, indx)
        split.append(data)
    return split


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
    runner = Runner(args.conf, args.case)
