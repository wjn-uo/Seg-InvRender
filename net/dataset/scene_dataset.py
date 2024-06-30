import os
from glob import glob

import torch
import numpy as np
import torchvision

from net.utils import rend_util
import json
import cv2 as cv


def read_cam_dict(cam_dict_file):
    with open(cam_dict_file) as fp:
        cam_dict = json.load(fp)
        for x in sorted(cam_dict.keys()):
            K = np.array(cam_dict[x]['K']).reshape((4, 4))
            W2C = np.array(cam_dict[x]['W2C']).reshape((4, 4))
            C2W = np.linalg.inv(W2C)

            cam_dict[x]['K'] = K
            cam_dict[x]['W2C'] = W2C
            cam_dict[x]['C2W'] = C2W
    return cam_dict


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG', '*.exr']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 conf
                 ):

        self.instance_dir = conf.get_string('data_dir')
        print('Creating dataset from: ', self.instance_dir)
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.gamma = conf.get_float('gamma', default=2.2)
        self.train_cameras = conf.get_bool('train_cameras', default=False)
        self.subsample = conf.get_int('subsample', default=1)
        wo_mask = conf.get_float('wo_mask', default=False)

        self.label_list = conf.get_list('label', default=[])

        image_dir = os.path.join(self.instance_dir, 'image')
        image_paths = sorted(glob_imgs(image_dir))

        mask_dir = os.path.join(self.instance_dir, 'mask')
        mask_paths = sorted(glob_imgs(mask_dir))

        label_dir = os.path.join(self.instance_dir, 'label')
        label_paths = sorted(glob_imgs(label_dir))

        cam_dict = read_cam_dict(os.path.join(self.instance_dir, 'cam_dict_norm.json'))
        print('Found # images, # masks, # cameras: ', len(image_paths), len(mask_paths), len(cam_dict))
        self.n_cameras = int(len(image_paths))
        self.image_paths = image_paths

        self.single_imgname = None
        self.single_imgname_idx = None
        self.sampling_idx = None
        self.sampling_rays = None

        self.intrinsics_all = []
        self.pose_all = []
        for x in sorted(cam_dict.keys()):
            intrinsics = cam_dict[x]['K'].astype(np.float32)
            pose = cam_dict[x]['C2W'].astype(np.float32)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        if len(image_paths) > 0:
            assert (len(image_paths) == self.n_cameras)
            self.has_groundtruth = True
            H, W = rend_util.load_rgb(image_paths[0]).shape[:2]
            self.img_res = [H, W]
            self.total_pixels = self.img_res[0] * self.img_res[1]
            self.rgb_images = []
            print('Applying inverse gamma correction: ', self.gamma)
            for path in image_paths:
                rgb = rend_util.load_rgb(path)
                rgb = np.power(rgb, self.gamma)
                rgb = rgb.reshape(-1, 3)
                self.rgb_images.append(torch.from_numpy(rgb).float())
        else:
            self.has_groundtruth = False
            K = cam_dict.values()[0]['K']  # infer image resolution from camera mat
            W = int(2. / K[0, 0])
            H = int(2. / K[1, 1])
            print('No ground-truth images available. Image resolution of predicted images: ', H, W)
            self.img_res = [H, W]
            self.total_pixels = self.img_res[0] * self.img_res[1]
            self.rgb_images = [torch.ones((self.total_pixels, 3), dtype=torch.float32), ] * self.n_cameras

        if len(mask_paths) > 0 and not wo_mask:
            assert (len(mask_paths) == self.n_cameras)
            self.has_surfpts = True
            self.object_masks = []
            for path in mask_paths:
                object_mask = rend_util.load_mask(path)
                # print('Loaded mask: ', path)
                object_mask = object_mask.reshape(-1)
                self.object_masks.append(torch.from_numpy(object_mask).bool())
        else:
            self.object_masks = [torch.ones((self.total_pixels,)).bool(), ] * self.n_cameras

        if len(label_paths) > 0:
            assert (len(label_paths) == self.n_cameras)
            self.label = []
            for path in label_paths:
                label_png = cv.imread(path).reshape(-1, 3)
                for i in range(len(self.label_list)):
                    label_png[label_png[:] == self.label_list[i]] = i
                self.label.append(torch.from_numpy(label_png).float())
        else:
            self.label = [torch.ones((self.total_pixels, 3), dtype=torch.float32), ] * self.n_cameras

        if self.subsample is not None and self.subsample != 1:
            print("resizing data with subsample=", self.subsample)
            self.resize()

    def resize(self):
        # update config
        old_img_res = (self.img_res[0], self.img_res[1])
        new_img_res = (int(old_img_res[0] * self.subsample), int(old_img_res[1] * self.subsample))
        self.img_res = [new_img_res[0], new_img_res[1]]
        self.total_pixels = self.img_res[0] * self.img_res[1]

        scale = max(new_img_res) / max(old_img_res)

        # resize intrinsics
        for i in range(len(self.intrinsics_all)):
            intrinsics = self.intrinsics_all[i]
            intrinsics[0, 0] *= scale
            intrinsics[0, 2] *= scale
            intrinsics[1, 1] *= scale
            intrinsics[1, 2] *= scale

        # resize image
        rgb_resizer = torchvision.transforms.Resize(new_img_res, antialias=True)
        for i in range(len(self.rgb_images)):
            rgb_img = self.rgb_images[i].reshape(old_img_res[0], old_img_res[1], 3)  # HxWxC
            rgb_img = rgb_img.permute(2, 0, 1)  # CxHxW
            rgb_img_new = rgb_resizer(rgb_img)
            self.rgb_images[i] = rgb_img_new.reshape(3, -1).transpose(1, 0).float()  # H*W x C

        # resize mask
        for i in range(len(self.object_masks)):
            mask = self.object_masks[i]
            mask_img = mask.float().reshape(1, old_img_res[0], old_img_res[1])
            mask_img_new = rgb_resizer(mask_img)  # 1, new_img_res[0], new_img_res[1]
            mask_img_new = mask_img_new > 0.5
            self.object_masks[i] = mask_img_new.reshape(-1).bool()

    def __len__(self):
        return self.n_cameras

    def return_single_img(self, img_name):
        self.single_imgname = img_name
        for idx in range(len(self.image_paths)):
            if os.path.basename(self.image_paths[idx]) == self.single_imgname:
                self.single_imgname_idx = idx
                break
        print('Always return: ', self.single_imgname, self.single_imgname_idx)

    def img_idx(self, idx):
        self.single_imgname_idx = idx

    def __getitem__(self, idx):
        if self.single_imgname_idx is not None:
            idx = self.single_imgname_idx

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)  # W*H x 2

        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "label": self.label[idx]
        }
        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["label"] = self.label[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        sample["uv"] = self.ray_sample(sample["uv"])

        return idx, sample, ground_truth

    def ray_sample(self, s_uv):
        # s_uv: Sx2
        if self.sampling_rays is not None:
            # self.sample_rays [rays x 2]
            sample_rays_inter = self.sampling_rays[None, ...].to(s_uv.device)  # 1xRx2
            s_uv = s_uv[:, None, ...]  # S x 1 x 2
            s_uv = s_uv + sample_rays_inter  # S x R x 2
        return s_uv

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def input_at_image(self, idx):

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)  # W*H x 2

        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
        }

        sample["uv"] = self.ray_sample(sample["uv"])
        return sample
