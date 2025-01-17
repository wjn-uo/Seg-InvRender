import numpy as np
import imageio

imageio.plugins.freeimage.download()
import skimage
import cv2
import torch
from torch.nn import functional as F
import traceback
from scipy import linalg
from PIL import Image
from torchvision import transforms as T


def load_rgb(path):
    img = imageio.imread(path)[:, :, :3]
    img = np.float32(img)
    if not path.endswith('.exr'):
        img = img / 255.

    return img


def load_mask(path):
    alpha = imageio.imread(path, as_gray=True)
    alpha = np.float32(alpha) / 255.
    object_mask = alpha > 0.5

    return object_mask


def factorize(P):
    P = P[:3, :4]

    # RQ factorize the submatrix
    K, R = linalg.rq(P[:3, :3])

    # fix the intrinsic and rotation matrix
    ##### intrinsic matrix's diagonal entries must be all positive
    ##### rotation matrix's determinant must be 1; otherwise there's an reflection component
    neg_sign_cnt = int(K[0, 0] < 0) + int(K[1, 1] < 0) + int(K[2, 2] < 0)
    if neg_sign_cnt == 1 or neg_sign_cnt == 3:
        K = -K
        R = -R

    new_neg_sign_cnt = int(K[0, 0] < 0) + int(K[1, 1] < 0) + int(K[2, 2] < 0)
    assert (new_neg_sign_cnt == 0 or new_neg_sign_cnt == 2)

    fix = np.diag((1, 1, 1))
    if K[0, 0] < 0 and K[1, 1] < 0:
        fix = np.diag((-1, -1, 1))
    elif K[0, 0] < 0 and K[2, 2] < 0:
        fix = np.diag((-1, 1, -1))
    elif K[1, 1] < 0 and K[2, 2] < 0:
        fix = np.diag((1, -1, -1))
    K = K @ fix
    R = fix @ R

    # normalize the K matrix
    scale = K[2, 2]
    K /= scale
    P[:3, :4] /= scale
    # normalize the sign of R's determinant
    R_det = np.linalg.det(R)
    if R_det < 0.:
        R *= -1.
        P[:3, :4] *= -1.

    t = linalg.lstsq(K, P[:3, 3:4])[0]

    return K, R, t


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    K, R, t = factorize(P)
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K.astype(np.float32)

    W2C = np.eye(4)
    W2C[:3, :4] = np.hstack((R, t))
    pose = np.linalg.inv(W2C).astype(np.float32)
    return intrinsics, pose


def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:, :4])
        p = torch.eye(4).repeat(pose.shape[0], 1, 1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else:  # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc


def get_camera_for_plot(pose):
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:].detach()
        R = quat_to_rot(pose[:, :4].detach())
    else:  # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        R = pose[:, :3, :3]
    cam_dir = R[:, :3, 2]
    return cam_loc, cam_dir


def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1) - sk.unsqueeze(
        -1) * y / fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)


def project(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    u = x / z * fx.unsqueeze(-1) + cx.unsqueeze(-1) - cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(
        -1) + sk.unsqueeze(-1) * y / fy.unsqueeze(-1)
    v = y / z * fy.unsqueeze(-1) + cy.unsqueeze(-1)

    # homogeneous
    return torch.stack((u, v), dim=-1)


def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3)).cuda()
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj ** 2 + qk ** 2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi ** 2 + qk ** 2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi ** 2 + qj ** 2)
    return R


def rot_to_quat(R):
    batch_size, _, _ = R.shape
    q = torch.ones((batch_size, 4)).cuda()

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 0] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[:, 1] = (R21 - R12) / (4 * q[:, 0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def points2uv(points, pose, intrinsics):
    # transfer points from world coordinate to camera coordinate
    batch_size, num_samples, _ = points.shape
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:, :4])
        pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda().float()
        pose[:, :3, 3] = cam_loc
        pose[:, :3, :3] = R

    points_hom = torch.cat((points, torch.ones((batch_size, num_samples, 1)).cuda()), dim=2)

    points_hom = points_hom.permute(0, 2, 1)
    points_cam = torch.inverse(pose).bmm(points_hom)  # NxRx4
    points_cam = points_cam.permute(0, 2, 1)

    # project
    x = points_cam[:, :, 0]  # NxR
    y = points_cam[:, :, 1]  # NxR
    z = points_cam[:, :, 2]  # NxR
    uv = project(x, y, z, intrinsics)  # NxRx2

    return uv
