import torch
from torch import nn
from torch.nn import functional as F
from net.sdfmodel.embedder import get_embedder


class Loss(nn.Module):
    def __init__(self, idr_rgb_weight,
                 sg_rgb_weight, latent_smooth_weight, cato_loss_weight, background_rgb_weight,
                 loss_type='L1'):
        super().__init__()
        self.idr_rgb_weight = idr_rgb_weight
        self.background_rgb_weight = background_rgb_weight
        self.sg_rgb_weight = sg_rgb_weight
        self.latent_smooth_weight = latent_smooth_weight
        self.cato_loss_weight = cato_loss_weight

        self.env_loss = nn.MSELoss(reduction='mean')

        if loss_type == 'L1':
            print('Using L1 loss for comparing images!')
            self.img_loss = nn.L1Loss(reduction='sum')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing images!')
            self.img_loss = nn.MSELoss(reduction='sum')
        else:
            raise Exception('Unknown loss_type!')

    def get_rgb_loss(self, rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.img_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_latent_smooth_loss(self, model_outputs, network_object_mask, object_mask):
        mask = (network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        d_diff = model_outputs['diffuse_albedo'][mask]
        d_rough = model_outputs['roughness'][..., 0][mask]
        d_xi_diff = model_outputs['random_xi_diffuse_albedo'][mask]
        d_xi_rough = model_outputs['random_xi_roughness'][..., 0][mask]
        loss = nn.L1Loss()(d_diff, d_xi_diff) + nn.L1Loss()(d_rough, d_xi_rough)
        return loss

    def get_cato_loss(self, value_raw, onehot_raw, network_object_mask, object_mask):

        mask = (network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()

        value = value_raw[mask]
        onehot = onehot_raw[mask]
        catoloss = 0.0
        for i in range(1, onehot.shape[-1]):
            cmask = onehot[:, i] > 0
            cmask_sum = cmask.float().sum()
            if cmask_sum != 0:
                mean = torch.mean(value[cmask], dim=0).unsqueeze(0).expand(value[cmask].shape[0], -1)
                catoloss += nn.MSELoss()(value[cmask], mean)

        return catoloss

    def get_background_rgb_loss(self, sg_rgb_values, rgb_gt, network_object_mask, object_mask):
        mask = (~network_object_mask) & (~object_mask)
        if self.background_rgb_weight <= 0 or mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()

        sg_rgb_values = sg_rgb_values[mask].reshape((-1, 3))
        rgb_gt = rgb_gt.reshape(-1, 3)[mask].reshape((-1, 3))

        sg_rgb_loss = self.env_loss(sg_rgb_values, rgb_gt)

        return sg_rgb_loss

    def forward(self, model_outputs, ground_truth, cato_en):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        pred_rgb = model_outputs['sg_rgb']
        idr_rgb = model_outputs['idr_rgb']
        sg_rgb_loss = self.get_rgb_loss(pred_rgb, rgb_gt, network_object_mask, object_mask)
        idr_rgb_loss = self.get_rgb_loss(idr_rgb, rgb_gt, network_object_mask, object_mask)

        latent_smooth_loss = self.get_latent_smooth_loss(model_outputs, network_object_mask, object_mask)
        cato_loss = self.get_cato_loss(model_outputs['diffuse_albedo'], cato_en, network_object_mask, object_mask)
        background_rgb_loss = self.get_background_rgb_loss(model_outputs['sg_rgb'], rgb_gt, network_object_mask,
                                                           object_mask)
        loss = self.sg_rgb_weight * sg_rgb_loss + \
               self.idr_rgb_weight * idr_rgb_loss + \
               self.latent_smooth_weight * latent_smooth_loss + \
               self.background_rgb_weight * background_rgb_loss + self.cato_loss_weight * cato_loss

        output = {
            'sg_rgb_loss': sg_rgb_loss,
            'idr_rgb_loss': idr_rgb_loss,
            'background_rgb_loss': background_rgb_loss,
            'cato_loss': cato_loss,
            'latent_smooth_loss': latent_smooth_loss,
            'loss': loss}

        return output
