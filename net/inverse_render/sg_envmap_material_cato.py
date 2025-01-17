import os

import imageio
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from net.sdfmodel.embedder import get_embedder


### uniformly distribute points on a sphere
def fibonacci_sphere(samples=1):
    '''
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    :param samples:
    :return:
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points


def compute_energy(lgtSGs):
    lgtLambda = torch.abs(lgtSGs[:, 3:4])  # [M, 1]
    lgtMu = torch.abs(lgtSGs[:, 4:])  # [M, 3]
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy


class EnvmapMaterialNetwork(nn.Module):
    def __init__(self, multires=0, dims=[256, 256, 256],
                 white_specular=False,
                 white_light=False,
                 num_lgt_sgs=32,
                 num_base_materials=2,
                 upper_hemi=False,
                 fix_specular_albedo=False,
                 specular_albedo=[-1., -1., -1.],
                 init_specular_reflectance=-1,
                 correct_normal=False,
                 roughness_mlp=False,
                 specular_mlp=False,
                 same_mlp=False,
                 dims_roughness=[256, 256, 256],
                 dims_specular=[256, 256, 256],
                 feature_vector_size=256,
                 num_classes=8,
                 use_normal=False,
                 light_type='sg'):
        super().__init__()

        self.correct_normal = correct_normal
        self.roughness_mlp = roughness_mlp
        self.specular_mlp = specular_mlp
        self.same_mlp = same_mlp
        self.feature_vector_size = feature_vector_size
        self.num_classes = num_classes
        self.fix_specular_albedo = fix_specular_albedo
        self.fake_roughness = False
        self.fake_specular = False
        self.light_type = light_type

        self.envmap = None

        input_dim = 3
        self.embed_fn = None
        if multires > 0:
            self.embed_fn, input_dim = get_embedder(multires)

        self.use_normal = use_normal
        if use_normal:
            input_dim += 3

        input_dim += self.feature_vector_size
        input_dim += self.num_classes

        # self.actv_fn = nn.ReLU()
        self.actv_fn = nn.ELU()
        # self.actv_fn = nn.LeakyReLU(0.05)
        ############## spatially-varying diffuse albedo############
        print('Diffuse albedo network size: ', dims)
        diffuse_albedo_layers = []
        dim = input_dim
        dim_o = 3
        if self.roughness_mlp and self.same_mlp: dim_o += 1
        if not self.fix_specular_albedo and self.specular_mlp and self.same_mlp: dim_o += 1
        for i in range(len(dims)):
            diffuse_albedo_layers.append(nn.Linear(dim, dims[i]))
            diffuse_albedo_layers.append(self.actv_fn)
            dim = dims[i]
        diffuse_albedo_layers.append(nn.Linear(dim, dim_o))

        self.diffuse_albedo_layers = nn.Sequential(*diffuse_albedo_layers)
        # self.diffuse_albedo_layers.apply(weights_init)

        if self.correct_normal:
            ############## spatially-varying normal############
            print('Delta normal network size: ', dims)
            delta_normal_layers = []
            dim = input_dim
            for i in range(len(dims)):
                delta_normal_layers.append(nn.Linear(dim, dims[i]))
                delta_normal_layers.append(self.actv_fn)
                dim = dims[i]
            delta_normal_layers.append(nn.Linear(dim, 2))

            self.delta_normal_layers_layers = nn.Sequential(*delta_normal_layers)
        else:
            self.delta_normal_layers_layers = None

        ##################### specular rgb ########################
        self.numLgtSGs = num_lgt_sgs
        self.numBrdfSGs = num_base_materials
        print('Number of Light SG: ', self.numLgtSGs)
        print('Number of BRDF SG: ', self.numBrdfSGs)
        if self.light_type == "sg":
            # by using normal distribution, the lobes are uniformly distributed on a sphere at initialization
            self.white_light = white_light
            if self.white_light:
                print('Using white light!')
                self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 5),
                                           requires_grad=True)  # [M, 5]; lobe + lambda + mu
                # self.lgtSGs.data[:, -1] = torch.clamp(torch.abs(self.lgtSGs.data[:, -1]), max=0.01)
            else:
                self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7),
                                           requires_grad=True)  # [M, 7]; lobe + lambda + mu
                self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))
                # self.lgtSGs.data[:, -3:] = torch.clamp(torch.abs(self.lgtSGs.data[:, -3:]), max=0.01)

            # make sure lambda is not too close to zero
            self.lgtSGs.data[:, 3:4] = 10. + torch.abs(self.lgtSGs.data[:, 3:4] * 20.)
            # make sure total energy is around 1.
            energy = compute_energy(self.lgtSGs.data)
            # print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())
            self.lgtSGs.data[:, 4:] = torch.abs(self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0,
                                                                                     keepdim=True) * 2. * np.pi * 0.8
            energy = compute_energy(self.lgtSGs.data)
            print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

            # deterministicly initialize lobes
            lobes = fibonacci_sphere(self.numLgtSGs).astype(np.float32)
            self.lgtSGs.data[:, :3] = torch.from_numpy(lobes)
            # check if lobes are in upper hemisphere
            self.upper_hemi = upper_hemi
            if self.upper_hemi:
                print('Restricting lobes to upper hemisphere!')
                self.restrict_lobes_upper = lambda lgtSGs: torch.cat(
                    (lgtSGs[..., :1], torch.abs(lgtSGs[..., 1:2]), lgtSGs[..., 2:]), dim=-1)

                # limit lobes to upper hemisphere
                self.lgtSGs.data = self.restrict_lobes_upper(self.lgtSGs.data)
        else:
            self.upper_hemi = False
            self.white_light = False
            self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, self.numLgtSGs, 3), requires_grad=True)  # [M, M, 3]
            self.lgtSGs.data = torch.abs(self.lgtSGs.data)
            # self.lgtSGs.data = self.lgtSGs.data / torch.sum(self.lgtSGs.data.reshape(-1, 3), dim=0).reshape(1, 1, 3)

        self.white_specular = white_specular
        if self.fix_specular_albedo:
            print('Fixing specular albedo: ', specular_albedo)
            specular_albedo = np.array(specular_albedo).astype(np.float32)
            assert (self.numBrdfSGs == 1)
            assert (np.all(np.logical_and(specular_albedo > 0., specular_albedo < 1.)))
            self.specular_reflectance = nn.Parameter(torch.from_numpy(specular_albedo).reshape((self.numBrdfSGs, 3)),
                                                     requires_grad=False)  # [K, 1]
        else:
            if not self.specular_mlp:
                if self.white_specular:
                    print('Using white specular reflectance!')
                    self.specular_reflectance = nn.Parameter(torch.randn(self.numBrdfSGs, 1),
                                                             requires_grad=True)  # [K, 1]
                else:
                    self.specular_reflectance = nn.Parameter(torch.randn(self.numBrdfSGs, 3),
                                                             requires_grad=True)  # [K, 3]
                self.specular_reflectance.data = torch.abs(self.specular_reflectance.data)

                if init_specular_reflectance > 0:
                    self.specular_reflectance.data[:] = np.log(1 / (1 - init_specular_reflectance) - 1)
                    print('init specular_reflectance manually!')

                print('init specular_reflectance: ', 1.0 / (1.0 + np.exp(-self.specular_reflectance.data)))
            elif not self.same_mlp:
                output_specular_dim = 1 if self.white_specular else 3

                ############## spatially-varying specular############
                print('specular network size: ', dims_specular)
                specular_layers = []
                dim = input_dim
                for i in range(len(dims_specular)):
                    specular_layers.append(nn.Linear(dim, dims_specular[i]))
                    specular_layers.append(self.actv_fn)
                    dim = dims_specular[i]
                specular_layers.append(nn.Linear(dim, output_specular_dim))
                specular_layers.append(nn.Sigmoid())

                self.specular_layers = nn.Sequential(*specular_layers)

        if not self.roughness_mlp:
            if self.numBrdfSGs > 1:
                roughness = [np.random.uniform(-1.5, 2.0) for i in range(self.numBrdfSGs)]
            else:
                # optimize
                # roughness = [np.random.uniform(-1.5, -1.0) for i in range(self.numBrdfSGs)]       # small roughness
                roughness = [np.random.uniform(1.5, 2.0) for i in range(self.numBrdfSGs)]  # big roughness
            roughness = np.array(roughness).astype(dtype=np.float32).reshape((self.numBrdfSGs, 1))  # [K, 1]
            print('init roughness: ', 1.0 / (1.0 + np.exp(-roughness)))
            self.roughness = nn.Parameter(torch.from_numpy(roughness),
                                          requires_grad=True)
        elif not self.same_mlp:
            ############## spatially-varying roughness############
            print('roughness network size: ', dims_roughness)
            roughness_layers = []
            dim = input_dim
            for i in range(len(dims_roughness)):
                roughness_layers.append(nn.Linear(dim, dims_roughness[i]))
                roughness_layers.append(self.actv_fn)
                dim = dims_roughness[i]
            roughness_layers.append(nn.Linear(dim, 1))
            roughness_layers.append(nn.Sigmoid())

            self.roughness_layers = nn.Sequential(*roughness_layers)

        # blending weights
        self.blending_weights_layers = []
        if self.numBrdfSGs > 1:
            dim = input_dim
            for i in range(3):
                self.blending_weights_layers.append(nn.Sequential(nn.Linear(dim, 256), self.actv_fn))
                dim = 256
            self.blending_weights_layers.append(nn.Linear(dim, self.numBrdfSGs))
            self.blending_weights_layers = nn.Sequential(*self.blending_weights_layers)

    def get_light(self):
        lgtSGs = self.lgtSGs.clone().detach()
        if self.white_light:
            lgtSGs = torch.cat((lgtSGs, lgtSGs[..., -1:], lgtSGs[..., -1:]), dim=-1)
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        return lgtSGs

    def load_light(self, path):
        sg_path = os.path.join(path, 'sg_128.npy')
        device = self.lgtSGs.data.device
        load_sgs = torch.from_numpy(np.load(sg_path)).to(device)
        self.lgtSGs.data = load_sgs

        energy = compute_energy(self.lgtSGs.data)
        print('loaded envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        envmap_path = path + '.exr'
        envmap = np.float32(imageio.imread(envmap_path)[:, :, :3])
        self.envmap = torch.from_numpy(envmap).to(device)

    def get_base_materials(self):
        if not self.roughness_mlp:
            roughness = torch.sigmoid(self.roughness.clone().detach())
        else:
            roughness = torch.zeros(1, 1)

        if self.fix_specular_albedo:
            specular_reflectacne = self.specular_reflectance
        else:
            if not self.specular_mlp:
                specular_reflectacne = torch.sigmoid(self.specular_reflectance.clone().detach())
                if self.white_specular:
                    specular_reflectacne = specular_reflectacne.expand((-1, 3))  # [K, 3]
            else:
                specular_reflectacne = torch.zeros(1, 3)
        return roughness, specular_reflectacne

    def forward(self, points, feature_vector=None, cato=None):

        if self.embed_fn is not None:
            points = self.embed_fn(points)
        if feature_vector is not None:
            points = torch.cat([points, feature_vector], dim=-1)
        if cato is not None:
            points = torch.cat([points, cato], dim=-1)

        brdf = self.diffuse_albedo_layers(points)
        diffuse_albedo = torch.sigmoid(brdf[..., :3])

        brdf_xi = self.diffuse_albedo_layers(points + torch.randn(points.shape).cuda() * 0.01)
        random_xi_diffuse = torch.sigmoid(brdf_xi[..., :3])

        offset = 3
        if self.roughness_mlp and self.same_mlp:
            roughness = torch.sigmoid(brdf[..., offset:offset + 1])
            random_xi_roughness = torch.sigmoid(brdf_xi[..., offset:offset + 1])
            offset += 1
        if not self.fix_specular_albedo and self.specular_mlp and self.same_mlp:
            specular_reflectacne = torch.sigmoid(brdf[..., offset:offset + 1])
            offset += 1

        if self.numBrdfSGs > 1:
            blending_weights = F.softmax(self.blending_weights_layers(points), dim=-1)
        else:
            blending_weights = None

        if self.fix_specular_albedo:
            specular_reflectacne = self.specular_reflectance
        else:
            if not self.specular_mlp:
                specular_reflectacne = torch.sigmoid(self.specular_reflectance)
            elif not self.same_mlp:
                specular_reflectacne = self.specular_layers(points)

            if self.white_specular:
                specular_reflectacne = specular_reflectacne.expand((-1, 3))  # [K, 3]

        if not self.roughness_mlp:
            roughness = torch.sigmoid(self.roughness)
            random_xi_roughness = torch.sigmoid(self.roughness)
        elif not self.same_mlp:
            roughness = self.roughness_layers(points)
            random_xi_roughness = self.roughness_layers(points + torch.randn(points.shape).cuda() * 0.01)

        # prevent roughness become zero.
        # when become zero, the material is pure mirror, the general brdf shading cannot handle this case
        # set the roughness clamp as 0.089 according to float32 precision. reference to https://google.github.io/filament/Filament.html#toc4.8.3.3
        TINNY_ROUGHNESS = 0.089
        # roughness[roughness < TINNY_ROUGHNESS] += TINNY_ROUGHNESS
        roughness = (1 - TINNY_ROUGHNESS) * roughness + TINNY_ROUGHNESS
        random_xi_roughness = (1 - TINNY_ROUGHNESS) * random_xi_roughness + TINNY_ROUGHNESS

        if self.fake_roughness:
            roughness = 0 * roughness + 0.5
            random_xi_roughness = 0 * random_xi_roughness + 0.5

        if self.fake_specular:
            specular_reflectacne = 0 * specular_reflectacne + 0.5

        # remap specular according to https://google.github.io/filament/Filament.html#toc4.8.3.2
        specular_reflectacne = self.specular_remap(specular_reflectacne)

        lgtSGs = self.get_lgtSGs()

        ret = dict([
            ('sg_lgtSGs', lgtSGs),
            ('sg_specular_reflectance', specular_reflectacne),
            ('sg_roughness', roughness),
            ('sg_diffuse_albedo', diffuse_albedo),
            ('sg_blending_weights', blending_weights),
            ('random_xi_roughness', random_xi_roughness),
            ('random_xi_diffuse_albedo', random_xi_diffuse),
        ])
        return ret

    def get_lgtSGs(self):
        lgtSGs = self.lgtSGs
        if self.light_type == 'sg':
            if self.white_light:
                lgtSGs = torch.cat((lgtSGs, lgtSGs[..., -1:], lgtSGs[..., -1:]), dim=-1)
            if self.upper_hemi:
                # limit lobes to upper hemisphere
                lgtSGs = self.restrict_lobes_upper(lgtSGs)
        else:
            lgtSGs = torch.abs(lgtSGs)

        return lgtSGs

    @staticmethod
    # remap specular according to https://google.github.io/filament/Filament.html#toc4.8.3.2
    def specular_remap(specular_reflectacne):
        return 0.16 * specular_reflectacne ** 2

    @staticmethod
    def specular_inv_remap(specular_reflectacne):
        return (specular_reflectacne / 0.16) ** 0.5
