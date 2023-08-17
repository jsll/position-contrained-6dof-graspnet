import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
from models import losses
from pointnet2_ops import pointnet2_modules as pointnet2


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch + 1 + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
        elif classname.find('batchnorm.BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_classifier(opt, gpu_ids, init_type, init_gain, device):
    net = None
    net = GraspSamplerVAE(opt.model_scale, opt.pointnet_radius,
                            opt.pointnet_nclusters, opt.latent_size, device)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_loss(opt):
    kl_loss = losses.kl_divergence
    reconstruction_loss = losses.control_point_l1_loss
    return kl_loss, reconstruction_loss


class GraspSampler(nn.Module):
    def __init__(self, latent_size, device):
        super(GraspSampler, self).__init__()
        self.latent_size = latent_size
        self.device = device

    def create_decoder(self, model_scale, pointnet_radius, pointnet_nclusters,
                       num_input_features):
        # The number of input features for the decoder is 3+k+u where 3
        # represents the x, y, z position of the point-cloud, k is the latent size (most often 2),
        # and u is 1 as we train a constrained network.

        self.decoder = base_network(pointnet_radius, pointnet_nclusters,
                                    model_scale, num_input_features)
        self.q = nn.Linear(model_scale * 1024, 4)
        self.t = nn.Linear(model_scale * 1024, 3)

    def decode(self, xyz, z, features=None):
        xyz_features = self.concatenate_z_with_pc(xyz,
                                                  z)
        if features is not None:
            xyz_features = torch.cat((xyz_features, features), -1)
        else:
            query_point_encoding = self.setup_query_point_feature(
                xyz.shape[0], xyz.shape[1])
            xyz_features = torch.cat((xyz_features, query_point_encoding), -1)

        xyz_features = xyz_features.transpose(-1, 1).contiguous()
        for module in self.decoder[0]:
            xyz, xyz_features = module(xyz, xyz_features)
        x = self.decoder[1](xyz_features.squeeze(-1))
        predicted_qt = torch.cat(
            (F.normalize(self.q(x), p=2, dim=-1), self.t(x)), -1)

        return predicted_qt

    def concatenate_z_with_pc(self, pc, z):
        z.unsqueeze_(1)
        z = z.expand(-1, pc.shape[1], -1)
        return torch.cat((pc, z), -1)

    def setup_query_point_feature(self, batch_size, num_points):
        query_point_feature = torch.zeros(
            (batch_size, num_points, 1)).to(self.device)
        query_point_feature[:, -1] = 1
        return query_point_feature


class GraspSamplerVAE(GraspSampler):
    """Network for learning a generative VAE grasp-sampler
    """

    def __init__(self,
                 model_scale,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 latent_size=2,
                 device="cpu"):
        super(GraspSamplerVAE, self).__init__(
            latent_size, device)

        self.create_encoder(model_scale, pointnet_radius,
                            pointnet_nclusters, 19+1)

        self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                            latent_size + 3 + 1)
        self.create_bottleneck(model_scale * 1024, latent_size)

    def create_encoder(
            self,
            model_scale,
            pointnet_radius,
            pointnet_nclusters,
            num_input_features
    ):
        # The number of input features for the encoder is 20: the x, y, z
        # position of the point-cloud, the flattened 4x4=16 grasp pose matrix,
        # and, a 1/0 binary encoding representing which point we want to generate
        # grasps around
        self.encoder = base_network(pointnet_radius, pointnet_nclusters,
                                    model_scale, num_input_features)

    def create_bottleneck(self, input_size, latent_size):
        mu = nn.Linear(input_size, latent_size)
        logvar = nn.Linear(input_size, latent_size)
        self.latent_space = nn.ModuleList([mu, logvar])

    def encode(self, pc_xyz, grasps, position_contraint_feature):
        grasp_features = grasps.unsqueeze(1).expand(-1, pc_xyz.shape[1], -1)
        features = torch.cat(
            (pc_xyz, grasp_features),
            -1)
        
        features = torch.cat((features, position_contraint_feature), -1)
        features = features.transpose(-1, 1).contiguous()
        for module in self.encoder[0]:
            pc_xyz, features = module(pc_xyz, features)
        return self.encoder[1](features.squeeze(-1))

    def bottleneck(self, z):
        return self.latent_space[0](z), self.latent_space[1](z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, pc, grasp=None, features=None, train=True):
        if train:
            return self.forward_train(pc, grasp, features)
        else:
            return self.forward_test(pc, grasp, features)

    def forward_train(self, pc, grasp, features):
        z = self.encode(pc, grasp, features)
        mu, logvar = self.bottleneck(z)
        z = self.reparameterize(mu, logvar)
        qt = self.decode(pc, z, features)
        return qt,  mu, logvar

    def forward_test(self, pc, grasp, features):
        z = self.encode(pc, grasp, features)
        mu, _ = self.bottleneck(z)
        qt = self.decode(pc, mu, features)
        return qt

    def sample_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_size).to(self.device)

    def generate_grasps(self, pc, z=None, features=None):
        if z is None:
            z = self.sample_latent(pc.shape[0])
        qt = self.decode(pc, z, features)
        return qt, z.squeeze()


def base_network(pointnet_radius, pointnet_nclusters, scale, in_features):
    sa1_module = pointnet2.PointnetSAModule(
        npoint=pointnet_nclusters,
        radius=pointnet_radius,
        nsample=64,
        mlp=[in_features, 64 * scale, 64 * scale, 128 * scale])
    sa2_module = pointnet2.PointnetSAModule(
        npoint=32,
        radius=0.04,
        nsample=128,
        mlp=[128 * scale, 128 * scale, 128 * scale, 256 * scale])

    sa3_module = pointnet2.PointnetSAModule(
        mlp=[256 * scale, 256 * scale, 256 * scale, 512 * scale])

    sa_modules = nn.ModuleList([sa1_module, sa2_module, sa3_module])
    fc_layer = nn.Sequential(nn.Linear(512 * scale, 1024 * scale),
                             nn.BatchNorm1d(1024 * scale), nn.ReLU(True),
                             nn.Linear(1024 * scale, 1024 * scale),
                             nn.BatchNorm1d(1024 * scale), nn.ReLU(True))
    return nn.ModuleList([sa_modules, fc_layer])
