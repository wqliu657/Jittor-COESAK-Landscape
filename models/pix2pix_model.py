"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from jittor import init
from jittor import nn
import models.networks as networks
import util.util as util

import warnings
warnings.filterwarnings("ignore")

class Pix2PixModel(nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def start_grad(self):
        for param in self.netD.parameters():
            param.start_grad()

    def stop_grad(self):
        for param in self.netD.parameters():
            param.stop_grad()

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = jt.float32
        self.ByteTensor = jt.float32
        (self.netG, self.netD, self.netE) = self.initialize_networks(opt)
        # self.labelmix_function = nn.MSELoss()
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = nn.L1Loss()
            if (not opt.no_vgg_loss):
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    def execute(self, data, mode):
        (input_semantics, label_existence, real_image, real_edge) = self.preprocess_input(data)
        if (mode == 'generator'):
            (g_loss, generated, generated2, generated_edge) = self.compute_generator_loss(input_semantics, label_existence, real_image, real_edge)
            return (g_loss, generated, generated2, generated_edge)
        elif (mode == 'discriminator'):
            d_loss = self.compute_discriminator_loss(input_semantics, label_existence, real_image, real_edge)
            return d_loss
        elif (mode == 'encode_only'):
            (z, mu, logvar) = self.encode_z(real_image)
            return (mu, logvar)
        elif (mode == 'inference'):
            with jt.no_grad():
                (fake_image, fake_image2, fake_edge, _) = self.generate_fake(input_semantics, label_existence, real_image)
            return (fake_image, fake_image2, fake_edge)
        else:
            raise ValueError('|mode| is invalid')

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())
        (beta1, beta2) = (opt.beta1, opt.beta2)
        if opt.no_TTUR:
            (G_lr, D_lr) = (opt.lr, opt.lr)
        else:
            (G_lr, D_lr) = ((opt.lr / 2), (opt.lr * 2))
        optimizer_G = jt.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = jt.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        return (optimizer_G, optimizer_D)

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = (networks.define_D(opt) if opt.isTrain else None)
        netE = (networks.define_E(opt) if opt.use_vae else None)
        if ((not opt.isTrain) or opt.continue_train):
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        return (netG, netD, netE)

    def preprocess_input(self, data):
        data['label'] = data['label'].long()
        #if self.use_gpu():
        #    data['label'] = data['label'].cuda()
        #    data['instance'] = data['instance'].cuda()
        #    data['image'] = data['image'].cuda()
        #    data['edge'] = data['edge'].cuda()
        # label_map = data['label']
        # (bs, _, h, w) = label_map.shape
        # target_label_existence = jt.squeeze(label_map, dim=1)
        # tvect = jt.zeros(bs, self.opt.label_nc)
        # for i in range(bs):
        #     hist = jt.histc(target_label_existence[i].cpu().data.float(), bins=self.opt.label_nc, min=0, max=(self.opt.label_nc - 1))
        #     vect = (hist > 0)
        #     tvect[i] = vect
        
        # nc = ((self.opt.label_nc + 1) if self.opt.contain_dontcare_label else self.opt.label_nc)
        # input_label = self.FloatTensor(bs, nc, h, w).zero_()
        # input_semantics = input_label.scatter_(1, label_map, 1.0)

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label else self.opt.label_nc
        input_label = jt.zeros((bs, nc, h, w), dtype=self.FloatTensor)
        input_semantics = input_label.scatter_(1, label_map, jt.float32(1.0))

        if (not self.opt.no_instance):
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = jt.contrib.concat((input_semantics, instance_edge_map), dim=1)
        return (input_semantics, label_map, data['image'], data['edge'])

    def compute_generator_loss(self, input_semantics, label_existence, real_image, real_edge):
        G_losses = {}
        (fake_image, fake_image2, fake_edge, KLD_loss) = self.generate_fake(input_semantics, label_existence, real_image, compute_kld_loss=self.opt.use_vae)
        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss
        (pred_fake, pred_real) = self.discriminate(input_semantics, fake_image, real_image)
        (pred_fake2, pred_real2) = self.discriminate(input_semantics, fake_image2, real_image)
        (pred_fake_edge, pred_real_egde) = self.discriminate(input_semantics, fake_edge, real_edge)
        G_losses['GAN'] = ((self.criterionGAN(pred_fake, True, for_discriminator=False) + self.criterionGAN(pred_fake_edge, True, for_discriminator=False)) + (self.criterionGAN(pred_fake2, True, for_discriminator=False) * 2))
        if (not self.opt.no_ganFeat_loss):
            num_D = len(pred_fake)
            # GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            GAN_Feat_loss = jt.zeros([1]).float32()
            for i in range(num_D):
                # print("11",i,len(pred_fake[i]))
                num_intermediate_outputs = (len(pred_fake[i]) - 1)
                # print(num_intermediate_outputs)
                for j in range(num_intermediate_outputs):
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    unweighted_loss2 = self.criterionFeat(pred_fake2[i][j], pred_real2[i][j].detach())
                    unweighted_loss_edge = self.criterionFeat(pred_fake_edge[i][j], pred_real_egde[i][j].detach())
                    # print("11",unweighted_loss, unweighted_loss2, unweighted_loss_edge)
                    GAN_Feat_loss += ((((unweighted_loss + (unweighted_loss2 * 2)) + unweighted_loss_edge) * self.opt.lambda_feat) / num_D)
            G_losses['GAN_Feat'] = GAN_Feat_loss
        if (not self.opt.no_vgg_loss):
            G_losses['VGG'] = (((self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg) + (self.criterionVGG(fake_edge, real_edge) * self.opt.lambda_vgg)) + ((self.criterionVGG(fake_image2, real_image) * self.opt.lambda_vgg) * 2))
        return (G_losses, fake_image, fake_image2, fake_edge)

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        loss = 0
        for i in range(len(output_D_mixed)):
            output_D_mixed_i = output_D_mixed[i]
            output_D_real_i = output_D_real[i]
            output_D_fake_i = output_D_fake[i]
            if isinstance(output_D_mixed_i, list):
                output_D_mixed_i = output_D_mixed_i[-1]
                output_D_real_i = output_D_real_i[-1]
                output_D_fake_i = output_D_fake_i[-1]
            n, c, h, w = output_D_fake_i.shape
            mask_i = nn.interpolate(mask, (h,w), mode='nearest')
            mixed_D_output_i = mask_i * output_D_real_i + (1-mask_i) * output_D_fake_i
            loss += self.labelmix_function(mixed_D_output_i, output_D_mixed_i)
        return loss
    
    def compute_discriminator_loss(self, input_semantics, label_existence, real_image, real_edge):
        D_losses = {}
        with jt.no_grad():
            (fake_image, fake_image2, fake_edge, _) = self.generate_fake(input_semantics, label_existence, real_image)
            fake_image = fake_image.detach()
            fake_image.start_grad()
            fake_image2 = fake_image2.detach()
            fake_image2.start_grad()
            fake_edge = fake_edge.detach()
            fake_edge.start_grad()
        (pred_fake, pred_real) = self.discriminate(input_semantics, fake_image, real_image)
        (pred_fake2, pred_real2) = self.discriminate(input_semantics, fake_image2, real_image)
        (pred_fake_edge, pred_real_edge) = self.discriminate(input_semantics, fake_edge, real_edge)
        D_losses['D_Fake'] = ((self.criterionGAN(pred_fake, False, for_discriminator=True) + self.criterionGAN(pred_fake_edge, False, for_discriminator=True)) + (self.criterionGAN(pred_fake2, False, for_discriminator=True) * 2))
        D_losses['D_real'] = ((self.criterionGAN(pred_real, True, for_discriminator=True) + self.criterionGAN(pred_real_edge, True, for_discriminator=True)) + (self.criterionGAN(pred_real2, True, for_discriminator=True) * 2))
        return D_losses

    def encode_z(self, real_image):
        (mu, logvar) = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        # print("1", mu)
        # print("2", logvar)
        # print("3", z)
        return (z, mu, logvar)

    def generate_fake(self, input_semantics, label_existence, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            (z, mu, logvar) = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = (self.KLDLoss(mu, logvar) * self.opt.lambda_kld)
        (fake_image, fake_image2, fake_edge) = self.netG(input_semantics, label_existence, z=z)
        assert ((not compute_kld_loss) or self.opt.use_vae), 'You cannot compute KLD loss if opt.use_vae == False'
        return (fake_image, fake_image2, fake_edge, KLD_loss)

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = jt.contrib.concat([input_semantics, fake_image], dim=1)
        real_concat = jt.contrib.concat([input_semantics, real_image], dim=1)
        fake_and_real = jt.contrib.concat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)
        (pred_fake, pred_real) = self.divide_pred(discriminator_out)
        return (pred_fake, pred_real)

    def divide_pred(self, pred):
        if (type(pred) == list):
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:(tensor.shape[0] // 2)] for tensor in p])
                real.append([tensor[(tensor.shape[0] // 2):] for tensor in p])
        else:
            fake = pred[:(pred.shape[0] // 2)]
            real = pred[(pred.shape[0] // 2):]
        return (fake, real)

    def get_edges(self, t):
        edge = self.ByteTensor(t.shape).zero_()
        edge[:, :, :, 1:] = (edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :(- 1)]))
        edge[:, :, :, :(- 1)] = (edge[:, :, :, :(- 1)] | (t[:, :, :, 1:] != t[:, :, :, :(- 1)]))
        edge[:, :, 1:, :] = (edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :(- 1), :]))
        edge[:, :, :(- 1), :] = (edge[:, :, :(- 1), :] | (t[:, :, 1:, :] != t[:, :, :(- 1), :]))
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = jt.exp((0.5 * logvar))
        eps = jt.randn_like(std)
        return (eps.multiply(std) + mu)

    def use_gpu(self):
        return (len(self.opt.gpu_ids) > 0)