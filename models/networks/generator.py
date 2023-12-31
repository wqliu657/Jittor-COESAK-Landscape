"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from jittor import init
from jittor import nn
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock

class SPADEGenerator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='normal', help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        (self.sw, self.sh) = self.compute_latent_vector_size(opt)
        if opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, (((16 * nf) * self.sw) * self.sh))
        else:
            self.fc = nn.Conv(self.opt.semantic_nc, (16 * nf), 3, padding=1)
        self.head_0 = SPADEResnetBlock((16 * nf), (16 * nf), opt)
        self.G_middle_0 = SPADEResnetBlock((16 * nf), (16 * nf), opt)
        self.G_middle_1 = SPADEResnetBlock((16 * nf), (16 * nf), opt)
        self.up_0 = SPADEResnetBlock((16 * nf), (8 * nf), opt)
        self.up_1 = SPADEResnetBlock((8 * nf), (4 * nf), opt)
        self.up_2 = SPADEResnetBlock((4 * nf), (2 * nf), opt)
        self.up_3 = SPADEResnetBlock((2 * nf), (1 * nf), opt)
        final_nc = nf
        if (opt.num_upsampling_layers == 'most'):
            self.up_4 = SPADEResnetBlock((1 * nf), (nf // 2), opt)
            final_nc = (nf // 2)
        self.conv_img1 = nn.Conv(final_nc, 32, 3, padding=1)
        self.conv_img2 = nn.Conv(32, 16, 3, padding=1)
        self.conv_img3 = nn.Conv(16, 8, 3, padding=1)
        self.conv_img4 = nn.Conv(8, 3, 3, padding=1)
        self.conv_edge1 = nn.Conv(final_nc, 32, 3, padding=1)
        self.conv_edge2 = nn.Conv(32, 16, 3, padding=1)
        self.conv_edge3 = nn.Conv(16, 8, 3, padding=1)
        self.conv_edge4 = nn.Conv(8, 3, 3, padding=1)
        self.conv_comb = nn.Conv((((final_nc + 3) + 3) + self.opt.semantic_nc), 3, 3, padding=1)
        self.conv_comb1 = nn.Conv((((final_nc + 3) + 3) + self.opt.semantic_nc), self.opt.label_nc, 3, padding=1)
        self.conv_comb2 = nn.Conv(self.opt.label_nc, (((final_nc + 3) + 3) + self.opt.semantic_nc), 3, padding=1)
        self.fc1 = nn.Linear(self.opt.label_nc, self.opt.label_nc)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if (opt.num_upsampling_layers == 'normal'):
            num_up_layers = 5
        elif (opt.num_upsampling_layers == 'more'):
            num_up_layers = 6
        elif (opt.num_upsampling_layers == 'most'):
            num_up_layers = 7
        else:
            raise ValueError(('opt.num_upsampling_layers [%s] not recognized' % opt.num_upsampling_layers))
        sw = (opt.crop_size // (2 ** num_up_layers))
        sh = round((sw / opt.aspect_ratio))
        return (sw, sh)

    def execute(self, input, label_existence, z=None):
        seg = input
        if self.opt.use_vae:
            if (z is None):
                z = jt.randn(input.shape[0], self.opt.z_dim)
            x = self.fc(z)
            x = x.view(((- 1), (16 * self.opt.ngf), self.sh, self.sw))
        else:
            x = nn.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)
        # print("sa",x)
        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        if ((self.opt.num_upsampling_layers == 'more') or (self.opt.num_upsampling_layers == 'most')):
            x = self.up(x)
        x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)
        if (self.opt.num_upsampling_layers == 'most'):
            x = self.up(x)
            x = self.up_4(x, seg)
        edge1 = self.conv_edge1(nn.leaky_relu(x, 0.2))
        edge1_att = jt.sigmoid(edge1)
        edge2 = self.conv_edge2(nn.leaky_relu(edge1, 0.2))
        edge2_att = jt.sigmoid(edge2)
        edge3 = self.conv_edge3(nn.leaky_relu(edge2, 0.2))
        edge3_att = jt.sigmoid(edge3)
        image1 = self.conv_img1(nn.leaky_relu(x, 0.2))
        image1 = ((image1 * edge1_att) + image1)
        image2 = self.conv_img2(nn.leaky_relu(image1, 0.2))
        image2 = ((image2 * edge2_att) + image2)
        image3 = self.conv_img3(nn.leaky_relu(image2, 0.2))
        image3 = ((image3 * edge3_att) + image3)
        edge4 = self.conv_edge4(nn.leaky_relu(edge3, 0.2))
        edge_att = jt.sigmoid(jt.tanh(edge4))
        edge = jt.tanh(edge4)
        image4 = self.conv_img4(nn.leaky_relu(image3, 0.2))
        image = jt.tanh(image4)
        image = ((image * edge_att) + image)
        feature_image_edge_seg_combine = jt.contrib.concat((x, image, edge, seg), dim=1)
        class_feature = self.conv_comb1(nn.leaky_relu(feature_image_edge_seg_combine, 0.2))
        (b, c, _, _) = class_feature.shape
        gamma = self.avg_pool(class_feature).view((b, c))
        gamma = jt.sigmoid(gamma).view((b, c, 1, 1))
        refined_class_feature = (class_feature + (class_feature * gamma))
        refined_class_feature = self.conv_comb2(nn.leaky_relu(refined_class_feature, 0.2))
        feature_image_edge_seg_combine = self.conv_comb(nn.leaky_relu(refined_class_feature, 0.2))
        image2 = jt.tanh(feature_image_edge_seg_combine)
        return (image, image2, edge)

class Pix2PixHDGenerator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3, help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7, help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        iput_nc = ((opt.label_nc + (1 if opt.contain_dontcare_label else 0)) + (0 if opt.no_instance else 1))
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU()
        model = []
        model += [nn.ReflectionPad2d((opt.resnet_initial_kernel_size // 2)), norm_layer(nn.Conv(input_nc, opt.ngf, opt.resnet_initial_kernel_size, padding=0)), activation]
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv((opt.ngf * mult), ((opt.ngf * mult) * 2), 3, stride=2, padding=1)), activation]
            mult *= 2
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock((opt.ngf * mult), norm_layer=norm_layer, activation=activation, kernel_size=opt.resnet_kernel_size)]
        for i in range(opt.resnet_n_downsample):
            nc_in = int((opt.ngf * mult))
            nc_out = int(((opt.ngf * mult) / 2))
            model += [norm_layer(nn.ConvTranspose(nc_in, nc_out, 3, stride=2, padding=1, output_padding=1)), activation]
            mult = (mult // 2)
        model += [nn.ReflectionPad2d(3), nn.Conv(nc_out, opt.output_nc, 7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def execute(self, input, z=None):
        return self.model(input)
