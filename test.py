"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# ecgan_jittor_test

import jittor as jt
from jittor import init
from jittor import nn

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import cv2
import numpy as np
from PIL import Image
import ntpath
from util import util

jt.flags.use_cuda = 1

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
# web_dir = os.path.join(opt.output_path, opt.name,
#                        '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir,
#                     'Experiment = %s, Phase = %s, Epoch = %s' %
#                     (opt.name, opt.phase, opt.which_epoch))

def read_file(image_numpy):
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis = 2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)
    s = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)
    s = cv2.cvtColor(s,cv2.COLOR_BGR2LAB)
	
    return s
    
def get_mean_and_std(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean,2))
	x_std = np.hstack(np.around(x_std,2))
	return x_mean, x_std

def convert_visuals_to_numpy(opt, visuals):
    tile = opt.batchSize > 8
    t = util.tensor2im(visuals, tile=tile)
    return t

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated, generated2, generated_edge = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('ref_image', data_i['image'][b]),
                               ('synthesized_edge', generated_edge[b]),
                               ('synthesized_image', generated[b]),
                               ('synthesized_image2', generated2[b])
                               ])
        # visualizer.save_images(webpage, visuals, img_path[b:b + 1])
        short_path = ntpath.basename(img_path[b:b + 1][0])
        name = os.path.splitext(short_path)[0]
        
        image_name = os.path.join('%s.jpg' % (name))
        s1 = generated[b]
        s2 = generated2[b]
        t = data_i['image'][b]
        s1 = read_file(convert_visuals_to_numpy(opt, s1))
        s2 = read_file(convert_visuals_to_numpy(opt, s2))
        t = read_file(convert_visuals_to_numpy(opt, t))
	    
        mean_in1 = np.mean(s1, axis=(0, 1), keepdims=True)
        mean_in2 = np.mean(s2, axis=(0, 1), keepdims=True)
        mean_ref = np.mean(t, axis=(0, 1), keepdims=True)
        std_in1 = np.std(s1, axis=(0, 1), keepdims=True)
        std_in2 = np.std(s2, axis=(0, 1), keepdims=True)
        std_ref = np.std(t, axis=(0, 1), keepdims=True)
        img_arr_out1 = (s1 - mean_in1) / std_in1 * std_ref + mean_ref
        img_arr_out1[img_arr_out1 < 0] = 0
        img_arr_out1[img_arr_out1 > 255] = 255
        img_arr_out2 = (s2 - mean_in2) / std_in2 * std_ref + mean_ref
        img_arr_out2[img_arr_out2 < 0] = 0
        img_arr_out2[img_arr_out2 > 255] = 255

        folder = os.path.exists(opt.output_path)
        if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(opt.output_path) 
        

        # path1 = os.path.join(opt.output_path,'r1')
        # path2 = os.path.join(opt.output_path, 'r2')
        
        # folder = os.path.exists(path1)
        # if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        #     os.makedirs(path1) 

        # folder = os.path.exists(path2)
        # if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        #     os.makedirs(path2)         
                
        # img_arr_out1 = cv2.cvtColor(img_arr_out1.astype("uint8"),cv2.COLOR_LAB2BGR)
        # cv2.imwrite(opt.output_path + '/r1/'+image_name,img_arr_out1)

        img_arr_out2 = cv2.cvtColor(img_arr_out2.astype("uint8"),cv2.COLOR_LAB2BGR)
        cv2.imwrite(opt.output_path + '/' + image_name, img_arr_out2)


# webpage.save()
