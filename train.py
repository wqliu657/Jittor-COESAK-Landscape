"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# ecgan_jittor_train

import subprocess
import jittor as jt
from jittor import init
from jittor import nn

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer

import ipdb
import tqdm
import warnings
import os
import cv2
import shutil
warnings.filterwarnings("ignore")


def edge_extract(input_dir):
    output_dir = os.path.join(input_dir, 'edges')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_dir = os.path.join(input_dir,'imgs_new')
    for i, filename in enumerate(os.listdir(input_dir)):
        try:
        # Load image and convert to grayscale
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Sobel edge detection
            edges_x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
            edges_y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)
            edges = cv2.convertScaleAbs(cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0))

        # Save output image
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")
            cv2.imwrite(output_path, edges)

            print(f"Processed image {i+1}/{len(os.listdir(input_dir))}: {filename}")

        except Exception as e:
            print(f"Failed to process image {i + 1}/{len(os.listdir(input_dir))}: {filename} - {str(e)}")

def remove_images_from_dataset(imgs_dir, delete_names):
    for dirs in os.listdir(imgs_dir):
        if dirs == 'imgs':
            dst_path = os.path.join(imgs_dir, 'imgs_new')
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
        elif dirs == 'labels':
            dst_path = os.path.join(imgs_dir, 'labels_new')
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
        else:
            pass
        if dirs == 'imgs' or dirs == 'labels':
            for i in os.listdir(os.path.join(imgs_dir, dirs)):
                img_name = os.path.splitext(i)[0]
                
                if img_name in delete_names:
                    pass
                else:
                    img_path = os.path.join(imgs_dir,dirs,i)
                    shutil.copy(img_path, dst_path)

jt.flags.use_cuda = 1

# parse options
opt = TrainOptions().parse()

# data_prepare
delete_txt_path = "pre_deal/dirty.txt"
# 读取待删除的图像名称列表
with open(delete_txt_path, "r") as f:
    delete_names = [line.strip() for line in f]
# 移除指定的图像文件
remove_images_from_dataset(opt.input_path, delete_names)
edge_extract(opt.input_path)

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)



for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        
        iter_counter.record_one_iteration()
        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            # print("i",i)
            trainer.run_generator_one_step(data_i)
        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('synthesized_image2', trainer.get_latest_generated2()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
