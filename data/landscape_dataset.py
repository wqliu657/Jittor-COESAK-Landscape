"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class LANDSCAPEDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=29)
        parser.set_defaults(aspect_ratio=1.333)
        parser.set_defaults(batchSize=1)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        root = opt.input_path
        phase = 'val' if opt.phase == 'test' else 'train'

        if opt.phase == 'train':
            label_dir = os.path.join(root, 'labels_new')
            label_paths = make_dataset(label_dir, recursive=True)

            image_dir = os.path.join(root, 'imgs_new')
            image_paths = make_dataset(image_dir, recursive=True)

            edge_dir = os.path.join(root, 'edges')
            edge_paths = make_dataset(edge_dir, recursive=True)
        
        elif opt.phase == 'test':
            label_dir = os.path.join(root, 'val_B_labels_resized')
            label_paths = make_dataset(label_dir, recursive=True)

            image_dir = os.path.join(root, 'val_B_labels_resized')
            image_paths = make_dataset(image_dir, recursive=True)

            edge_dir = os.path.join(root, 'val_B_labels_resized')
            edge_paths = make_dataset(edge_dir, recursive=True)
            
        if not opt.no_instance:
            instance_paths = [p for p in label_paths if p.endswith('_instanceIds.png')]
        else:
            instance_paths = []

        return label_paths, image_paths, instance_paths, edge_paths

    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        # compare the first 3 components, [city]_[id1]_[id2]
        return '_'.join(name1.split('_')[:3]) == \
            '_'.join(name2.split('_')[:3])
