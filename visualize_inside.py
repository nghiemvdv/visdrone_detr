# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
### copy and paste from https://github.com/thedeepreader/detr_tutorial/blob/master/detr/test.py
### with minor changes
"""
import math
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np

import torch

import util.misc as utils

from models import build_model
from datasets.visdrone_det import make_visdrone_transforms

import matplotlib.pyplot as plt
import time

from tqdm import tqdm ###
# from torchsummary import summary
from util.detr_torchsummary import summary ###



colors = [
    (255,255,255),  # ignored
    (255, 0, 0),    # pedestrian
    (255, 128, 0),  # people
    (255, 255, 0),  # bicycle
    (128, 255, 0),  # car
    (0, 255, 0),    # van
    (0, 255, 128),  # truck
    (0, 255, 255),  # tricycle
    (0, 128, 255),  # awning-tricycle
    (0, 0, 255),    # bus
    (128, 0, 255),  # motor
    (0,0,0)         # others
    ]

label_dict = {
	0: 'ignored',
	1: 'pedestrian',
	2: 'people',
	3: 'bicycle',
	4: 'car',
	5: 'van',
	6: 'truck',
	7: 'tricycle',
	8: 'awning-tricycle',
	9: 'bus',
	10: 'motor',
	11: 'others'
}

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))

    return img_files

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1000, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--visdrone_det_test_path', type=str) ###
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    # parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--num_workers', default=2, type=int)

    # # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--thresh', default=0.5, help='output detection threshold', type=float)
    parser.add_argument('--visualize', help='bounding box output', action='store_true')
    parser.add_argument('--annotation', help='visdrone format annotation', action='store_true')

    return parser


@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_path):
    # print(model)
    model.eval()
    duration = 0
    ###
    # images_output_path = os.path.join(output_path, 'images')
    # annotations_output_path = os.path.join(output_path, 'annotations')
    pbar = tqdm(total=len(images_path), position=0, leave=True)
    ###
    for img_sample in images_path:
        ###
        # anno_output = []
        ###
        filename = os.path.basename(img_sample)
        ###print("processing...{}".format(filename))
        orig_image = Image.open(img_sample)
        w, h = orig_image.size
        transform = make_visdrone_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)])
        }
        image, targets = transform(orig_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)


        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)

            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])

            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1])

            ),

        ]

        ###
        # print(image.size())
        # print(type(model))
        summary(model, (3, 750, 1333))
        print('done!!!')
        sys.exit()
        ###

        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.85
        max_pred = probas.max(-1)
        keep = max_pred.values > args.thresh
        categories = max_pred.indices[keep].cpu().data.numpy()

        # ###
        # print(categories)
        # sys.exit()
        # ###

        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], orig_image.size)
        probas = probas[keep].cpu().data.numpy()

        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0].cpu()
        dec_attn_weights = dec_attn_weights[0].cpu()

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]

        # if len(bboxes_scaled) == 0:
        #     continue

        ###
        dec_attn_weights = dec_attn_weights[:,keep,:].cpu()
        ###
        # print(dec_attn_weights)
        # print(dec_attn_weights.shape)
        # print(keep.nonzero().shape)
        # print(type(dec_attn_weights[0, 0]))
        # sys.exit()
        ###
        im = orig_image
        fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(90, 10))
        # colors = COLORS * 100
        for idx, (idxx, ax_i, (xmin, ymin, xmax, ymax)) in enumerate(zip(keep.nonzero(), axs.T, bboxes_scaled)):
            ax = ax_i[0]
            ax.imshow(dec_attn_weights[0, idx].view(h, w))
            ax.axis('off')
            ax.set_title(f'query id: {idxx.item()}')
            ax = ax_i[1]
            ax.imshow(im)
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color='blue', linewidth=3))
            ax.axis('off')
            ax.set_title(label_dict[categories[idx]])
        fig.tight_layout()
        fig.savefig(os.path.join(output_path, 'inside.png'))
        if args.visualize:
            print('outputing visualization')
            fig.savefig(os.path.join(output_path, 'dec_inside.png'))
        else:
            print('visualization turned off')
        ###

        ###
        # output of the CNN
        f_map = conv_features['0']
        print("Encoder attention:      ", enc_attn_weights[0].shape)
        print("Feature map:            ", f_map.tensors.shape)



        # get the HxW shape of the feature maps of the CNN
        shape = f_map.tensors.shape[-2:]
        # and reshape the self-attention to a more interpretable shape
        sattn = enc_attn_weights[0].reshape(shape + shape)
        print("Reshaped self-attention:", sattn.shape)



        # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
        fact = 32

        # let's select 4 reference points for visualization
        idxs = [(200, 200), (280, 400), (200, 600), (440, 800),]

        # here we create the canvas
        fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
        # and we add one plot per reference point
        gs = fig.add_gridspec(2, 4)
        axs = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[0, -1]),
            fig.add_subplot(gs[1, -1]),
        ]

        # for each one of the reference points, let's plot the self-attention
        # for that point
        for idx_o, ax in zip(idxs, axs):
            idx = (idx_o[0] // fact, idx_o[1] // fact)
            ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
            ax.axis('off')
            ax.set_title(f'self-attention{idx_o}')

        # and now let's add the central image, with the reference points as red circles
        fcenter_ax = fig.add_subplot(gs[:, 1:-1])
        fcenter_ax.imshow(im)
        for (y, x) in idxs:
            scale = im.height / image.shape[-2]
            x = ((x // fact) + 0.5) * fact
            y = ((y // fact) + 0.5) * fact
            fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
            fcenter_ax.axis('off')
        if args.visualize:
            print('outputing visualization')
            fig.savefig(os.path.join(output_path, 'enc_inside.png'))
        else:
            print('visualization turned off')
        ###

        # img = np.array(orig_image)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # for box, category, proba in zip(bboxes_scaled, categories, probas):
        #     bbox = box.cpu().data.numpy()
        #     bbox = bbox.astype(np.int32)
        #     # bbox = np.array([
        #     #     [bbox[0], bbox[1]],
        #     #     [bbox[2], bbox[1]],
        #     #     [bbox[2], bbox[3]],
        #     #     [bbox[0], bbox[3]],
        #     #     ])
        #     # bbox = bbox.reshape((4, 2))
        #     # cv2.polylines(img, [bbox], True, (0, 255, 0), 2)
        #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[category], 1)
        #     cv2.putText(img, label_dict[category], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_PLAIN, 0.7, colors[category])
        #     anno_output.append([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], proba[category], category, -1, -1])

        # if not os.path.exists(images_output_path):
        #     print('{} not detected, creating folder...'.format(images_output_path))
        #     os.mkdir(images_output_path)
        # if not os.path.exists(annotations_output_path):
        #     print('{} not detected, creating folder...'.format(annotations_output_path))
        #     os.mkdir(annotations_output_path)  
        # if args.visualize:
        #     img_save_path = os.path.join(images_output_path, filename)
        #     cv2.imwrite(img_save_path, img)
        # if args.annotation:
        #     txt_filename = filename[:-3] + 'txt'
        #     anno_save_path = os.path.join(annotations_output_path, txt_filename)
        #     with open(anno_save_path, 'w') as f:
        #         for line in anno_output:
        #             csv_line = ",".join(list(map(str, line)))
        #             f.write("%s\n" % csv_line)

        # cv2.imshow("img", img)
        # cv2.waitKey()
        infer_time = end_t - start_t
        duration += infer_time
        ###print("Processing...{} ({:.3f}s)".format(filename, infer_time))
        pbar.update(1)

    pbar.close()
    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    image_paths = get_images(args.visdrone_det_test_path)

    infer(image_paths, model, postprocessors, device, args.output_dir)