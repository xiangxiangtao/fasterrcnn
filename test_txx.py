# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from lib.model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.net_utils import save_net, load_net, vis_detections
from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.faster_rcnn.resnet import resnet


# try:
#     xrange          # Python 2
# except NameError:
#     xrange = range  # Python 3


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def parse_args():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset', default='pascal_voc', type=str, help='training dataset')
    parser.add_argument('--cfg', dest='cfg_file', default='cfgs/res101.yml', type=str, help='optional config file')######
    parser.add_argument('--net', dest='net', default='res101', type=str, help='vgg16, res50, res101, res152')############
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set config keys')
    parser.add_argument('--load_dir', dest='load_dir', default="models", type=str, help='directory to load models')
    parser.add_argument('--cuda', dest='cuda', default=True,action='store_true', help='whether use CUDA')
    parser.add_argument('--ls', dest='large_scale', action='store_true', help='whether use large imag scale')
    parser.add_argument('--mGPUs', dest='mGPUs', action='store_true', help='whether use multiple GPUs')
    parser.add_argument('--cag', dest='class_agnostic', action='store_true',
                        help='whether perform class_agnostic bbox regression')
    parser.add_argument('--parallel_type', dest='parallel_type', default=0, type=int,
                        help='which part of model to parallel, 0: all, 1: model before roi pooling')
    parser.add_argument('--checksession', dest='checksession', default=1, type=int, help='checksession to load model')
    parser.add_argument('--checkepoch', dest='checkepoch', default=10, type=int, help='checkepoch to load network')
    parser.add_argument('--checkpoint', dest='checkpoint', default=2399, type=int, help='checkpoint to load network')
    parser.add_argument('--vis', dest='vis', action='store_true', help='visualization mode')
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def evaluation(name, net=None,vis=False,cuda=True,class_agnostic=False):
    cfg.TRAIN.USE_FLIPPED = False

    imdb, roidb, ratio_list, ratio_index = combined_roidb(name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    if not net:

        input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
        # input_dir = 'weight'
        if not os.path.exists(input_dir):
            raise Exception('There is no input directory for loading network from ' + input_dir)
        # load_name = os.path.join(input_dir,
        #                          'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

        load_name = os.path.join(input_dir,'faster_rcnn_{}_best.pth'.format(cfg['POOLING_MODE']))

        # initilize the network here.
        if args.net == 'vgg16':
            fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()

        fasterRCNN.create_architecture()

        print("load checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        print('optimal epoch is %s' %(checkpoint['epoch']))
        fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        print('load model successfully!')


    else:

        fasterRCNN=net

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if cuda:
        cfg.CUDA = True

    if cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    # vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):

        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in range(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time, nms_time))
        # print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
        #       .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            cv2.imwrite('result.png', im2show)
            pdb.set_trace()
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    map=imdb.evaluate_detections(all_boxes, output_dir)
    # print(map)
    end = time.time()
    print("test time: %0.4fs" % (end - start))
    return map



def test_weight(weight_path):
    args = parse_args()

    # print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_train"
        args.imdbval_name = "voc_2007_val"
        args.imdbtest_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']


    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    # pprint.pprint(cfg)


    # map=evaluation(name=args.imdbval_name,vis=args.vis,cuda=args.cuda,class_agnostic=args.class_agnostic)

    name = args.imdbval_name
    # name=args.imdbtest_name################################################################################################
    vis=args.vis
    cuda=args.cuda
    class_agnostic=args.class_agnostic

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(name, False)
    imdb.competition_mode(on=True)

    # print('{:d} roidb entries'.format(len(roidb)))



    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()


    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if cuda:
        cfg.CUDA = True

    if cuda:
        fasterRCNN.cuda()

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                            imdb.num_classes, training=False, normalize=False)
    print("*"*50)
    print("len_dataset={}".format(len(dataset)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False, num_workers=0,
                                            pin_memory=True)

    print('load model...')

    print("-"*100)
    print(weight_path)
    checkpoint = torch.load(weight_path)
    # print('optimal epoch is %s' %(checkpoint['epoch']))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']


    start = time.time()
    max_per_image = 100

    # vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(imdb.image_index)
    # print("num_images={}".format(num_images))

    all_boxes = [[[] for _ in range(num_images)]
                for _ in range(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)


    data_iter = iter(dataloader)

    # _t = {'im_detect': time.time(), 'misc': time.time()}

    _t = {'im_detect': Timer(), 'misc': Timer()}

    det_file = os.path.join(output_dir, 'detections.pkl')

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):
        # print("{}/{}".format(i,num_images))
        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

        # det_tic = time.time()

        _t['im_detect'].tic()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        # det_toc = time.time()
        # detect_time = det_toc - det_tic
        # misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in range(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                        for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        detect_time = _t['im_detect'].toc(average=True)
        print('current_im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))


        # misc_toc = time.time()
        # nms_time = misc_toc - misc_tic

        # sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
        #                 .format(i + 1, num_images, detect_time, nms_time))
        # print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
        #       .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            cv2.imwrite('result.png', im2show)
            pdb.set_trace()
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # print('Evaluating detections')
    map=imdb.evaluate_detections(all_boxes, output_dir)
    end = time.time()

    # print("test time: %0.4fs" % (end - start))
    # print('map={}'.format(map))



def test_weights_in_folder():
    args = parse_args()

    # print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_train"
        args.imdbval_name = "voc_2007_val"
        args.imdbtest_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']


    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    # pprint.pprint(cfg)


    # map=evaluation(name=args.imdbval_name,vis=args.vis,cuda=args.cuda,class_agnostic=args.class_agnostic)

    name = args.imdbval_name
    # name=args.imdbtest_name################################################################################################
    vis=args.vis
    cuda=args.cuda
    class_agnostic=args.class_agnostic

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(name, False)
    imdb.competition_mode(on=True)

    # print('{:d} roidb entries'.format(len(roidb)))



    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()


    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if cuda:
        cfg.CUDA = True

    if cuda:
        fasterRCNN.cuda()

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                            imdb.num_classes, training=False, normalize=False)
    print("*"*50)
    print("len_dataset={}".format(len(dataset)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False, num_workers=0,
                                            pin_memory=True)

    print('load model...')
    # weight_folder="models/vgg16/weight_fasterrcnn_composite6"###########################################################
    weight_folder = "models/res101/weight_fasterrcnn_composite18.1_flipped"  ###########################################################
    print("weight_folder={}".format(weight_folder))
    weight_list=os.listdir(weight_folder)

    weight_list.sort(key=lambda x: int(x[x.index("_", x.index("align") + 6) + 1:x.index(".pth")]))  # sort by step
    weight_list.sort(key=lambda x:int(x[x.index("align")+6:x.index("_",x.index("align")+6)]))   #sort by epoch

    print(weight_list)
    for weight in weight_list:
        # print(weight)
        if weight.endswith(".pth"):
        # if weight.endswith(".pth") and weight in ["faster_rcnn_align_3.pth"]:
            weight_path=os.path.join(weight_folder,weight)

            print("-"*100)
            print(weight_path)
            checkpoint = torch.load(weight_path)
            # print('optimal epoch is %s' %(checkpoint['epoch']))
            fasterRCNN.load_state_dict(checkpoint['model'])
            if 'pooling_mode' in checkpoint.keys():
                cfg.POOLING_MODE = checkpoint['pooling_mode']


            start = time.time()
            max_per_image = 100

            # vis = args.vis

            if vis:
                thresh = 0.05
            else:
                thresh = 0.0

            save_name = 'faster_rcnn_10'
            num_images = len(imdb.image_index)
            # print("num_images={}".format(num_images))

            all_boxes = [[[] for _ in range(num_images)]
                        for _ in range(imdb.num_classes)]

            output_dir = get_output_dir(imdb, save_name)


            data_iter = iter(dataloader)

            # _t = {'im_detect': time.time(), 'misc': time.time()}

            _t = {'im_detect': Timer(), 'misc': Timer()}

            det_file = os.path.join(output_dir, 'detections.pkl')

            fasterRCNN.eval()
            empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
            for i in range(num_images):
                # print("{}/{}".format(i,num_images))
                data = next(data_iter)
                with torch.no_grad():
                    im_data.resize_(data[0].size()).copy_(data[0])
                    im_info.resize_(data[1].size()).copy_(data[1])
                    gt_boxes.resize_(data[2].size()).copy_(data[2])
                    num_boxes.resize_(data[3].size()).copy_(data[3])

                # det_tic = time.time()

                _t['im_detect'].tic()
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                scores = cls_prob.data
                boxes = rois.data[:, :, 1:5]

                if cfg.TEST.BBOX_REG:
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        # Optionally normalize targets by a precomputed mean and stdev
                        if class_agnostic:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                pred_boxes /= data[1][0][2].item()

                scores = scores.squeeze()
                pred_boxes = pred_boxes.squeeze()

                # det_toc = time.time()
                # detect_time = det_toc - det_tic
                # misc_tic = time.time()
                if vis:
                    im = cv2.imread(imdb.image_path_at(i))
                    im2show = np.copy(im)
                for j in range(1, imdb.num_classes):
                    inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if class_agnostic:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        if vis:
                            im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                        all_boxes[j][i] = cls_dets.cpu().numpy()
                    else:
                        all_boxes[j][i] = empty_array

                # Limit to max_per_image detections *over all classes*
                if max_per_image > 0:
                    image_scores = np.hstack([all_boxes[j][i][:, -1]
                                              for j in range(1, imdb.num_classes)])
                    if len(image_scores) > max_per_image:
                        image_thresh = np.sort(image_scores)[-max_per_image]
                        for j in range(1, imdb.num_classes):
                            keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                            all_boxes[j][i] = all_boxes[j][i][keep, :]
                detect_time = _t['im_detect'].toc(average=True)
                # print('current_im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))


                # misc_toc = time.time()
                # nms_time = misc_toc - misc_tic

                # sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                #                 .format(i + 1, num_images, detect_time, nms_time))
                # print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                #       .format(i + 1, num_images, detect_time, nms_time))
                sys.stdout.flush()

                if vis:
                    cv2.imwrite('result.png', im2show)
                    pdb.set_trace()
                    # cv2.imshow('test', im2show)
                    # cv2.waitKey(0)

            with open(det_file, 'wb') as f:
                pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

            # print('Evaluating detections')
            map=imdb.evaluate_detections(all_boxes, output_dir)
            end = time.time()

            # print("test time: %0.4fs" % (end - start))
            # print('map={}'.format(map))



if __name__ == '__main__':

    weight_path = "weights/weight_fasterrcnn_res101_composite18.1_epoch1_step4000.pth"
    test_weight(weight_path)

    # test_weights_in_folder()