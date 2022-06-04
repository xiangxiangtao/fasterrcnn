# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.faster_rcnn.resnet import resnet
from eval import evaluation
from logger import Logger

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',default='pascal_voc', type=str,help='training dataset')
  parser.add_argument('--net', dest='net',default='res101', type=str, help='vgg16, res101')#########################
  parser.add_argument('--start_epoch', dest='start_epoch', default=1, type=int, help='starting epoch')
  parser.add_argument('--epochs', dest='max_epochs',default=5, type=int,help='number of epochs to train')#########
  parser.add_argument('--disp_interval', dest='disp_interval',default=100, type=int, help='number of iterations to display')
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',default=500, type=int, help='number of iterations to display')######################
  parser.add_argument('--save_dir', dest='save_dir', default="models",type=str, help='directory to save models')
  parser.add_argument('--nw', dest='num_workers', default=8, type=int, help='number of workers to load data')
  parser.add_argument('--cuda', dest='cuda',default=True, action='store_true', help='whether use CUDA')
  parser.add_argument('--ls', dest='large_scale',action='store_true',help='whether use large imag scale')
  parser.add_argument('--mGPUs', dest='mGPUs',action='store_true',help='whether use multiple GPUs')
  parser.add_argument('--bs', dest='batch_size',default=1, type=int, help='batch_size')
  parser.add_argument('--cag', dest='class_agnostic', action='store_true',help='whether to perform class_agnostic bbox regression')

# config optimization
  parser.add_argument('--o', dest='optimizer', default="sgd", type=str,help='training optimizer')
  # parser.add_argument('--lr', dest='lr',default=0.001, type=float,help='starting learning rate')#vgg16
  parser.add_argument('--lr', dest='lr',default=0.001, type=float,help='starting learning rate')#res101##########################
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',default=1, type=int, help='step to do learning rate decay, unit is epoch')#######################
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',default=0.1, type=float, help='learning rate decay ratio')

# set training session
  parser.add_argument('--s', dest='session',default=1, type=int,help='training session')

# resume trained model
  parser.add_argument('--r', dest='resume',default=False, type=bool, help='resume checkpoint or not')
  parser.add_argument('--checksession', dest='checksession',default=1, type=int, help='checksession to load model')
  parser.add_argument('--checkepoch', dest='checkepoch', default=1, type=int, help='checkepoch to load model')
  parser.add_argument('--checkpoint', dest='checkpoint',default=0, type=int,help='checkpoint to load model')
# log and display
#   parser.add_argument('--use_tfb', dest='use_tfboard',action='store_true',help='whether use tensorboard')
  parser.add_argument('--use_tfb', dest='use_tfboard',default=True,help='whether use tensorboard')

  parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      # args.imdb_name = "voc_2007_train"
      args.imdb_name = "voc_2007_trainval"
      # args.imdbval_name = "voc_2007_val"#####################
      args.imdbtest_name = "voc_2007_test"  #####################
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  # elif args.dataset == "pascal_voc_0712":
  #     args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
  #     args.imdbval_name = "voc_2007_test"
  #     args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  # elif args.dataset == "coco":
  #     args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
  #     args.imdbval_name = "coco_2014_minival"
  #     args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  # elif args.dataset == "imagenet":
  #     args.imdb_name = "imagenet_train"
  #     args.imdbval_name = "imagenet_val"
  #     args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  # elif args.dataset == "vg":
  #     # train sizes: train, smalltrain, minitrain
  #     # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
  #     args.imdb_name = "vg_150-50-50_minitrain"
  #     args.imdbval_name = "vg_150-50-50_minival"
  #     args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True######################
  # cfg.TRAIN.USE_FLIPPED = False
  cfg.USE_GPU_NMS = args.cuda
  print("loading train data")
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  # output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  output_dir = args.save_dir + "/" + args.net + "/" + "weight_"
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  best_map=0

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.cuda:
    fasterRCNN.cuda()
      
  if args.optimizer == "adam":
    lr = 0.001###################
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    # from tensorboardX import SummaryWriter
    # logger = SummaryWriter("logs")
    logger = Logger("logs/log_current")###########################

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    print("-"*100)
    print("epoch={}".format(epoch))
    sum_loss=0
    sum_loss_rpn_cls=0
    sum_loss_rpn_box=0
    sum_loss_rcnn_cls=0
    sum_loss_rcnn_box=0

    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      data = next(data_iter)
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(data[2].size()).copy_(data[2])
              num_boxes.resize_(data[3].size()).copy_(data[3])
      # print(im_data.size(),num_boxes)

      fasterRCNN.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
      loss_temp += loss.item()

      # backward
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        # print("step={}".format(step))
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        # print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
        #                         % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        # print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        # print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
        #               % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

        # if args.use_tfboard:
        #   info = {
        #     'loss': loss_temp,
        #     'loss_rpn_cls': loss_rpn_cls,
        #     'loss_rpn_box': loss_rpn_box,
        #     'loss_rcnn_cls': loss_rcnn_cls,
        #     'loss_rcnn_box': loss_rcnn_box
        #   }
          # logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)


        loss_temp = 0

        sum_loss += loss.item()
        sum_loss_rpn_cls += rpn_loss_cls.item()
        sum_loss_rpn_box += rpn_loss_box.item()
        sum_loss_rcnn_cls += RCNN_loss_cls.item()
        sum_loss_rcnn_box += RCNN_loss_bbox.item()

        start = time.time()

      if step>0 and step % args.checkpoint_interval == 0:########################################################################################
        # print("step={}".format(step))
        # print("\n---- Evaluating Model ----")
        # map = evaluation(name=args.imdbval_name,net=fasterRCNN)############################################################
        # map = evaluation(name=args.imdbtest_name, net=fasterRCNN)



        # if map > best_map:###
        if True:
          # save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
          # save_name = os.path.join(output_dir, 'faster_rcnn_' + cfg['POOLING_MODE'] + '_best.pth')
          save_name = os.path.join(output_dir, 'faster_rcnn_' + cfg['POOLING_MODE'] + '_{}_{}.pth'.format(epoch,step))#########models/vgg16/
          save_checkpoint({
            'session': args.session,
            'epoch': epoch,
            'step': step,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
          }, save_name)
          print('save model: {}'.format(save_name))

          # best_map = map###

    # if epoch % args.evaluation_interval==0:
    #   # log
    #   if args.use_tfboard:
    #     sum_loss /= iters_per_epoch
    #     sum_loss_rpn_cls /= iters_per_epoch
    #     sum_loss_rpn_box /= iters_per_epoch
    #     sum_loss_rcnn_cls /= iters_per_epoch
    #     sum_loss_rcnn_box /= iters_per_epoch
    #     logger_info = [
    #       ('loss', sum_loss),
    #       ('loss_rpn_cls', sum_loss_rpn_cls),
    #       ('loss_rpn_box', sum_loss_rpn_box),
    #       ('loss_rcnn_cls', sum_loss_rcnn_cls),
    #       ('loss_rcnn_box', sum_loss_rcnn_box),
    #       ('test_map', map),  ###
    #     ]
    #     logger.list_of_scalars_summary(logger_info, epoch)



  # if args.use_tfboard:
  #   logger.close()
