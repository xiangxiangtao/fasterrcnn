# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import glob

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.misc import imread
import shutil
from matplotlib.ticker import NullLocator






def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # print("-"*50)
  print("voc_eval...")
  print("cachedir={}".format(cachedir))
  print("imagesetfile=",imagesetfile)
  conf_thresh = 0.6  ###################################################################################################

  print("annopath={}".format(annopath))
  dataset_name=annopath.split("/")[-4]
  print("dataset_name={}".format(dataset_name))
  dataset_split=annopath.split("/")[-3]
  print("dataset_split={}".format(dataset_split))
  img_folder = annopath[:annopath.index("/label")] + "/gt"
  project_folder = "/home/ecust/txx/project/fasterrcnn_txx"
  output_folder = os.path.join(project_folder, "detect_when_computing_map", "{}".format(dataset_name), "{}".format(dataset_split),
                               "confThresh{:.2f}_iouThresh{:.2f}".format(conf_thresh,ovthresh))
  os.makedirs(output_folder, exist_ok=True)

  vis = False###########################################################################################################



  colors = [(1,0,1), (0, 0, 0)]

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  # cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
  cachefile = os.path.join(cachedir, '{}_annots.pkl'.format(dataset_split))
  # read list of images
  # with open(imagesetfile, 'r') as f:
  #   lines = f.readlines()
  # imagenames = [x.strip() for x in lines]
  a = sorted(glob.glob("%s/*.*" % imagesetfile))
  imagenames = [path.split('/')[-1].split('.')[0].strip() for path in a]
  # print("imagenames=",imagenames)

  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')

  # extract gt objects for this class
  class_recs = {}
  npos = 0
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}

  # read dets
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
    lines = f.readlines()

  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  # print("image_ids=",image_ids)
  # print("len_image_ids={}".format(len(image_ids)))
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  nd = len(image_ids)
  # print("nd=",nd)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    if vis:
      pass
    print("annopath={}".format(annopath))
    img_folder = annopath[:annopath.index("/label")] + "/gt"
    print("img_folder=",img_folder)
    img_path_list=os.listdir(img_folder)
    for img_name in img_path_list:
      # print("*")
      # print(img_name)
      shutil.copy(os.path.join(img_folder,img_name),os.path.join(output_folder,img_name))


    # go down dets and mark TPs and FPs
    for d in range(nd):
      # print(d,"/",nd)
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        # print("type of overlaps=",type(overlaps))
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
          else:
            fp[d] = 1.
      else:
        fp[d] = 1.

      current_conf=sorted_scores[d]*(-1)
      # print("conf={:.2f},tp={},fp={}".format(current_conf,tp[d],fp[d]))
      if current_conf>conf_thresh and  (tp[d] == 1 or fp[d]==1):
        # print("conf={:.2f},iou={:.2f},tp={},fp={}".format(current_conf,ovmax,tp[d],fp[d]))
        if vis:
          # read images
          img_name="{}.png".format(image_ids[d])
          img_path = os.path.join(output_folder, img_name)  ###########################
          # print(img_path)
          im_in = np.array(imread(img_path))#use pillow to read image
          if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
          # rgb -> bgr
          im_in = im_in[:, :, :3]
          im = im_in[:, :, ::-1]

          im2show = np.copy(im)
          print(im2show.shape)
          plt.figure()
          fig, ax = plt.subplots(1)
          ax.imshow(im2show)


        if vis:
          x1 = bb[0] if bb[0] > 0 else 0
          y1 = bb[1] if bb[1] > 0 else 0
          x2 = bb[2] if bb[2] < im2show.shape[1] else im2show.shape[1]
          y2 = bb[3] if bb[3] < im2show.shape[0] else im2show.shape[0]
          box_w = x2 - x1
          box_h = y2 - y1

          # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

          if tp[d]==1:
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=3, edgecolor=colors[0], facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)

            # Add text
            plt.text(x1,y1,
              s="TP" + "conf={:.2f} ".format(current_conf) + "iou={:.2f}".format(ovmax),
              color="red",
              verticalalignment="top",
              bbox={"color": colors[0], "pad": 0},
            )
          elif fp[d]==1:
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=3, edgecolor=colors[1], facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)

            # Add text
            plt.text(x1,y1,
              s="FP " + "conf={:.2f} ".format(current_conf) + "iou={:.2f}".format(ovmax),
              color="red",
              verticalalignment="top",
              bbox={"color": colors[1], "pad": 0},
            )
        if vis:
          result_path = os.path.join(output_folder, img_name)
          plt.axis("off")
          plt.gca().xaxis.set_major_locator(NullLocator())
          plt.gca().yaxis.set_major_locator(NullLocator())

          plt.savefig(result_path, bbox_inches="tight", pad_inches=0.0)
          plt.close()


  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap
