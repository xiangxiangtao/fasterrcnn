from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from lib.datasets.imdb import imdb
from lib.datasets.imdb import ROOT_DIR
from lib.datasets import ds_utils
from lib.datasets.voc_eval import voc_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from lib.model.utils.config import cfg

#
# try:
#     xrange          # Python 2
# except NameError:
#     xrange = range  # Python 3
# <<<< obsolete

iou_thresh=0.5########################################################################

dataset_name_composite='composite_gas_gmy_500_400'
dataset_name_composite_1='composite_gas_1_gmy_500_400'
dataset_name_composite_2='composite_gas_2_gmy_500_400'
dataset_name_composite_6='composite_gas_6_gmy_500_400'
dataset_name_composite_8='composite8_gmy'
dataset_name_composite_8_point_1='composite8.1_gmy'
dataset_name_composite_8_point_2='composite_8.2_gmy'
dataset_name_composite_8_point_3='composite_8.3_gmy'
dataset_name_composite_8_point_4='composite_8.4_gmy'
dataset_name_composite_8_point_5='composite_8.5_gmy'
dataset_name_composite_9_point_1='composite_9.1_gmy'
dataset_name_composite_9_point_2='composite_9.2_gmy'
dataset_name_composite_9_point_3='composite_9.3_gmy'
dataset_name_composite_9_point_4='composite_9.4_gmy'
dataset_name_composite_9_point_5='composite_9.5_gmy'
dataset_name_composite_9_point_6='composite_9.6_gmy'
dataset_name_composite_9_point_7='composite_9.7_gmy'
dataset_name_composite_9_point_8='composite_9.8_gmy'
dataset_name_composite_9_point_9='composite_9.9_gmy'
dataset_name_composite_9_point_10='composite_9.10_gmy'
dataset_name_composite_9_point_11='composite_9.11_gmy'
dataset_name_composite_9_point_12='composite_9.12_gmy'
dataset_name_composite_8_point_6='composite_8.6_gmy'
dataset_name_composite_10_point_1='composite_10.1_gmy'
dataset_name_composite_11_point_1='composite_11.1_gmy'
dataset_name_composite_12_point_1='composite_12.1_gmy'
dataset_name_composite_13_point_1='composite_13.1_gmy'
dataset_name_composite_14_point_1='composite_14.1_gmy'
dataset_name_composite_15_point_1='composite_15.1_gmy'
dataset_name_composite_16_point_1='composite_16.1_gmy'
dataset_name_composite_16_point_2='composite_16.2_gmy'
dataset_name_composite_16_point_3='composite_16.3_gmy'
dataset_name_composite_17_point_2='composite_17.2_gmy'
dataset_name_composite_18_point_1='composite_18.1_gmy'
dataset_name_composite_18_point_2='composite_18.2_gmy'
dataset_name_composite_18_point_3='composite_18.3_gmy'
dataset_name_composite_18_point_4='composite_18.4_gmy'
dataset_name_composite_18_point_5='composite_18.5_gmy'
dataset_name_composite_18_point_6='composite_18.6_gmy'
dataset_name_composite_18_point_8='composite_18.8_gmy'
dataset_name_composite_18_point_9='composite_18.9_gmy'
dataset_name_composite_18_point_10='composite_18.10_gmy'


dataset_name_real_annotated='real_annotated'
dataset_name_real_annotated_1='real_annotated_1'
dataset_name_real_annotated_2='real_annotated_2'
dataset_name_real_3='real_3_gmy'
dataset_name_real_5='real_5_gmy'
dataset_name_real_6='real_6_gmy'
dataset_name_real_7='real_7_gmy'
dataset_name_real_annotated_gmy='real_annotated_gmy'
dataset_name_real_annotated_gmy_all='real_annotated_gmy_all'
dataset_name_real_annotated_all_split_gmy='real_annotated_all_split_gmy'

# dataset_name=dataset_name_composite_18_point_1###########################################
dataset_name=dataset_name_real_7
# dataset_name=dataset_name_real_annotated_all_split_gmy


dataset_list0=[dataset_name_composite,dataset_name_composite_1,dataset_name_composite_2,dataset_name_composite_6,dataset_name_composite_8,
               dataset_name_composite_8_point_1,dataset_name_composite_8_point_2,dataset_name_composite_8_point_3,dataset_name_composite_8_point_4,
               dataset_name_composite_8_point_5,dataset_name_composite_9_point_1,dataset_name_composite_9_point_2,dataset_name_composite_9_point_3,
               dataset_name_composite_9_point_4,dataset_name_composite_9_point_5,dataset_name_composite_9_point_6,dataset_name_composite_9_point_7,
               dataset_name_composite_9_point_8,dataset_name_composite_9_point_9,dataset_name_composite_9_point_10,dataset_name_composite_9_point_11,
               dataset_name_composite_9_point_12,dataset_name_composite_8_point_6,dataset_name_composite_10_point_1,dataset_name_composite_11_point_1,
               dataset_name_composite_12_point_1,dataset_name_composite_13_point_1,dataset_name_composite_14_point_1,dataset_name_composite_15_point_1,
               dataset_name_composite_16_point_1,dataset_name_composite_16_point_2,dataset_name_composite_16_point_3,dataset_name_composite_17_point_2,
               dataset_name_composite_18_point_1,dataset_name_composite_18_point_2,dataset_name_composite_18_point_4,dataset_name_composite_18_point_3,
               dataset_name_composite_18_point_5,dataset_name_composite_18_point_6,dataset_name_composite_18_point_9,dataset_name_composite_18_point_8,
               dataset_name_composite_18_point_10]
dataset_list1=[dataset_name_real_annotated_gmy,dataset_name_real_annotated,dataset_name_real_annotated_1,dataset_name_real_annotated_2,
               dataset_name_real_3,dataset_name_real_5,dataset_name_real_6,dataset_name_real_7,
               dataset_name_real_annotated_gmy_all,dataset_name_real_annotated_all_split_gmy]
if dataset_name in dataset_list0:
    myclass='gas'
    img_ext='jpg'
    dataset_mode="composite"
elif dataset_name in dataset_list1:
    myclass='smoke'
    img_ext='png'
    dataset_mode = "real"
else:
    print('dataset name error!!!')


class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path="data/dataset/{}".format(dataset_mode)):##
        print("*"*10)
        print("running in pascal_voc.py...")
        # print("cache_path=",self.cache_path)
        
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        if dataset_name in [dataset_name_real_annotated_all_split_gmy]:
            # image_set = "all"############################################################################################for real_annotated_all_split_gmy
            image_set = "17_env"
        self._image_set = image_set
        if dataset_name==dataset_name_real_annotated_gmy and (self._image_set in ["val","test"]):
          self._image_set="val"
        print("current_image_set={}".format(self._image_set))
        self._devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        os.makedirs(self._devkit_path, exist_ok=True)
        # print("current__devkit_path={}".format(self._devkit_path))

        # self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        # self._classes = ('__background__',  # always index 0
        #                  'aeroplane', 'bicycle', 'bird', 'boat',
        #                  'bottle', 'bus', 'car', 'cat', 'chair',
        #                  'cow', 'diningtable', 'dog', 'horse',
        #                  'motorbike', 'person', 'pottedplant',
        #                  'sheep', 'sofa', 'train', 'tvmonitor')

        # self._data_path = os.path.join(self._devkit_path, 'data', 'IR', self._image_set)



        self._data_path = os.path.join(self._devkit_path, dataset_name, self._image_set)
        # self._data_path = os.path.join(self._devkit_path, dataset_name, "73")##################################################used for test single environment
        print("current_dataset_path={}".format(self._data_path))




        # self._data_path = os.path.join(self._devkit_path, 'real_annotated', self._image_set)#######))
        # self._classes = ('__background__',  # always index 0
        #                  'crazing', 'inclusion', 'patches',
        #                  'pitted_surface', 'rolled-in_scale', 'scratches')
        
        self._classes = ('__background__', '{}'.format(myclass))
        print("current_classes={}".format(self._classes))

        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        self._image_ext = '.{}'.format(img_ext)

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'image', index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'image')

        a = sorted(glob.glob("%s/*.*" % image_set_file))
        image_index = [path.split('/')[-1].split('.')[0].strip() for path in a]

        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        # return '/home/ecust/txx/project/gmy_2080_copy/faster-rcnn.pytorch-pytorch-1.0/faster-rcnn.pytorch-pytorch-1.0'
        return "/workspace/fasterrcnn_txx"###########################


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_folder=os.path.join(self.cache_path, dataset_name)
        if not os.path.exists(cache_folder):
          os.makedirs(cache_folder)
        cache_file = os.path.join(cache_folder,'{}_gt_roidb.pkl'.format(self._image_set))#################
        print('cache_dir=',cache_file)
        if os.path.exists(cache_file):
            print("------")
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, dataset_name + '_selective_search_roidb.pkl')####################

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'valid':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'valid':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in range(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        # filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        filename = os.path.join(self._data_path, 'label', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        # filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        filedir = os.path.join(self._devkit_path, 'result', 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        # print("+"*100)
        print("_do_python_eval...")
        annopath = os.path.join(self._data_path, 'label', '{:s}.xml')
        imagesetfile = os.path.join(self._data_path, 'image')
        # imagesetfile = os.path.join('{}_{}'.format(dataset_name, self._image_set))
        # print("imagesetfile=",imagesetfile)

        # cachedir = os.path.join(self._devkit_path, 'result', 'annotations_cache')
        cache_folder = os.path.join(self.cache_path, dataset_name)
        aps = []
        # The PASCAL VOC metric changed in 2010
        # use_07_metric = True if int(self._year) < 2010 else False
        use_07_metric = False
        # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cache_folder, ovthresh=iou_thresh,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        # print('')
        # print('--------------------------------------------------------------')
        # print('Results computed with the **unofficial** Python eval code.')
        # print('Results should be very close to the official MATLAB eval code.')
        # print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        # print('-- Thanks, The Management')
        # print('--------------------------------------------------------------')
        return np.mean(aps)

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        map=self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)
        return map

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    d = pascal_voc('valid', '2007')
    res = d.roidb
    print(len(d.image_index))
    # from IPython import embed;
    #
    # embed()
