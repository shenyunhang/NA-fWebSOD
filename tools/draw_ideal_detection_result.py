"""Script to convert Mutiscale Combinatorial Grouping proposal boxes into the Detectron proposal
file format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle
import numpy as np
import scipy.io as sio
import sys
import os
import math

import cv2

from detectron.datasets.json_dataset_wsl import JsonDataset


def gray2jet(f):
    # plot short rainbow RGB
    a = f / 0.25  # invert and group
    X = math.floor(a)  # this is the integer part
    Y = math.floor(255 * (a - X))  # fractional part from 0 to 255
    Z = math.floor(128 * (a - X))  # fractional part from 0 to 128

    if X == 0:
        r = 0
        g = Y
        b = 128 - Z
    elif X == 1:
        r = Y
        g = 255
        b = 0
    elif X == 2:
        r = 255
        g = 255 - Z
        b = 0
    elif X == 3:
        r = 255
        g = 128 - Z
        b = 0
    elif X == 4:
        r = 255
        g = 0
        b = 0
    # opencv is bgr, not rgb
    return (b, g, r)


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    proposal_file = sys.argv[2]
    output_dir = sys.argv[3]
    prefix = ''
    suffix = ''

    ds = JsonDataset(dataset_name)

    roidb = ds.get_roidb(
        gt=True,
        proposal_file=proposal_file,
    )
    print(len(roidb))

    for i in range(len(roidb)):
        b = i
        if i % 1000 == 0:
            print('{}/{}'.format(i + 1, len(roidb)))

        gt_overlaps = roidb[i]['gt_overlaps'].toarray()

        print('gt_overlaps: ', gt_overlaps.shape)

        rois = roidb[i]['boxes'].copy()

        num_rois = roidb[i]['boxes'].shape[0]
        num_rois_this = min(1000, num_rois)

        gt_inds = np.where(roidb[i]['gt_classes'] > 0)[0]
        assert len(gt_inds) > 0, roidb[i]['gt_classes']
        assert len(
            gt_inds
        ) > 0, 'Empty ground truth empty for image is not allowed. Please check.'
        gt_classes = roidb[i]['gt_classes'][gt_inds].copy()
        num_valid_objs = gt_classes.shape[0]
        if num_valid_objs < 2:
            continue
        if 14 not in gt_classes:
            continue
        for o in range(num_valid_objs):
            c = gt_classes[o] - 1
            print(c)

            im_S = cv2.imread(roidb[i]['image'])

            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_' + suffix + '.png')
            cv2.imwrite(file_name, im_S)

            roi_score = gt_overlaps[:, 1:]
            print(roi_score.shape)

            argsort = np.argsort(-np.abs(roi_score[:, c]))
            argsort = argsort[:num_rois_this]
            argsort = argsort[::-1]
            print(argsort.shape)
            for n in range(num_rois_this):
                scale_p = 1.0
                scale_p = 1.0 / roi_score[:, c].max()
                roi = rois[argsort[n]]
                thickness = 4
                if roi_score[argsort[n]][c] * scale_p > 0.8:
                    thickness = 4
                    scale_p = 1.0
                elif roi_score[argsort[n]][c] * scale_p > 0.7:
                    thickness = 3
                    scale_p = 0.7
                elif roi_score[argsort[n]][c] * scale_p > 0.6:
                    thickness = 3
                    scale_p = 0.6
                elif roi_score[argsort[n]][c] * scale_p > 0.5:
                    thickness = 2
                    scale_p = 0.5
                else:
                    thickness = 2
                jet = gray2jet(roi_score[argsort[n]][c] * scale_p)
                cv2.rectangle(im_S, (roi[0], roi[1]), (roi[2], roi[3]), jet,
                              thickness)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_' +
                suffix + '.png')
            cv2.imwrite(file_name, im_S)

            im_S = cv2.imread(roidb[i]['image'])

            for n in range(num_rois_this):
                roi = rois[argsort[n]]
                thickness = 2
                jet = gray2jet(0)
                cv2.rectangle(im_S, (roi[0], roi[1]), (roi[2], roi[3]), jet,
                              thickness)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_bg_' +
                suffix + '.png')
            cv2.imwrite(file_name, im_S)
