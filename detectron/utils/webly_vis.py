from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os
import math
from PIL import Image, ImageDraw, ImageFont

from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir


def vis_training(cur_iter):
    prefix = ''
    if cfg.WEBLY.MINING:
        prefix = 'mining_'
    if not (cfg.WSL.DEBUG or
            (cfg.WSL.SAMPLE and cur_iter % cfg.WSL.SAMPLE_ITER == 0)):
        return

    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    sample_dir = os.path.join(output_dir, 'webly_sample')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for gpu_id in range(cfg.NUM_GPUS):
        data_ids = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'data_ids'))
        ims = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'data'))
        labels_oh = workspace.FetchBlob('gpu_{}/{}'.format(
            gpu_id, 'labels_oh'))
        im_score = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'cls_prob'))
        roi_score = workspace.FetchBlob('gpu_{}/{}'.format(
            gpu_id, prefix + 'rois_pred'))
        # roi_score_softmax = workspace.FetchBlob('gpu_{}/{}'.format(
        # gpu_id, prefix + 'rois_pred_softmax'))
        rois = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, prefix + 'rois'))
        # anchor_argmax = workspace.FetchBlob('gpu_{}/{}'.format(
        # gpu_id, 'anchor_argmax'))

        preffix = 'iter_' + str(cur_iter) + '_gpu_' + str(gpu_id)
        save_im(labels_oh, im_score, ims, cfg.PIXEL_MEANS, preffix, sample_dir)

        save_rois(labels_oh, im_score, roi_score, ims, rois, cfg.PIXEL_MEANS,
                  preffix, '', sample_dir)

        # continue
        if cfg.WEBLY.ENTROPY:
            pass
        else:
            continue

        class_weight = workspace.FetchBlob('gpu_{}/{}'.format(
            gpu_id, prefix + 'rois_class_weight'))
        rois_pred_hatE = workspace.FetchBlob('gpu_{}/{}'.format(
            gpu_id, prefix + 'rois_pred_hatE'))
        rois_pred_E = workspace.FetchBlob('gpu_{}/{}'.format(
            gpu_id, prefix + 'rois_pred_E'))
        y_logN__logy = workspace.FetchBlob('gpu_{}/{}'.format(
            gpu_id, prefix + 'rois_pred_y_logN__logy'))
        save_entropy(labels_oh, im_score, class_weight, roi_score, ims, rois,
                     cfg.PIXEL_MEANS, preffix, '', sample_dir, rois_pred_hatE,
                     rois_pred_E, y_logN__logy)


def save_im(labels_oh, im_score, ims, pixel_means, prefix, output_dir):
    batch_size, num_classes = im_score.shape
    for b in range(batch_size):
        for c in range(num_classes):
            # if labels_oh[b][c] == 0.0:
                # continue
            if im_score[b][c] < 0.1:
                continue
            im = ims[b, :, :, :].copy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            im += pixel_means
            im = im.astype(np.uint8)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '.png')
            cv2.imwrite(file_name, im)


def save_rois(labels_oh, im_score, roi_score, ims, rois, pixel_means, prefix,
              suffix, output_dir):
    num_rois, num_classes = roi_score.shape
    batch_size, _, height, weight = ims.shape

    has_bg = False

    num_rois_this = min(500, num_rois)
    for b in range(batch_size):
        for c in range(num_classes):
            # if labels_oh[b][c] == 0.0:
                # continue
            if im_score[b][c] < 0.1:
                if has_bg:
                    continue
                has_bg = True
            im = ims[b, :, :, :].copy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            im += pixel_means
            im = im.astype(np.uint8)
            im_S = im.copy()
            im_A = im.copy()

            argsort = np.argsort(-np.abs(roi_score[:, c]))
            argsort = argsort[:num_rois_this]
            argsort = argsort[::-1]
            if im_score[b][c] < 0.1:
                scale_p = 1.0
            else:
                scale_p = 1.0 / roi_score[:, c].max()
            for n in range(num_rois_this):
                roi = rois[argsort[n]]
                if roi[0] != b:
                    continue
                if roi_score[argsort[n]][c] * scale_p < 0.4:
                    thickness = 3
                else:
                    thickness = 6
                jet = gray2jet(roi_score[argsort[n]][c] * scale_p)
                cv2.rectangle(im_S, (roi[1], roi[2]), (roi[3], roi[4]), jet, thickness)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_' +
                suffix + '.png')
            cv2.imwrite(file_name, im_S)

            continue
            num_anchors = anchor_argmax.shape[0]
            for n in range(num_rois):
                roi = rois[n]
                if roi[0] != b:
                    continue

                for a in range(num_anchors):
                    if anchor_argmax[a][n] == 1.0:
                        break

                jet = gray2jet(1.0 * a / num_anchors)
                cv2.rectangle(im_A, (roi[1], roi[2]), (roi[3], roi[4]), jet, 1)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_A_' +
                suffix + '.png')
            cv2.imwrite(file_name, im_A)


def save_entropy(labels_oh, im_score, class_weight, roi_score, ims, rois,
                 pixel_means, prefix, suffix, output_dir, rois_pred_hatE,
                 rois_pred_E, y_logN__logy):
    num_rois, num_classes = roi_score.shape
    batch_size, _, height, weight = ims.shape
    rois_pred_E_sum = np.sum(rois_pred_E, axis=0).reshape(1, -1)
    E_sum_norm = np.true_divide(rois_pred_E_sum, y_logN__logy)
    E_sum_norm = np.where(E_sum_norm > 1., 1., E_sum_norm)
    E_class_weight = 1 - E_sum_norm
    for b in range(batch_size):
        for c in range(num_classes):
            if labels_oh[b][c] == 0.0 and im_score[b][c] < 0.1:
                continue
            im = ims[b, :, :, :].copy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            im += pixel_means
            im = im.astype(np.uint8)
            im_S = im.copy()
            im_A = im.copy()
            im_hatE = im.copy()
            im_E = im.copy()
            _NUM = 10
            argsort_roi = np.argsort(roi_score[:, c])[::-1]
            argsort_hatE = np.argsort(rois_pred_hatE[:, c])[::-1]
            argsort_E = np.argsort(rois_pred_E[:, c])[::-1]
            if len(argsort_roi) >= _NUM:
                _NUM = 10
            else:
                _NUM = len(argsort_roi)
            argsort_roi = argsort_roi[:_NUM][::-1]
            argsort_hatE = argsort_hatE[:_NUM][::-1]
            argsort_E = argsort_E[:_NUM][::-1]
            argsort_hatE = argsort_roi
            argsort_E = argsort_roi

            scale_p = 1.0 / roi_score[:, c].max()
            scale_p = 1.0
            for n in range(_NUM):
                roi = rois[argsort_roi[n]]
                hatE_roi = rois[argsort_hatE[n]]
                E_roi = rois[argsort_E[n]]
                if roi[0] != b:
                    continue

                # draw roi
                jet = gray2jet(roi_score[argsort_roi[n]][c] * scale_p)
                bgr = jet
                rgb = (jet[2], jet[1], jet[0])
                # roi location
                cv2.rectangle(im_S, (roi[1], roi[2]), (roi[3], roi[4]),
                              bgr,
                              2,
                              lineType=cv2.LINE_AA)

                text = "{:.4f}".format(roi_score[argsort_roi[n]][c])
                im_S = putText_with_TNR(im_S, int(roi[1]), int(roi[2]), 15,
                                        jet, rgb, text)

                if hatE_roi[0] != b:
                    continue
                # draw rois_pred_hatE
                # jet = gray2jet(rois_pred_hatE[argsort_hatE[n]][c] * scale_p)
                # bgr = jet
                # rgb = (jet[2], jet[1], jet[0])
                # roi location
                cv2.rectangle(im_hatE, (hatE_roi[1], hatE_roi[2]),
                              (hatE_roi[3], hatE_roi[4]),
                              bgr,
                              2,
                              lineType=cv2.LINE_AA)
                # put Text hat_E
                text = "{:.4f}".format(rois_pred_hatE[argsort_hatE[n]][c])
                im_hatE = putText_with_TNR(im_hatE, int(hatE_roi[1]),
                                           int(hatE_roi[2]), 15, jet, rgb,
                                           text)

                if E_roi[0] != b:
                    continue
                # draw rois_pred_E
                # jet = gray2jet(rois_pred_E[argsort_E[n]][c] * scale_p)
                # bgr = jet
                # rgb = (jet[2], jet[1], jet[0])
                # roi location
                cv2.rectangle(im_E, (E_roi[1], E_roi[2]), (E_roi[3], E_roi[4]),
                              bgr,
                              2,
                              lineType=cv2.LINE_AA)
                # put Text E
                text = "{:.4f}".format(rois_pred_E[argsort_E[n]][c])
                im_E = putText_with_TNR(im_E, int(E_roi[1]), int(E_roi[2]), 15,
                                        jet, rgb, text)

            # write im_score
            text = "{:.4f}".format(im_score[b][c])
            im_S = putText_with_TNR(im_S, 0, 0, 20, (0, 140, 255),
                                    (255, 255, 255), text)

            # write class_weight
            text = "{:.4f}".format(class_weight[b][c])
            im_hatE = putText_with_TNR(im_hatE, 0, 0, 20, (0, 140, 255),
                                       (255, 255, 255), text)

            # write class_weight
            text = "{:.4f}".format(E_class_weight[b][c])
            im_E = putText_with_TNR(im_E, 0, 0, 20, (0, 140, 255),
                                    (255, 255, 255), text)

            file_name_roi = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_roi' +
                suffix + '.png')
            cv2.imwrite(file_name_roi, im_S)

            file_name_hatE = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) +
                '_hatE' + suffix + '.png')
            cv2.imwrite(file_name_hatE, im_hatE)

            file_name_E = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_E' +
                suffix + '.png')
            cv2.imwrite(file_name_E, im_E)


def dump_proto_files(model, output_dir):
    """Save prototxt descriptions of the training network and parameter
    initialization network."""
    with open(os.path.join(output_dir, model.net.Proto().name), 'w') as fid:
        fid.write(str(model.net.Proto()))
    with open(os.path.join(output_dir,
                           model.param_init_net.Proto().name), 'w') as fid:
        fid.write(str(model.param_init_net.Proto()))


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


def putText_with_TNR(img, x, y, size, fontColor, bgColor, string):
    thickness = 2
    font_scale = 1.1
    font = cv2.FONT_HERSHEY_SIMPLEX
    s = cv2.getTextSize(string, font, font_scale, thickness)

    cv2.rectangle(
        img,
        (x + thickness, y + thickness),
        (x + thickness + s[0][0] + 2, y + thickness + s[0][1] + 2),
        # (0, 140, 255),
        fontColor,
        cv2.FILLED,
        lineType=cv2.LINE_AA)

    position = (x + thickness + 1, y + thickness + s[0][1] + 1)
    cv2.putText(img, string, position, font, font_scale, (255, 255, 255),
                thickness, cv2.LINE_AA)

    return img

    # from OpenCV to PIL
    font = "/home/chenzhiwei/Documents/myFonts/timesnewroman.ttf"
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(font, size)
    position = (x + 3, y - 2)
    draw = ImageDraw.Draw(img_PIL)
    offsetx, offsety = font.getoffset(string)
    width, height = font.getsize(string)
    draw.rectangle((offsetx + x + 2, offsety + y - 3, offsetx + x + width + 3,
                    offsety + y + height - 3),
                   fill=bgColor)
    draw.text(position, string, font=font, fill=fontColor)
    # back to OpenCV type
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img_OpenCV
