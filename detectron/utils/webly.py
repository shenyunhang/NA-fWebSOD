from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import logging
import numpy as np
import os
import math

from caffe2.python import memonger
from caffe2.python import workspace

from caffe2.python import muji
import detectron.utils.c2 as c2_utils

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.modeling import model_builder_wsl


def softmax_surgery(model):
    print('softmax surgery')

    gpu_prefixs = ['gpu_' + str(i) for i in range(cfg.NUM_GPUS)]

    old_ops = model.net._net.op[:]
    num_op = len(model.net._net.op)
    is_end = False

    del model.net._net.op[:]

    gpu_point = {gpu_prefix: -1 for gpu_prefix in gpu_prefixs}
    while (True):
        for gpu_prefix in gpu_prefixs:
            for i, op in enumerate(old_ops):
                if i <= gpu_point[gpu_prefix]:
                    continue

                gpu = op.input[0].split('/')[0]
                if gpu == gpu_prefix:
                    pass
                else:
                    continue

                if op.type == 'Softmax' and 'fc8d_t' in op.input[0]:
                    gpu_point[gpu_prefix] = i
                    # print(op)
                    print('find softmax: ', op.input[0], '\t-->\t',
                          op.output[0])
                    break
                model.net._net.op.extend([op])

                if i == num_op - 1:
                    is_end = True

        if is_end:
            break

        if gpu_point[gpu_prefixs[0]] == -1 or gpu_point[
                gpu_prefixs[1]] == -1 or gpu_point[
                    gpu_prefixs[2]] == -1 or gpu_point[gpu_prefixs[3]] == -1:
            break

        assert old_ops[gpu_point[gpu_prefixs[0]]].input[0].split('/')[
            1] == old_ops[gpu_point[gpu_prefixs[1]]].input[0].split('/')[1]
        assert old_ops[gpu_point[gpu_prefixs[0]]].input[0].split('/')[
            1] == old_ops[gpu_point[gpu_prefixs[2]]].input[0].split('/')[1]
        assert old_ops[gpu_point[gpu_prefixs[0]]].input[0].split('/')[
            1] == old_ops[gpu_point[gpu_prefixs[3]]].input[0].split('/')[1]

        in_blobs = []
        out_blobs = []
        for gpu_prefix in gpu_prefixs:
            in_blob = old_ops[gpu_point[gpu_prefix]].input[0]
            in_blobs.append(in_blob)

            out_blob = old_ops[gpu_point[gpu_prefix]].output[0]
            out_blobs.append(out_blob)
        in_blob_name = in_blobs[0].split('/')[1]
        out_blob_name = out_blobs[0].split('/')[1]

        for gpu_prefix in gpu_prefixs:
            gpu_id = int(gpu_prefix.split('_')[1])
            with c2_utils.CudaScope(gpu_id):
                for i in range(cfg.NUM_GPUS):
                    if gpu_id == i:
                        continue
                    model.net.Copy(
                        in_blobs[i],
                        gpu_prefix + '/' + in_blob_name + '_gpu_' + str(i))
                    model.net.StopGradient(
                        gpu_prefix + '/' + in_blob_name + '_gpu_' + str(i),
                        gpu_prefix + '/' + in_blob_name + '_gpu_' + str(i))
                concat_in_blobs = [
                    gpu_prefix + '/' + in_blob_name + '_gpu_' + str(i)
                    for i in range(cfg.NUM_GPUS)
                ]
                concat_in_blobs[gpu_id] = in_blobs[gpu_id]
                model.net.Concat(
                    concat_in_blobs, [
                        gpu_prefix + '/' + in_blob_name + '_cross',
                        gpu_prefix + '/' + in_blob_name + '_cross_split_info'
                    ],
                    axis=1)

                op = old_ops[gpu_point[gpu_prefix]]
                op.input[0] = gpu_prefix + '/' + in_blob_name + '_cross'
                op.output[0] = gpu_prefix + '/' + out_blob_name + '_cross'
                model.net._net.op.extend([op])

                split_out_blobs = [
                    gpu_prefix + '/' + str(i) + '_useless'
                    for i in range(len(out_blobs))
                ]

                split_out_blobs[gpu_id] = out_blobs[gpu_id]
                model.net.Split([
                    gpu_prefix + '/' + out_blob_name + '_cross',
                    gpu_prefix + '/' + in_blob_name + '_cross_split_info'
                ],
                                split_out_blobs,
                                axis=1)
    return

    num_op = len(model.net._net.op)
    for i, op in enumerate(model.net._net.op):
        print(op)

    exit(0)
