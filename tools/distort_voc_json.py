from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle as pickle
import numpy as np
import scipy.io as sio
import sys
import os
import random

import json

from detectron.utils.io import save_object

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    p1 = 0.1
    p2 = 1.0

    with open(input_file, 'r') as f:
        datastore = json.load(f)

    print('------------------------------------------------------------------')
    print(datastore.keys())
    print(datastore['images'][0])
    print(len(datastore['images']))
    print(datastore['annotations'][0])
    print(len(datastore['annotations']))
    print(datastore['categories'][0])
    print(len(datastore['categories']))
    print('------------------------------------------------------------------')

    num_classes = len(datastore['categories'])

    for i in range(len(datastore['images'])):
        # print(datastore['images'][i])

        image_id = datastore['images'][i]['id']

        idxs = []
        for j in range(len(datastore['annotations'])):
            if datastore['annotations'][j]['image_id'] == image_id:
                # print(image_id)
                pass
            else:
                continue

            idxs.append(j)

        for idx in idxs:
            if random.random() >= p1:
                continue
            else:
                pass

            datastore['annotations'][idx]['category_id'] = random.randint(
                0, num_classes - 1)

        if random.random() >= p2:
            continue
        else:
            pass

        keep_idx = random.choice(idxs)
        for idx in reversed(idxs):
            if idx == keep_idx:
                continue
            else:
                pass

            del datastore['annotations'][idx]

    print('------------------------------------------------------------------')
    print(datastore.keys())
    print(datastore['images'][0])
    print(len(datastore['images']))
    print(datastore['annotations'][0])
    print(len(datastore['annotations']))
    print(datastore['categories'][0])
    print(len(datastore['categories']))
    print('------------------------------------------------------------------')

    with open(output_file, 'w') as f:
        json.dump(datastore, f)
