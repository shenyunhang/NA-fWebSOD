# Noise-Aware Fully Webly Supervised Object Detection

By [Yunhang Shen](), [Rongrong Ji](), [Zhiwei Chen](), [Xiaopeng Hong](), [Feng Zheng](), [Jianzhuang Liu](), [Mingliang Xu](), [Qi Tian]().

CVPR 2020 Paper.

This project is based on [Detectron](https://github.com/facebookresearch/Detectron).


## Introduction



## License

NA-fWebSOD is released under the [Apache 2.0 license](https://github.com/shenyunhang/NA-fWebSOD/blob/NA-fWebSOD/LICENSE). See the [NOTICE](https://github.com/shenyunhang/NA-fWebSOD/blob/NA-fWebSOD/NOTICE) file for additional details.


## Citing NA-fWebSOD

If you find NA-fWebSOD useful in your research, please consider citing:

```
@inproceedings{NA-fWebSOD_2020_CVPR,
	author = {Shen, Yunhang and Ji, Rongrong and Chen, Zhiwei and Hong, Xiaopeng and Zheng, Feng and Liu, Jianzhuang and Xu, Mingliang and Tian, Qi},
	title = {Noise-Aware Fully Webly Supervised Object Detection},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year = {2020},
}   
```


## Installation

**Requirements:**

- NVIDIA GPU, Linux, Python3.6
- Caffe2 in pytorch v1.3.0, various standard Python packages, and the COCO API; Instructions for installing these dependencies are found below

### Caffe2

Clone the pytorch repository:
```
# pytorch=/path/to/clone/pytorch
git clone https://github.com/pytorch/pytorch.git $pytorch
cd $pytorch
git checkout v1.3.0
git submodule update --init --recursive
```

Install Python dependencies:
```
pip3 install -r $pytorch/requirements.txt
```

Build caffe2:
```
cd $pytorch
sudo USE_OPENCV=On USE_LMDB=On BUILD_BINARY=On python3 setup.py install
```


### Other Dependencies

Install the [COCO API](https://github.com/cocodataset/cocoapi):
```
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Install the [pycococreator](https://github.com/waspinator/pycococreator):
```
pip3 install git+git://github.com/waspinator/pycococreator.git@0.2.0
```


###  NA-fWebSOD

Clone the NA-fWebSOD repository:
```
# NA-fWebSOD=/path/to/clone/NA-fWebSOD
git clone https://github.com/shenyunhang/NA-fWebSOD.git $NA-fWebSOD
cd $NA-fWebSOD
git submodule update --init --recursive
```

Install Python dependencies:
```
pip3 install -r requirements.txt
```

Set up Python modules:
```
make
```

Build the custom C++ operators library:
```
./build_ops.sh
```

### Dataset Preparation

#### Training Data

Download flickr_voc from this [here](https://1drv.ms/u/s!Am1oWgo9554dgQhBFu9FBPeCqjpz?e=WcVh9O) and untar File:
```
tar xvf flickr_voc.tar
ln -s /path/to/clone/flickr_voc $NA-fWebSOD/detectron/datasets/data/flickr_voc
```

Download flickr_coco from this [here](https://1drv.ms/u/s!Am1oWgo9554dgQhBFu9FBPeCqjpz?e=WcVh9O) and untar File:
```
tar xvf flickr_coco.tar
ln -s /path/to/clone/flickr_coco $NA-fWebSOD/detectron/datasets/data/flickr_coco
```

Download flickr_clean from this [here](https://1drv.ms/u/s!Am1oWgo9554dgQhBFu9FBPeCqjpz?e=WcVh9O) and untar File:
```
tar xvf flickr_clean.tar
ln -s /path/to/clone/flickr_clean $NA-fWebSOD/detectron/datasets/data/flickr_clean
```

#### Testing Data

Please follow [this](https://github.com/shenyunhang/NA-fWebSOD/blob/NA-fWebSOD/detectron/datasets/data/README.md#creating-symlinks-for-pascal-voc) to creating symlinks for PASCAL VOC.

Download MCG proposal from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/) to detectron/datasets/data, and transform it to pickle serialization format:
```
cd detectron/datasets/data
tar xvzf MCG-Pascal-Main_trainvaltest_2007-boxes.tgz
cd ../../../
python3 tools/convert_mcg.py voc_2007_train detectron/datasets/data/MCG-Pascal-Main_trainvaltest_2007-boxes detectron/datasets/data/proposals/mcg_voc_2007_train.pkl
python3 tools/convert_mcg.py voc_2007_val detectron/datasets/data/MCG-Pascal-Main_trainvaltest_2007-boxes detectron/datasets/data/proposals/mcg_voc_2007_val.pkl
python3 tools/convert_mcg.py voc_2007_test detectron/datasets/data/MCG-Pascal-Main_trainvaltest_2007-boxes detectron/datasets/data/proposals/mcg_voc_2007_test.pkl
```

Finnally, We have the following directory structure:
```
NA-fWebSOD
|_ detectron
|_ datasets
|_ data
|_ flickr_voc
|_ images
|_ images.json
|_ images.txt
|_ ...
|_ flickr_coco
|_ images
|_ images.json
|_ images.txt
|_ ...
|_ flickr_clean
|_ images
|_ images.json
|_ images.txt
|_ ...
|_ VOC2007
|_ coco
|_ ...
```

### Model Preparation

Download models from this [here](https://1drv.ms/u/s!Am1oWgo9554dgQhBFu9FBPeCqjpz?e=WcVh9O) and untar File:
```
tar xvf models.tar
mv models $NA-fWebSOD
```

Then We have the following directory structure:
```
NA-fWebSOD
|_ models
|  |_ VGG
|  |_ |_ VGG_ILSVRC_16_layers_v1.pkl
|_ ...
```

## Quick Start: Using NA-fWebSOD
### NA-fWebSOD

Flickr voc
```
./scripts/train_wsl.sh --cfg configs/flickr_voc/webly_wsddn_V-16-C5_1x.yaml OUTPUT_DIR experiments/webly_wsddn_v-16_flickr_voc_`date +'%Y-%m-%d_%H-%M-%S'`
```

Flickr clean
```
./scripts/train_wsl.sh --cfg configs/flickr_clean/webly_wsddn_V-16-C5_1x.yaml OUTPUT_DIR experiments/webly_wsddn_v-16_flickr_clean_`date +'%Y-%m-%d_%H-%M-%S'`
```

Flickr coco
```
./scripts/train_wsl.sh --cfg configs/flickr_coco/webly_wsddn_V-16-C5_1x.yaml OUTPUT_DIR experiments/webly_wsddn_v-16_flickr_coco_`date +'%Y-%m-%d_%H-%M-%S'`
```
