# Visloc-Retrieval: Library for Image Retrival and Place Recognition

- [Introduction](#introduction)
- [Welcome](#welcome)
- [Train](#train)
- [Results](#results)
- [Licenses](#licenses)
- [Citing](#citing)

## Introduction:

Visloc-Retrieval is a collectio of image retrieval and place recognition models and algorithms for robotics and autonomous system applications.

## Welcome

Welcome to the `Visloc-Retrieval` 

The library can be installed with pip:

```
pip install git+https://github.com/Tarekbouamer/visloc_retrieval.git
```

Main system requirements:
  * Python 3.7
  * CUDA 11.3
  * Faiss library (faiss-gpu 1.7.2)

```
conda create -n loc
conda activate loc
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install -c conda-forge faiss-gpu 
pip install gdown timm==0.5.4
```

Pretrained models can be loaded using `retrieval.create_model`:

```python
import retrieval

extractor = retrieval.create_model('resnet50_c4_gem_1024')
extractor.eval()
```

List available model architectures:
```python
import retrieval

model_names = retrieval.list_models('*resne*t*')
print(model_names)
>>> [
  'resnet50_c4_gem_1024'
  ...
]
```

## Train
The provided train script [this subtext](scripts/train.py) will train a new network from scratch, to resume training add --resume_path and set to a full path, filename and extension to an existing checkpoint file. Note to resume our provided models, first remove the WPCA layers.













### TO DO?
* Add shield IO
* Add results on roxford5k rparis6k.

### Prerequisites

Main system requirements:
  * Python 3.6.9
  * Linux with GCC 7 or 8
  * PyTorch 1.13.0 Torchvision 0.14.0
  * CUDA 11.1
  * Faiss library (faiss-gpu 1.7.2)


### Setup

To install all other dependencies using pip:

```bash
pip install -r requirements.txt
```

Our code is split into two main components: a library containing implementations for the various network modules,
algorithms and utilities, and a set of scripts for training and testing the networks.

The library, called `cirtorch`, can be installed with:
```bash
git clone https://github.com/Tarekbouamer/Image-Retrieval-for-Image-Based-Localization.git
cd Image-Retrieval-for-Image-Based-Localization
python setup.py install
```

## Training

Training involves three main steps: dataset preparation (automatic download), creating a configuration file and running the training
script.

The script downloads [The Retrieval-SfM-120k](https://arxiv.org/abs/1711.02512) trainning dataset and [The Revisiting Oxford and Paris](https://github.com/filipradenovic/revisitop) test sets and benchmarking to `$DataFolder`.

The configuration file is a simple text file in `ini` format. The default value of each configuration parameter, as well as a short description of what it does, is available in
[cirtorch/configuration/defaults/](cirtorch/configuration/defaults/).

To run the training:
```bash
sh scripts/train_globalF.sh 
```

It's also highly recommended to train on multiple GPUs in order to obtain good results.
Training logs, both in text and Tensorboard formats, will be written under `experiments`.


### Results
Table: Large-scale image retrieval results of our models on Revisited Oxford and Paris datasets. 
We evaluate against the Easy, Medium and Hard with the mAP metric.

  | Models       |        | Oxford |        |        | Paris  |        | Download |
  |:------------:|:------:|:------:|:------:|:------:|:------:|:------:|:---------|
  |   mAP        | Easy   | Medium | Hard   | Easy   | Medium | Hard   |          |
  | ResNet50-GeM | 66.20  | 51.78  | 28.76  | 79.28  | 62.35  | 36.66  |[resnet50](https://drive.google.com/file/d/1mZpzcAHLFkeKLKROC4ljT7kuy0AUh6WV/view?usp=sharing)|
