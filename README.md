# Visloc-Retrieval: Library for Image Retrival and Place Recognition

## Content
- [Introduction](#introduction)
- [Welcome](#welcome)
- [Datasets](#datasets)
- [Training](#training)
- [Roadmap](#roadmap)
- [Results](#results)
- [Licenses](#licenses)
- [Citing](#citing)

## Introduction:

Visloc-Retrieval library is a collection of an image retrieval and place recognition models and algorithms for robotics and autonomous system applications.

## Welcome

* Welcome to the `Visloc-Retrieval` :sparkles:

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

<p align="right"><a href="#content">:arrow_up:</a></p>

## Datasets
The models are trained in well known public dataset as:
 * Retrieval-SfM-120k-images [retrieval-SfM-120k-images](http://cmp.felk.cvut.cz/cnnimageretrieval/) 
 * Google Landmark [v1](https://www.kaggle.com/datasets/google/google-landmarks-dataset), [v2 and v2_clean](https://github.com/cvdfoundation/google-landmark). 

We evaluate trained models on the [RParis6K and ROxford5K](https://github.com/filipradenovic/revisitop) benchmark and with R1M if possible. The datasets contain 4993 and 6322 images respectively, with 70 query images. Optionally, a set of 1 million distractor images (R1M) can be added to each dataset for large scale benchmarking.

Retrieval-SfM-120k-images and Revisiting Oxford and Paris datasets can be downloaded using `TODO` script or automatically in training script below.

 Note! Other training and evaluation datasets will be added soon !!!




## Training
We trained our models using the [training](scripts/train.py) script with a specific configuration provided in `ini` format as in [default](retrieval/configuration/defaults/default.ini), where you can specify the trainning loss, optimizer, scheduler, training and test datasets, data augmentation policy if needed ... etc

start your training:

```sh
DATA_DIR='/......./data'
EXPERIMENT='./experiments/'
CONFIG='./image_retrieval/configuration/defaults/defaults.ini'

python3 ./scripts/train.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config $CONFIG \
      --eval 

```
<p align="right"><a href="#content">:arrow_up:</a></p>

## Roadmap

- [ ] Add Changelog
- [ ] Add shield IO
- [ ] Add results on roxford5k rparis6k.


<p align="right"><a href="#content">:arrow_up:</a></p>

## Results

### :large_blue_diamond: ROxford5k RParis6k (Single Scale Benchmark)

  | Models               | |       | Oxford|       | |       | Paris |       |
  |----------------------|-|:-----:|:-----:|:-----:|-|:-----:|:-----:|:-----:|
  |                      | | Easy  | Medium| Hard  | | Easy  | Medium| Hard  |
  | resnet50_gem_2048    | | 83.83 | 66.01 | 38.96 | | 91.83 | 77.16 | 55.82 |
  | resnet50_c4_gem_1024 | | 79.22 | 60.53 | 34.30 | | 89.24 | 71.77 | 49.14 |
  | resnet101_gem_2048   | | 81.73 | 65.67 | 39.93 | | 90.79 | 74.85 | 52.67 |
  | resnet101_c4_gem_1024| | 82.12 | 62.81 | 36.56 | | 90.44 | 74.64 | 52.67 |


<p align="right"><a href="#content">:arrow_up:</a></p>






## Licenses

<p align="right"><a href="#content">:arrow_up:</a></p>

## Citing

<p align="right"><a href="#content">:arrow_up:</a></p>
