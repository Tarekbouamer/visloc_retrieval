![demo_vid](assets/VisLoc-logos.jpeg)

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
      --eval                # evaluate during training and save best model

```

## Testing
You can evaluate our pretrained models using [test](scripts/test.py) script and default configuration [config.ini](retrieval/configuration/defaults/test.ini) file. The evaluation script extracts image features with a `max_size=1024` in a single scale, multi-scale feature extraction and evaluation is also supported and the list of scales can be intoduced as (`--scales 0.7071,1.0,1.4142`). Our results are presented in [Results](#results) section.

```sh
DATA_DIR='/......./data'
MODEL='resnet50_gem_2048'
SCALES=0.7071,1.0,1.4142

python3 ./scripts/test.py \
      --data $DATA_DIR \
      --model $MODEL \
      --scales $SCALES \
```

<p align="right"><a href="#content">:arrow_up:</a></p


## Roadmap

- [ ] Add Changelog
- [ ] Add shield IO
- [ ] Add results on roxford5k rparis6k.

<p align="right"><a href="#content">:arrow_up:</a></p>

## Results

### 

<details><summary> Global Benchmark</summary>
  
  | Models                  | |     | ROxford5k |     | |     | RParis6k |      |
  |-------------------------|-|:-----:|:-----:|:-----:|-|:-----:|:-----:|:-----:|
  |                         | | Easy  | Medium| Hard  | | Easy  | Medium| Hard  |
  | resnet50_gem_2048       | | 83.83 | 66.01 | 38.96 | | 91.83 | 77.16 | 55.82 |
  | resnet50_c4_gem_1024    | | 79.22 | 60.53 | 34.30 | | 89.24 | 71.77 | 49.14 |
  | resnet101_gem_2048      | | 82.80 | 66.26 | 40.39 | | 91.29 | 75.23 | 53.21 |
  | resnet101_c4_gem_1024   | | 82.12 | 62.81 | 36.56 | | 90.44 | 74.64 | 52.67 |
  | gl18_resnet101_gem_2048 | | 81.79 | 65.58 | 40.72 | | 91.38 | 76.71 | 56.63 |

</details>

#### :blue_square: Single-Scale Benchmark

  | Models                  | |     | ROxford5k |     | |     | RParis6k |      |
  |-------------------------|-|:-----:|:-----:|:-----:|-|:-----:|:-----:|:-----:|
  |                         | | Easy  | Medium| Hard  | | Easy  | Medium| Hard  |
  | resnet50_gem_2048       | | 83.83 | 66.01 | 38.96 | | 91.83 | 77.16 | 55.82 |
  | resnet50_c4_gem_1024    | | 79.22 | 60.53 | 34.30 | | 89.24 | 71.77 | 49.14 |
  | resnet101_gem_2048      | | 82.80 | 66.26 | 40.39 | | 91.29 | 75.23 | 53.21 |
  | resnet101_c4_gem_1024   | | 82.12 | 62.81 | 36.56 | | 90.44 | 74.64 | 52.67 |
  | gl18_resnet101_gem_2048 | | 81.79 | 65.58 | 40.72 | | 91.38 | 76.71 | 56.63 |


#### :orange_square: Multi-Scale Benchmark

  | Models                  | |     | ROxford5k |     | |     | RParis6k |      |
  |-------------------------|-|:-----:|:-----:|:-----:|-|:-----:|:-----:|:-----:|
  |                         | | Easy  | Medium| Hard  | | Easy  | Medium| Hard  |
  | resnet50_gem_2048       | | 84.96 | 67.19 | 40.45 | | 92.67 | 78.39 | 57.84 |
  | resnet50_c4_gem_1024    | | 80.99 | 61.90 | 34.90 | | 90.20 | 72.58 | 49.98 |
  | resnet101_gem_2048      | | 83.65 | 66.88 | 40.60 | | 92.11 | 76.63 | 55.11 |
  | resnet101_c4_gem_1024   | | 83.94 | 64.41 | 38.09 | | 91.66 | 76.70 | 55.28 |
  | gl18_resnet101_gem_2048 | | 84.76 | 68.05 | 43.42 | | 93.25 | 79.75 | 61.14 |

  :information_source: Moreover, we use 3 image scales `[ 0.7071, 1.0, 1.4142 ]` to extract global discriptors to benchmark our trained model, with a minimum size `100` and a maximum area `2000*2000`.

<p align="right"><a href="#content">:arrow_up:</a></p>






## Licenses

<p align="right"><a href="#content">:arrow_up:</a></p>

## Citing

<p align="right"><a href="#content">:arrow_up:</a></p>
