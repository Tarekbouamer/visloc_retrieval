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
## Datasets
the models are trained in well known public dataset as retrieval-SfM-120k-images [retrieval-SfM-120k-images](http://cmp.felk.cvut.cz/cnnimageretrieval/) and Google Landmark [v1](https://www.kaggle.com/datasets/google/google-landmarks-dataset), [v2 and v2_clean](https://github.com/cvdfoundation/google-landmark). We also evaluatetrained models mAP on the [Revisiting Oxford and Paris](https://github.com/filipradenovic/revisitop) benchmark. Retrieval-SfM-120k-images and Revisiting Oxford and Paris datasets can be downloaded using `TODO` script or automatically in training script below.



## Training
The provided [train](scripts/train.py) script will train a specific model (backbone + head) from our factory on a specific configuration provided in `ini` format as in [default](retrieval/configuration/defaults/default.ini).

In your config file you can specify the trainning loss, optimizer, scheduler, training and test datasets, augmentation ... etc

Run training:

```
DATA_DIR='/......./data'
EXPERIMENT='./experiments/'

python3 ./scripts/train.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config ./image_retrieval/configuration/defaults/defaults.ini \
      --eval 
```













### TO DO?
* Add shield IO
* Add results on roxford5k rparis6k.



### Results
Table: Large-scale image retrieval results of our models on Revisited Oxford and Paris datasets. 
We evaluate against the Easy, Medium and Hard with the mAP metric.

  | Models       |        | Oxford |        |        | Paris  |        | Download |
  |:------------:|:------:|:------:|:------:|:------:|:------:|:------:|:---------|
  |   mAP        | Easy   | Medium | Hard   | Easy   | Medium | Hard   |          |
  | ResNet50-GeM | 66.20  | 51.78  | 28.76  | 79.28  | 62.35  | 36.66  |[resnet50](https://drive.google.com/file/d/1mZpzcAHLFkeKLKROC4ljT7kuy0AUh6WV/view?usp=sharing)|
