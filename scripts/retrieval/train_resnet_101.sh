clear
python3 setup.py install

clear

N_GPUS=1
DATA_DIR='/home/loc/VL/Visloc/data'
# DATA_DIR='/home/torres/3D/Image-Retrieval-for-Image-Based-Localization/data/'
# DATA_DIR='/media/dl/Data/datasets'

EXPERIMENT='./experiments/'


Resume='./experiments/'
python3 ./scripts/retrieval/train.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config ./image_retrieval/configuration/defaults/resnet_101.ini \
      # --eval 
