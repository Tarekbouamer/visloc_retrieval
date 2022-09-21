clear
python3 setup.py install

clear

N_GPUS=1

# DATA_DIR='/home/loc/VL/Visloc/data'
# DATA_DIR='/media/dl/Data/datasets/'
DATA_DIR='/media/loc/ssd_5126/data'

EXPERIMENT='./experiments/'

python3 ./scripts/retrieval/search_timm_models.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config ./image_retrieval/configuration/defaults/search_timm_models.ini \
      --models_family "resnet"


