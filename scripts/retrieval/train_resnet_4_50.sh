clear
python3 setup.py install

clear

N_GPUS=1
# DATA_DIR='/home/loc/VL/Visloc/data'
DATA_DIR='/media/dl/Data/datasets/'
# DATA_DIR='/media/loc/ssd_512/data'

EXPERIMENT='./experiments/'


Resume='./experiments/res4_50/test_model_best.pth.tar'

python3 ./scripts/retrieval/train.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config ./image_retrieval/configuration/defaults/resnet_4_50.ini \
      # --eval 
