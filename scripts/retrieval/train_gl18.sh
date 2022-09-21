clear
python3 setup.py install

clear

N_GPUS=1

DATA_DIR='/media/dl/Data/datasets/gl18'

EXPERIMENT='./experiments/'

python3 ./scripts/retrieval/train.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config ./image_retrieval/configuration/defaults/gl18.ini \
      # --eval 
