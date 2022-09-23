clear
python3 setup.py install

clear

N_GPUS=1

DATA_DIR='/media/dl/Data/datasets/'
DATA_DIR='/media/loc/ssd_5126/data'

EXPERIMENT='./experiments/'

export PYTHONPATH=${PYTHONPATH}:$(realpath thirdparty/asmk/)

python3 ./scripts/train.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config ./image_retrieval/configuration/defaults/default.ini \
      # --eval 


