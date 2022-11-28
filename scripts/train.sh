clear
python3 setup.py install

clear

N_GPUS=1

DATA_DIR='/media/dl/Data/datasets/sfm'
# DATA_DIR='/media/loc/ssd_5127/tmp/how/how_data'

EXPERIMENT='./experiments/'

export PYTHONPATH=${PYTHONPATH}:$(realpath thirdparty/asmk/)

python3 ./scripts/train.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config ./image_retrieval/configuration/defaults/defaults.ini \
      # --eval 


