clear
python3 setup.py install

clear

N_GPUS=1

DATA_DIR='/media/dl/Data/datasets/gl18'

python3 ./scripts/retrieval/prepare_gl18.py \
      --data $DATA_DIR \

