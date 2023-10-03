N_GPUS=1

DATA_DIR='/media/dl/Data/datasets/sfm'
DATA_DIR='/media/loc/ssd_512/tmp/how/how_data/'

EXPERIMENT='./experiments/'

python3 ./scripts/train.py \
    --directory $EXPERIMENT \
    --data $DATA_DIR \
    --local_rank 0 \
    --config ./retrieval/configuration/default.yaml \
    --eval
