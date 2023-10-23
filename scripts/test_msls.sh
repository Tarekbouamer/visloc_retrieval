
DATA_DIR='/media/torres/data_5tb/datasets/'
DATA_DIR='/media/loc/data_5tb/datasets/'

MODEL='eigenplace_resnet101_gem_2048'

SCALES=1.0

python3 ./scripts/test_msls.py \
    --data $DATA_DIR \
    --model $MODEL \
    --scales $SCALES