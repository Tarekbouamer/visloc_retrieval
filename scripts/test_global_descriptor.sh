DATA_DIR='/media/dl/Data/datasets/'
DATA_DIR='/media/loc/ssd_512/tmp/how/how_data/'
# DATA_DIR='/media/torres/ssd_2tb/sfm'

MODEL='gl18_resnet50_gem_2048'

SCALES=1.0

python3 ./scripts/test_global_descriptor.py \
    --data $DATA_DIR \
    --model $MODEL \
    --scales $SCALES
