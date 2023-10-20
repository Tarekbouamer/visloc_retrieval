
DATA_DIR='/media/torres/data_5tb/datasets/'

MODEL='mapillary_vgg16_patchnetvlad_4096'

SCALES=1.0

python3 ./scripts/test_msls.py \
    --data $DATA_DIR \
    --model $MODEL \
    --scales $SCALES