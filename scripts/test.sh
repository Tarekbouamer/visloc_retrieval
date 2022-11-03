clear
python3 setup.py install

clear

# DATA_DIR='/media/dl/Data/datasets/'
DATA_DIR='/media/loc/ssd_5126/tmp/how/how_data/'
MODEL='resnet101_c4_gem_1024'

SCALES=0.25,0.3535,0.5,0.7071,1.0,1.4142,2.0

export PYTHONPATH=${PYTHONPATH}:$(realpath thirdparty/asmk/)

python3 ./scripts/test.py \
      --data $DATA_DIR \
      --model $MODEL \
      --scales $SCALES \


