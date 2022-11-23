clear
python3 setup.py install

clear

DATA_DIR='/media/dl/Data/datasets/'
DATA_DIR='/media/loc/ssd_5127/tmp/how/how_data/'
MODEL='sfm_resnet18_how_128'

SCALES=1.0
# SCALES=0.7071,1.0,1.4142

export PYTHONPATH=${PYTHONPATH}:$(realpath thirdparty/asmk/)

python3 ./scripts/test.py \
      --data $DATA_DIR \
      --model $MODEL \
      --scales $SCALES \


