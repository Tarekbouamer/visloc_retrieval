clear
python3 setup.py install

clear

DATA_DIR='/media/dl/Data/datasets/'
# DATA_DIR='/media/loc/ssd_5127/tmp/how/how_data/'
MODEL='resnet50_c4_how'
# MODEL='resnet18_how'

SCALES=0.7071,1.0,1.4142
# SCALES=1.0
# SCALES=2.0,1.414,1.0,0.707,0.5,0.353,0.25

export PYTHONPATH=${PYTHONPATH}:$(realpath thirdparty/asmk/)

python3 ./scripts/test.py \
      --data $DATA_DIR \
      --model $MODEL \
      --scales $SCALES \


