clear
python3 setup.py install

clear

# DATA_DIR='/media/dl/Data/datasets/test/oxford5k/jpg'
# DATA_DIR='/media/loc/ssd_5126/tmp/how/how_data/test/oxford5k/jpg'
DATA_DIR='/media/loc/Elements/revisitop/data/datasets/revisitop1m/jpg'

MODEL='sfm_resnet50_gem_2048'
MAX_SIZE=1024
SAVE_PATH='/media/loc/ssd_5126/'

export PYTHONPATH=${PYTHONPATH}:$(realpath thirdparty/asmk/)

python3 ./scripts/extract.py \
      --data $DATA_DIR \
      --model $MODEL \
      --max_size $MAX_SIZE \
      --save_path $SAVE_PATH \



