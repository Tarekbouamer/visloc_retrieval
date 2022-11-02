clear
python3 setup.py install

clear

DATA_DIR='/media/dl/Data/datasets/test/oxford5k/jpg'
DATA_DIR='/media/loc/ssd_5126/tmp/how/how_data/test/oxford5k/jpg'

MODEL='resnet50_gem_2048'
MAX_SIZE=1024

python3 ./scripts/extract.py \
      --data $DATA_DIR \
      --model $MODEL \
      --max_size $MAX_SIZE \


