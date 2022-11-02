clear
python3 setup.py install

clear

DATA_DIR='/media/dl/Data/datasets/test/oxford5k/jpg'
MODEL='resnet50_c4_gem_1024'
MAX_SIZE=100

python3 ./scripts/extract.py \
      --data $DATA_DIR \
      --model $MODEL \
      --max_size $MAX_SIZE \


