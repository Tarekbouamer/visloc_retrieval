clear
python3 setup.py install

clear

N_GPUS=1

DATA_DIR='/media/dl/Data/datasets/'

EXPERIMENT='./experiments/'

export PYTHONPATH=${PYTHONPATH}:$(realpath thirdparty/asmk/)

Resume=
python3 ./scripts/extract.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config ./image_retrieval/configuration/defaults/default.ini \
      --resume 
      # --eval 


