clear
python3 setup.py install

clear

N_GPUS=1

DATA_DIR='/media/loc/ssd_512/VBNS_data/output'

EXPERIMENT='./experiments/sat'

python3 ./scripts/retrieval/train_sat.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config ./image_retrieval/configuration/defaults/sat.ini \
      # --eval 

