clear
python3 setup.py install

clear

N_GPUS=1

DATA_DIR='/media/loc/HDD/'
DATA_DIR='/media/loc/ssd_512/data'

EXPERIMENT='./experiments/'

Resume='./experiments/retrieval-SfM-120k_resneck34_triplet_m0.50_GeM_Adam_lr5.0e-07_wd1.0e-06_nnum5_bsize5_uevery1_imsize1024/model_last.pth.tar'

python3 ./scripts/retrieval/train_resneck.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config ./image_retrieval/configuration/defaults/resneck.ini \
      --eval 
