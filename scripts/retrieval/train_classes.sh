clear
python3 setup.py install

clear

N_GPUS=1

DATA_DIR='/home/torres/3D/Image-Retrieval-for-Image-Based-Localization/data/'

EXPERIMENT='./experiments/'

Resume='./experiments/retrieval-SfM-120k_resnet_4_50_triplet_m0.50_GeM_Adam_lr1.0e-06_wd1.0e-04_nnum5_bsize5_uevery1_imsize1024_/model_last.pth.tar'

python3 ./scripts/retrieval/train_classes.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config ./image_retrieval/configuration/defaults/default.ini \
      # --resume $Resume
      # --eval 

