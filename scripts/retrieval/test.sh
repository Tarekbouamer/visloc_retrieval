clear
python3 setup.py install

clear

N_GPUS=1
DATA_DIR='/home/loc/VL/Visloc/data'
# DATA_DIR='/home/torres/3D/Image-Retrieval-for-Image-Based-Localization/data/'

EXPERIMENT='./experiments/'

Resume='./experiments/retrieval-SfM-120k_resnet50_triplet_m0.50_GeM_Adam_lr1.0e-06_wd1.0e-06_nnum5_bsize10_uevery1_imsize1024/'

python3 -m torch.distributed.launch --nproc_per_node=$N_GPUS ./scripts/test_globalF_all.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --resume $Resume \
      --config ./cirtorch/configuration/defaults/global_config_50.ini \

