cp checkpoint_100.pth.tar checkpoint.pth.tar
CUDA_VISIBLE_DEVICES=0 python  -u main.py  -a resnet18_ACD --epochs 100 --batch-size 256 --lr_mode LRnorm  /root/autodl-tmp/imagenet --seed 0 --resume checkpoint.pth.tar --lamda_start 1e2 --lamda_end 1e4 --epochs_lamda 9 --lamda_normA_start 1e-6 --lamda_normA_end 1e-3 --batches_per_epoch 5005 --epochs_lamda_normA 9

