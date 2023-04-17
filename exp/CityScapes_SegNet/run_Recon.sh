# CityScapes
seed=0
name=Recon
method='Recon'
sub_method='Baseline'
n_epoch=40

CUDA_VISIBLE_DEVICES=2 python ./scripts/train_cityscapes.py --name $name --method $method --sub_method $sub_method --n_epoch $n_epoch --seed $seed
