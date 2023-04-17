# NYUv2
seed=0
name=Recon
method='Recon'
sub_method='Baseline'
n_epoch=50

CUDA_VISIBLE_DEVICES=0 python ./scripts/train_nyuv2.py --name $name --method $method --sub_method $sub_method --n_epoch $n_epoch --seed $seed
