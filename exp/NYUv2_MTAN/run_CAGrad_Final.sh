# NYUv2
seed=0
name=CAGrad
method='CAGrad'
alpha=0.4

CUDA_VISIBLE_DEVICES=0 python ./scripts/train_nyuv2.py --name $name --method $method --seed $seed --alpha $alpha
