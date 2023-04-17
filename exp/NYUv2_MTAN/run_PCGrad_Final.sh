# NYUv2
seed=0
name=PCGrad
method='PCGrad'

CUDA_VISIBLE_DEVICES=0 python ./scripts/train_nyuv2.py --name $name --method $method --seed $seed
