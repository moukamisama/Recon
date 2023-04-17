# NYUv2
seed=0
name=MGD
method='MGD'

CUDA_VISIBLE_DEVICES=0 python ./scripts/train_cityscapes.py --name $name --method $method --seed $seed
