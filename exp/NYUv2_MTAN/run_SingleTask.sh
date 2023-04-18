#seed=0
#name=Semantic
#task='semantic'
#method='Baseline'
#
#CUDA_VISIBLE_DEVICES=1 python ./scripts/train_nyuv2.py --name $name --method $method --seed $seed --task $task

#seed=0
#name=Depth
#task='depth'
#method='Baseline'
#
#CUDA_VISIBLE_DEVICES=1 python ./scripts/train_nyuv2.py --name $name --method $method --seed $seed --task $task
#

seed=0
name=Normal
task='normal'
method='Baseline'

CUDA_VISIBLE_DEVICES=1 python ./scripts/train_nyuv2.py --name $name --method $method --seed $seed --task $task