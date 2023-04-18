#seed=0
#name=SingleTask_T1
#task='T1'
#method='Baseline'
#
#CUDA_VISIBLE_DEVICES=1 python ./scripts/train_mnist.py --name $name --method $method --seed $seed --task $task


seed=0
name=SingleTask_T2
task='T2'
method='Baseline'

CUDA_VISIBLE_DEVICES=1 python ./scripts/train_mnist.py --name $name --method $method --seed $seed --task $task