# MultiFashion+MNIST
seed=0
name=CAGrad
method='CAGrad'

CUDA_VISIBLE_DEVICES=0 python ./scripts/train_mnist.py --name $name --method $method --seed $seed
