# Using the modified model for training
seed=0
name=Baseline_with_Recon # CAGrad_with_Recon, Graddrop_with_Recon, Baseline_with_Recon, MGD_with_Recon, PCGrad_with_Recon
method='Baseline' # Support CAGrad, Graddrop, Baseline, MGD, PCGrad
topK=25
branch_type=branched
conflict_scores_file=./logs/multiFashion+MNIST_Recon/0_Baseline_ep39_lw_cos_S-0.1.json

CUDA_VISIBLE_DEVICES=1 python ./scripts/train_mnist.py --name $name --method $method --topK $topK --branch_type $branch_type --seed $seed --conflict_scores_file $conflict_scores_file