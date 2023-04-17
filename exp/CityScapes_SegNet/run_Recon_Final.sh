# Using the modified model for training
seed=0
name=Baseline_with_Recon # CAGrad_with_Recon, Graddrop_with_Recon, Baseline_with_Recon, MGD_with_Recon, PCGrad_with_Recon
method='Baseline' # Support CAGrad, Graddrop, Baseline, MGD, PCGrad
topK=40
branch_type=branched
conflict_scores_file=./logs/CityScapes+SegNet_Recon/0_Baseline_ep39_lw_cos_S0.0.json

CUDA_VISIBLE_DEVICES=2 python ./scripts/train_cityscapes.py --name $name --method $method --topK $topK --branch_type $branch_type --seed $seed --conflict_scores_file $conflict_scores_file
