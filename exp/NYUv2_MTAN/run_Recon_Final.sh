# Using the modified model for training
seed=0
name=CAGrad_with_Recon # CAGrad_with_Recon, Graddrop_with_Recon, Baseline_with_Recon, MGD_with_Recon, PCGrad_with_Recon
method='CAGrad' # Support CAGrad, Graddrop, Baseline, MGD, PCGrad
alpha=0.4
topK=15
branch_type=branched
conflict_scores_file=./logs/NYUv2+MTAN_Recon/0_Baseline_ep49_lw_cos_S-0.02.json

CUDA_VISIBLE_DEVICES=0 python ./scripts/train_nyuv2.py --name $name --method $method --topK $topK --branch_type $branch_type --seed $seed --alpha $alpha --conflict_scores_file $conflict_scores_file
