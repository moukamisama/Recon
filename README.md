 # Recon [[Paper Link]](https://arxiv.org/abs/2302.11289)

### Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning
[Guangyuan Shi](https://scholar.google.com/citations?user=fL_osukAAAAJ&hl=en), [Qimai Li](https://scholar.google.com/citations?user=i6yDLl8AAAAJ&hl=en&oi=sra), [Wenlong Zhang](https://scholar.google.com/citations?user=UnMImiUAAAAJ&hl=en&oi=sra), Jiaxin Chen and [Xiao-Ming Wu](https://www4.comp.polyu.edu.hk/~csxmwu/)

#### BibTeX
    @article{shi2023recon,
      title={Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning},
      author={Shi, Guangyuan and Li, Qimai and Zhang, Wenlong and Chen, Jiaxin and Wu, Xiao-Ming},
      journal={arXiv preprint arXiv:2302.11289},
      year={2023}
    }

## Updates
- ✅ 2023-04-17: Release the first version of the paper at Arxiv.
- ✅ 2022-04-17: Release the first version of codes and configs of Recon (including the implementation of [CAGrad](https://proceedings.neurips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf), [PCGrad](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf), [Graddrop](https://proceedings.neurips.cc/paper/2020/file/16002f7a455a94aa4e91cc34ebdb9f2d-Paper.pdf) and [MGDA](https://proceedings.neurips.cc/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf)).
- ✅ 2022-04-19: Upload the training scripts of Single-Task Learning Baseline.
- 🚧 **(To do)** Upload the training codes and configs on dataset PASCAL-Context and CelebA.
- 🚧 **(To do)** Upload implementations of [BMTAS](https://arxiv.org/pdf/2008.10292.pdf) and [RotoGrad](https://arxiv.org/pdf/2103.02631.pdf).

## Overview
<img src="https://raw.githubusercontent.com/moukamisama/Recon/master/overview.png" width="600"/>

## Dependencies and Installation
- [Python](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/)

1. Clone repo

    ```bash
    git clone https://github.com/moukamisama/FS-IL.git
    ```

2. Install [wandb](https://docs.wandb.com/quickstart) 

## Downloading the Datasets
- Refer to the README file in dataset folder.

## Training The Baselines
- Refer to the *./exp/* folder for the bash scripts of all baseline models on different datasets.
 For example, to train CAGrad on MultiFashion+MNIST datasets
```
./exp/MultiFashion+MNIST/run_CAGrad.sh
```

## Training Recon
- We provide the bash scripts of Recon on different datasets in the *./exp/* folder.
1. For example, to train Recon on MultiFashion+MNIST datasets, first we need to run the following codes for **calculating the cos similarity between each pair of shared layers**:
```
./exp/MultiFashion+MNIST/run_Recon.sh
```
2. Then we need to run the following code for **Calculating the S-conflict Score of each layer and obtain the layers permutation**:
```
./exp/MultiFashion+MNIST/calculate_Sconflict.sh
```
3. **Training the modified model**: Pre-calculated layer permutations are provided in *./logs/*. You can skip the first two steps and directly run the following command to train the modified model:
```
./exp/MultiFashion+MNIST/run_Recon_Final.sh 
```

- Evaluation results can be seen in the logger or wandb. In the paper, we repeat the experiments with **3 different seeds** for each dataset, and the **average results of the last iteration** are reported.

## For Different Datasets
- Our modified model can be easily applied to other datasets. The **layer permutations** we obtained can sometimes be directly used for other datasets. Tuning the hyperparameters (e.g., topK in the third procedure) directly on different datasets can lead to better performance.

- Generating specific models for different datasets leads to better performance.