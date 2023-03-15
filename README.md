# SDSC4016 Fundamentals of Machine Learning II

SDSC4016 Homework 3:

- [On Kaggle](https://www.kaggle.com/competitions/sdsc4016-fml-hw3/overview)

## Description

Solve a sentiment classification problem (Twitter Review) with LSTM.

## Getting Started

### Dependencies

- Python
  - Python 3.10+
  - Jupyter
  - pytorch

### Install mini-conda and mamba

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
conda install mamba -n base -c conda-forge
```

### Set up conda environment

```bash
mamba create -n 4016hw3
mamba activate 4016hw3
```

### Installing dependencies

```bash
# conda or mamba
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
mamba install -c conda-forge Jupyter ipykernel
mamba install -c conda-forge pandas numpy matplotlib scikit-learn tqdm gensim
```

<!-- ### Code

[Weak Baseline](src/Baseline.ipynb)

[Strong Baseline](src/Modified.Private.ipynb) -->

<!-- ### Dataset

[Training set](data/training/)

[Testing set](data/testing/) -->

<!-- ### Tested Result on Kaggle

[Results on Kaggle](md/kaggle.md) -->

<!-- ### Final Score (Strong Baseline)

- Public: 0.85156
- Private: 0.89874 -->
