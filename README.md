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
  - gensim

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
mamba install -c conda-forge pandas numpy scikit-learn gensim
```

### Code

#### Weak Baseline

- [Jupyter Notebook](https://github.com/CityU-SDSC4016-2022/SDSC4016-hw3/blob/notebook/src/HW3_Baseline.ipynb)

#### Strong Baseline

1. You can run it by ```script.sh```

2. Or you can run it by the following command:

    ```bash
    source ~/miniconda3/etc/profile.d/conda.sh
    source ~/miniconda3/etc/profile.d/mamba.sh
    mamba activate 4016hw3
    python src/script.py
    ```

### Dataset

[Training set](data/Train_label.txt)

[Testing set](data/Test.txt)

<!-- ### Tested Result on Kaggle

[Results on Kaggle](md/kaggle.md) -->

### Final Score (Strong Baseline)

- Public: 0.87228
- Private: 0.87208
