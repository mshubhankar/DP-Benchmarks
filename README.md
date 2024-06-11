# DP Benchmarks

Prerequisites
===
This project runs on Python 3.10 and necessary installation packages are in requirements.txt
* Run `pip install -r requirements.txt` to install all necessary packages

Conda can be also be used to building environment

Run `conda env create -f=environment.yml
$ conda activate dpbenchmarks`

This repository also uses the functional module of Opacus and hence needs opacus to be installed from scratch. 
This can be done using:

```
git clone https://github.com/pytorch/opacus.git
cd opacus
pip install -e .
```

Datasets
===

Datasets can be obtained from the following links:
1. [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
2. [EyePACS](https://paperswithcode.com/dataset/kaggle-eyepacs)


How to run
===
The CONFIG.py can be used to tweak the experiments. 

Run using `python main.py` 
