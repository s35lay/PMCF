# PMCF: Self-Supervised Learning for Multi-Level Structure-Aware Recommendation


## Environment
To accelerate PPR calculations with GPU, it is recommended to install [nx-cugraph](https://github.com/rapidsai/cugraph/blob/branch-24.10/python/nx-cugraph/README.md). The implementation for PMCF is under the following development environment:
* python=3.10.8
* torch=2.1.2+cu118
* torch-sparse=0.6.18 
* networkx=3.3
* numpy=1.26.4
* scipy=1.14.0

## Datasets
We utilize three datasets for evaluating PMCF: <i>Yelp, Gowalla, </i>and <i>Amazon</i>. Our evaluation follows the common implicit feedback paradigm. The datasets are divided into training set, validation set and test set by 70:5:25.
| Dataset | \# Users | \# Items | \# Interactions | Interaction Density |
|:-------:|:--------:|:--------:|:---------------:|:-------:|
|Yelp   |$42,712$|$26,822$|$182,357$|$1.6\times 10^{-4}$|
|Gowalla|$25,557$|$19,747$|$294,983$|$5.9\times 10^{-4}$|
|Amazon |$76,469$|$83,761$|$966,680$|$1.5\times 10^{-4}$|

## Usage
Switch the working directory to `methods/PMCF/`. The un-specified hyperparameters in the commands are set as default.

* Gowalla
```
python Main.py --data gowalla --lr 3e-4
```
* Yelp
```
python Main.py --data yelp --reg 2e-4 --epoch 150 --gcn_layer 5
```
* Amazon
```
python Main.py --data amazon --latdim 72 --head 8 --gcn_layer 3
```
