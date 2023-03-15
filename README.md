# Two-Stream Modeling for Egocentric Interaction Recognition.

Code repository for our solution to the [H2O-Action Challenge](https://codalab.lisn.upsaclay.fr/competitions/4820#results).

## Install

Our code is mainly based on [SlowFast](https://github.com/facebookresearch/SlowFast), please refer to this repository for more information.

## Data Preparation

You need to download the following data to start the experiment:

1. Download the [H2O](https://github.com/taeinkwon/h2odataset) dataset
2. Extract flow frames using [denseflow](https://github.com/xumingze0308/denseflow)

## Train

You can train on the H2O dataset by running: 

```shell
REPO_PATH='/xxx/SlowFast'
export PYTHONPATH=$PYTHONPATH:$REPO_PATH
python tools/run_net.py --cfg configs/H2O/SLOWFAST_4x16_R50.yaml
```

## Test

You can test on the H2O dataset by running: 

```shell
REPO_PATH='/xxx/SlowFast'
export PYTHONPATH=$PYTHONPATH:$REPO_PATH
python tools/run_net.py --cfg configs/H2O/SLOWFAST_4x16_R50_TEST.yaml
```

## Reference 

The majority of this repository is borrowed from [SlowFast](https://github.com/facebookresearch/SlowFast). Thank these authors for their great work.