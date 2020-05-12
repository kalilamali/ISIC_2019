# ISIC 2019

Skin lesion classifier for the ISIC 2019 challenge using Python and PyTorch.

## Getting Started

To start dowload the data from https://challenge2019.isic-archive.com

And place it into the following format:
```
isic19/
  images/
      image1.jpg
      image2.jpg
      ...
  labels.csv
```

## Prerequisites

Install conda according to instructions on https://docs.conda.io/en/latest/

See requirements.txt for details.

```
conda create -n your_environment_name
conda activate your_environment_name
./requirements.txt
```

## How to use the scripts

Use the scripts in this order.

```
build_dataset.py
train.py
fit.py
evaluate.py
```
