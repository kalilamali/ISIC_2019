#!/usr/bin/env python3

"""
Myutils.py
This script contains the auxiliary functions used in the ISIC 2019 project.

Author      K.Loaiza
Comments    Created: Thursday, April 16, 2020
"""

import os
import json
import torch
import random
import logging
import numpy as np
import importlib.util
import torch.optim as optim

from torch import nn
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def myseed(seed=42):
    """
    Function that takes a seed number to make results reproducible.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_num_parameters(net):
    """
    Function that takes a neural network and returns the total number of parameters
    and the total number of trainable parameters.
    """
    myseed(seed=42)
    total_params = sum(p.numel() for p in net.parameters())
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params, total_trainable_params


def get_weight(train):
  """
  Function that takes a dataframe of training data and returns a tensor of
  weights according to the number of classes
  """
  myseed(seed=42)
  counts = train.label_code.value_counts().to_list()
  summed = sum(counts)
  weight = [x / summed for x in counts]
  weight = torch.tensor(weight)
  weight = weight.to(device)
  return(weight)


def setup_logger(fname, level=logging.INFO, date=False):
    """
    Function that takes the filename of a log and returns a logger.
    Function that allows to setup as many loggers as required.
    """
    name = os.path.splitext(fname)[-2]
    handler = logging.FileHandler(fname)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    if date:
        handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def get_module(dir_path, script_name):
    """
    Function that takes a path to a dir that contains a script .py
    and returns the contents to be used as python modules.
    NOTE:
    **This function is only used with loaders because using it with neural networks
    or loss functions would imply calling their name explicitaly which is something
    we do not want to do in this project.***
    """
    fname = os.path.join(dir_dir, f'{script_name}.py')
    spec = importlib.util.spec_from_file_location(net_dir, fname)
    mymodule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mymodule)
    return mymodule


def get_network(net_dir, net_name):
    """
    Function that takes a neural network path and name
    and returns the neural network object from a file inside a folder.
    """
    # Reproducibility
    myseed(seed=42)
    fname = os.path.join(net_dir, f'{net_name}.py')
    spec = importlib.util.spec_from_file_location(net_dir, fname)
    mymodule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mymodule)
    net = getattr(mymodule, net_name)()
    return net


def get_loss_fn(net_dir, net_name, weight):
    """
    Function that takes a neural network path and name
    and returns the loss fn object from a file inside a folder.
    """
    # Reproducibility
    myseed(seed=42)
    fname = os.path.join(net_dir, f'{net_name}.py')
    spec = importlib.util.spec_from_file_location(net_dir, fname)
    mymodule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mymodule)
    #net = getattr(mymodule, net_name)()
    return mymodule.loss_fn(weight=weight)


def get_optimizer(optimizer_name, net, lr):
    """
    Function that takes an optimizer name with some params
    and returns the optimizer object.
    """
    # Reproducibility
    myseed(seed=42)
    if optimizer_name == "SGD":
        optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr)
    if optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(net.parameters(), lr)
    return optimizer


def average_metrics(folds, model_dir):
    """
    Function that takes the number of folds and a model path to
    merge several metric files in .json format.
    """
    metrics = {}
    for fold in range(1, folds+1):
        jsonfile = os.path.join(model_dir, f'metrics{fold}.json')
        with open(jsonfile) as f:
            d = json.load(f)
            metrics.update(d)

    total_loss = 0
    total_acc = 0
    for key in metrics:
        if key[:4] == 'loss':
            total_loss += metrics[key]
        if key[:3] == 'acc':
            total_acc += metrics[key]

    metrics['losscv'] = total_loss/fold
    metrics['acccv'] = total_acc/fold

    # Save metrics
    m = json.dumps(metrics)
    f = open(model_dir + '/metricscv' +'.json', 'w')
    f.write(m)
    f.close()


def run_fast_scandir(dir, ext):    # dir: str, ext: list
    """
    Function that takes a path and an extension
    and returns a list of file paths with that extension.
    """
    files = []
    for f in os.scandir(dir):
        if f.is_dir():
            subfolder = f.path
            subfiles = run_fast_scandir(subfolder, ext)
            files.extend(subfiles)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)
    return files
