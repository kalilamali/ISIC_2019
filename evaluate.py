#!/usr/bin/env python3

"""
Evaluate.py
This script evaluates a model in the ISIC19 project.

Author      K.Loaiza
Comments    Created: Thursday, May 6, 2020
"""

import os
import sys
import copy
import json
import myutils
import argparse
import torch
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/isic19', help="folder containing the dataset")
parser.add_argument('--file', default='test', help=".csv filename that will be evalutated")
parser.add_argument('--model_dir', default='experiments/model1', help="folder containing params.json")
parser.add_argument('--net_dir', default='networks_isic', help="folder containing artificial_neural_network.py")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir containing weights to load")


def eval(file, dataloaders, dataset_sizes, net, criterion):
    """
    Evaluate a net.
    """
    # Initialize
    # Logs
    fname = os.path.join(args.model_dir, f'{file}.log')
    logging_eval = myutils.setup_logger(fname)

    myutils.myseed(seed=42)  # Reproducibility
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load network
    net = net.to(device)

    # Reload weights from the saved file
    fname = f'{args.restore_file}.tar'
    restore_path = os.path.join(args.model_dir, fname)

    # Load
    checkpoint = torch.load(restore_path)
    net.load_state_dict(checkpoint['net_state_dict'])

    phase = 'val'

    predictions = []
    probabilities = []
    all_probabilities = []

    # Track statistics
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)
        probs, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        all_probabilities.extend(outputs.cpu().detach().numpy())
        probabilities.extend(probs.cpu().detach().numpy())
        predictions.extend(preds.cpu().detach().numpy())

        # statistics
        running_loss += loss.item() * inputs.size(0)  # This is batch loss
        running_corrects += torch.sum(preds == labels.data)

        # batch statistics
        batch_loss = running_loss / dataset_sizes[phase]
        batch_acc = running_corrects.double() / dataset_sizes[phase]
        logging_eval.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, batch_loss, batch_acc))
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, batch_loss, batch_acc))

    return probabilities, predictions, all_probabilities


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Could not find the dataset at {}".format(args.data_dir)
    assert os.path.isdir(args.model_dir), "Could not find the model at {}".format(args.model_dir)
    assert os.path.isdir(args.net_dir), "Could not find the network at {}".format(args.net_dir)

    # Set main log
    logs_dir_path = os.path.join(os.getcwd(),'Logs')
    if not os.path.exists(logs_dir_path):
        os.mkdir(logs_dir_path)

    log_file = os.path.join(logs_dir_path, 'process.log')
    logging_process = myutils.setup_logger(log_file, date=True)

    script_activated = ' '.join(sys.argv)  # command line input
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging_process.info(f'Script: {script_activated}, device: {device}')


    # Get the experiment parameters
    params_file = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(params_file), "No json configuration file found at {}".format(params_file)
    params = myutils.Params(params_file)

    dfs = {}
    net = myutils.get_network(args.net_dir, params.network)

    fname = os.path.join(args.data_dir, f'{args.file}.csv')
    frame = pd.read_csv(fname)
    dfs['val'] = frame
    loaders = myutils.get_loaders(args.net_dir, 'loaders')
    dataloaders, dataset_sizes = loaders.get_loaders(dfs, size=params.size, batch_size=params.batch_size, num_workers=params.num_workers)
    weight = None
    criterion = myutils.get_loss_fn(args.net_dir, params.network, weight)
    logging_process.info(f'Model: {args.model_dir}\tFile: {args.file}.csv\tWeight: {weight}')
    print('-'*10)
    num_steps = len(frame)/params.batch_size
    logging_process.info(f'Model: {args.model_dir}, evaluation has started for {num_steps} steps')
    probabilities, predictions, all_probabilities = eval(args.file, dataloaders, dataset_sizes, net, criterion)

    logging_process.info(f'Model: {args.model_dir}, evaluation has ended')

    fname = os.path.join(args.data_dir, f'{args.file}.csv')
    df = pd.read_csv(fname)
    df['probabilities'] = probabilities
    df['predictions'] = predictions
    df['all_probabilities'] = all_probabilities

    results_dir = os.path.join(args.model_dir, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    fname = fname = os.path.join(results_dir, f'{args.file}.csv')
    df.to_csv(fname)
