#!/usr/bin/env python3

"""
Fit.py
This script fits a model in the ISIC19 project.

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
parser.add_argument('--model_dir', default='experiments/model1', help="folder containing params.json")
parser.add_argument('--net_dir', default='networks_isic', help="folder containing artificial_neural_network.py")
parser.add_argument('--resume', default=False, help="resume training for more epochs")


def train_eval(dataloaders, dataset_sizes, net, criterion, optimizer, num_epochs):
    """
    Trains on the whole train set and evaluates a net on the val set.
    """
    # Initialize
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

    # Logs
    fname = os.path.join(args.model_dir, f'train.log')
    logging_train = myutils.setup_logger(fname)
    fname = os.path.join(args.model_dir,f'probs_val.log')
    logging_probs = myutils.setup_logger(fname)
    fname = os.path.join(args.model_dir, f'preds_val.log')
    logging_preds = myutils.setup_logger(fname)

    myutils.myseed(seed=42)  # Reproducibility
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load initial weights
    net = net.to(device)
    best_net_wts = copy.deepcopy(net.state_dict())
    best_acc, epoch = 0.0, 1

    # Set file paths to resume training for more epochs
    fname = f'last.tar'
    last_path = os.path.join(args.model_dir, fname)
    fname = f'best.tar'
    best_path = os.path.join(args.model_dir, fname)

    # To resume training for more epochs
    if args.resume:
        try:
            # Load best
            best_checkpoint = torch.load(best_path)
            best_net_wts = best_checkpoint['net_state_dict']
            best_acc = best_checkpoint['acc']

            # Load last
            last_checkpoint = torch.load(last_path)
            net.load_state_dict(last_checkpoint['net_state_dict'])
            optimizer.load_state_dict(last_checkpoint['optimizer_state_dict'])
            epoch = last_checkpoint['epoch'] + 1  # Since last epoch was saved we start with the next one
            logging_process.info(f'Model: {args.model_dir}\tLast epoch saved: {epoch-1}, resumming training since epoch: {epoch}')

        except FileNotFoundError as err:
            # This error happens when folds are present
            # If interrupted on fold 1 then best best_checkpoint for fold 2 does
            # not exists. This is fixed like this.
            logging_process.info(f'Model: {args.model_dir}\tError: {err}')

    best_val_loss = np.Inf
    epochs_no_improve = 0
    patience = 5 # Number of epochs not improve to stop
    # TRAINING LOOP
    for epoch in range(epoch, num_epochs+1):
        if epochs_no_improve == patience:
            print('Early stop')
            logging_process.info(f'Model: {args.model_dir}\tFold:{fold}\tEarly stop: {epoch}')
            break
        logging_train.info(f'Epoch {epoch}/{num_epochs}')
        print(f'Epoch {epoch}/{num_epochs}')

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                net.train()  # Set net to training mode
            else:
                net.eval()   # Set net to evaluate mode

            # Track statistics
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    probs, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'val':
                        logging_probs.info(probs.cpu().detach().numpy())
                        logging_preds.info(preds.cpu().detach().numpy())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)  # This is batch loss
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # epoch statistics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            logging_train.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Track the best loss
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            if phase == 'val':
              # Save last
              torch.save({
              'epoch': epoch,
              'net_state_dict': net.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': epoch_loss
              }, last_path)

            # deep copy the net
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_net_wts = net.state_dict()

                # Save best
                torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'acc': best_acc
                }, best_path)

                # Save best to json file
                best_metrics = {
                f'loss': epoch_loss,
                f'acc': best_acc.item()
                }
                with open (os.path.join(args.model_dir, f'metrics.json'), 'w') as f:
                    f.write(json.dumps(best_metrics))

    # load best net weights
    logging_process.info('Model: {}\tBest val Acc: {:4f}'.format(args.model_dir, best_acc))
    print('Best val Acc: {:4f}'.format(best_acc))
    net.load_state_dict(best_net_wts)


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

    # Fold loop
    dfs = {}

    net = myutils.get_network(args.net_dir, params.network)
    optimizer = myutils.get_optimizer(params.optimizer, net, params.learning_rate)

    fname = os.path.join(args.data_dir, 'train.csv')
    print(fname)
    train = pd.read_csv(fname)
    fname = os.path.join(args.data_dir, 'val.csv')
    val = pd.read_csv(fname)
    dfs['train'] = train
    dfs['val'] = val

    loaders = myutils.get_loaders(args.net_dir, 'loaders')
    dataloaders, dataset_sizes = loaders.get_loaders(dfs, size=params.size, batch_size=params.batch_size, num_workers=params.num_workers)
    weight = myutils.get_weight(train)
    criterion = myutils.get_loss_fn(args.net_dir, params.network, weight)
    logging_process.info(f'Model: {args.model_dir}\tFile: train.csv\tWeight: {weight}')

    # Train
    print('-'*10)
    logging_process.info(f'Model: {args.model_dir}, training has started for {params.num_epochs} epochs')
    train_eval(dataloaders, dataset_sizes, net, criterion, optimizer, num_epochs=params.num_epochs)
    logging_process.info(f'Model: {args.model_dir}, training has ended')
