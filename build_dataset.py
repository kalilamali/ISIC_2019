#!/usr/bin/env python3

"""
Build_dataset.py
This script builds the dataset in the ISIC 2019 project.

Author      K.Loaiza
Comments    Created: Thursday, May 6, 2020
"""

import os
import json
import sys
import argparse
import myutils
import pandas as pd

from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/binary', help="folder containing the dataset")
parser.add_argument('--folds', default=3, help='cross validation folds', type=int)


def load_data(data_dir):
    """
    Function that takes a folder, finds all .jpg files inside the folder,
    and creates a dataframe.
    """
    # Reproducibility
    myutils.myseed(seed=42)

    # Get the image paths
    filenames = myutils.run_fast_scandir(data_dir, [".jpg"])
    df1 = pd.DataFrame(data=filenames, columns=['image_path'])
    df1['image_id'] = df1['image_path'].apply(lambda x:os.path.splitext(os.path.basename(x))[0])
    df1 = df1.set_index('image_id')

    # Get the labels
    fname = os.path.join(data_dir, 'labels.csv')
    df2 = pd.read_csv(fname)
    df2 = df2.set_index('image')
    # Do not move function from here
    def get_disease(row):
        for c in df2.columns:
            if row[c] == 1:
                return c
    df2 = df2.apply(get_disease, axis=1).to_frame(name='label')
    df = pd.merge(df1, df2, left_index=True, right_index=True)
    df['label'] = df['label'].astype('category')
    mapping = dict(enumerate(df['label'].cat.categories ))
    df['label_code'] = pd.Categorical(df['label']).codes

    # Save the data as a .csv file
    df.to_csv(f'{data_dir}.csv', index=False)
    logging_data_process.info(f'Saved: {data_dir}.csv')

    # Save the mapping as a .json file
    with open(f'{data_dir}.json', 'w') as f:
        f.write(json.dumps(mapping))
        logging_data_process.info(f'Saved: {data_dir}.json')


def data_split(data_dir, folds):
    """
    Function that takes a data_dir and a number of folds,
    and splits images in data_dir into
    training(80%) and testing(20%) data.

    For fit.py training data is further splitted into
    training and validation sets.

    If cross validation is needed, training data is also splitted into
    train and validation folds.
    """
    # Reproducibility
    myutils.myseed(seed=42)
    seed = 42

    # Load the data with image paths and labels
    df = pd.read_csv(f'{data_dir}.csv')
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Test
    train_val, test = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    train_val, test = train_val.reset_index(drop=True), test.reset_index(drop=True)
    test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    logging_data_process.info(f"Saved: {os.path.join(data_dir, 'test.csv')}")

    # Train and validation
    train, val = train_test_split(train_val, test_size=0.2, random_state=seed, shuffle=True)
    train, val = train_val.reset_index(drop=True), val.reset_index(drop=True)
    train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
    logging_data_process.info(f"Saved: {os.path.join(data_dir, 'train.csv')}")
    logging_data_process.info(f"Saved: {os.path.join(data_dir, 'val.csv')}")

    # Cross validation folds
    if folds > 1:
        logging_data_process.info(f'Folds: {folds}')
        X = train_val[['image_path']]
        y = train_val[['label_code']]
        skf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
        fold = 0
        for train_idx, val_idx in skf.split(X, y):
            fold += 1
            train_idx, val_idx = list(train_idx), list(val_idx)
            train, val = train_val.iloc[train_idx,:], train_val.iloc[val_idx,:]
            train, val = train.reset_index(drop=True), val.reset_index(drop=True)
            train.to_csv(os.path.join(data_dir, f'train{fold}.csv'), index=False)
            val.to_csv(os.path.join(data_dir, f'val{fold}.csv'), index=False)
            logging_data_process.info(f"Saved: {os.path.join(data_dir, f'train{fold}.csv')}")
            logging_data_process.info(f"Saved: {os.path.join(data_dir, f'val{fold}.csv')}")


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Could not find the dataset at {}".format(args.data_dir)

    # Initialize main log folder
    logs_dir_path = os.path.join(os.getcwd(),'Logs')
    if not os.path.exists(logs_dir_path):
        os.mkdir(logs_dir_path)

    # Initialize main log file
    log_file = os.path.join(logs_dir_path, 'data.log')
    logging_process = myutils.setup_logger(log_file, date=True)

    # Save commandline settings to log
    script_activated = ' '.join(sys.argv)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging_process.info(f'Script: {script_activated}, device: {device}')

    # Build dataset
    logging_data_process.info('Script: load_data')
    load_data(args.data_dir)
    logging_data_process.info('Script: data_split')
    data_split(args.data_dir, args.folds)

    # DONE
    print(f'Done building dataset, saved to {args.data_dir}')
