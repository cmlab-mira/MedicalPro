import argparse
import csv
import logging
import random
import pandas as pd
import nibabel as nib
from pathlib import Path


def main(args):
    random.seed('FastMRI')
    data_dir = args.data_dir
    
    output_dir = args.output_dir
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)    
    split_csv_path = output_dir / 'FastMRI.csv'

    train_paths, valid_paths = [], []
    subdirs = data_dir.iterdir()
    for subdir in subdirs:
        data_paths = sorted(subdir.glob('*.nii.gz'))
        removed_paths = []
        for path in data_paths:
            h, w, d = nib.load(path.as_posix()).shape
            if (h < 64) or (w < 64) or (d < 32):
                removed_paths.append(path)

        data_paths = sorted(set(data_paths) - set(removed_paths))
        train_paths += sorted(random.sample(data_paths, int(len(data_paths) * 0.9)))
        valid_paths += sorted(set(data_paths) - set(train_paths))

    train_df = pd.DataFrame({'path': train_paths, 'type': ['train'] * len(train_paths)})
    valid_df = pd.DataFrame({'path': valid_paths, 'type': ['valid'] * len(valid_paths)})
    df = pd.concat([train_df, valid_df])
    df.to_csv(split_csv_path, index=False)


def _parse_args():
    parser = argparse.ArgumentParser(description="The FastMRI brain data split script.")
    parser.add_argument('data_dir', type=Path, help='The directory of the data.')
    parser.add_argument('output_dir', type=Path, help='The output directory of the data split files.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(name)-4s | %(levelname)-4s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
