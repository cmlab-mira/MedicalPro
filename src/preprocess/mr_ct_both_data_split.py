import argparse
import csv
import logging
import pandas as pd
from pathlib import Path


def main(args):
    mr_df = pd.read_csv(args.mr_csv.as_posix())
    ct_df = pd.read_csv(args.ct_csv.as_posix())
    
    output_dir = args.output_dir
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)    
    split_csv_path = output_dir / 'pretrain.csv'

    df = pd.concat([ct_df, mr_df])
    df.to_csv(split_csv_path, index=False)


def _parse_args():
    parser = argparse.ArgumentParser(description="Merge the data split files of both mr and ct.")
    parser.add_argument('mr_csv', type=Path, help='The path of the mr data split file.')
    parser.add_argument('ct_csv', type=Path, help='The path of the ct data split file.')
    parser.add_argument('output_dir', type=Path, help='The output directory of the data split files.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(name)-4s | %(levelname)-4s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
