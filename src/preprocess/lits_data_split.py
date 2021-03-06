import argparse
import csv
import logging
import random
from pathlib import Path


def main(args):
    # Randomly split the data into k folds.
    patient_dirs = sorted(dir_ for dir_ in (args.data_dir / 'training').iterdir() if dir_.is_dir())
    random.seed(3)
    folds = random.sample(patient_dirs, k=len(patient_dirs))

    output_dir = args.output_dir
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    test_size = args.test_size
    test_folds = folds[-test_size:]
    ratio = (len(folds) - test_size) // args.k
    for i in range(args.k):
        valid_start, valid_end = i * ratio, (i + 1) * ratio
        valid_folds = folds[valid_start:valid_end]
        train_folds = tuple(set(folds) - (set(test_folds) | set(valid_folds)))
        csv_path = output_dir / f'{i}.csv'
        logging.info(f'Write the data split file to "{csv_path.resolve()}".')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'type'])
            for path in sorted(train_folds):
                path = args.resampled_data_dir.resolve() / 'training' / path.name
                writer.writerow([path, 'train'])
            for path in sorted(valid_folds):
                writer.writerow([path.resolve(), 'valid'])
            for path in sorted(test_folds):
                writer.writerow([path.resolve(), 'test'])
                
    # Write adaptation split file
    csv_path = output_dir / 'adaptation.csv'
    train_folds = tuple(set(folds) - (set(test_folds)))
    logging.info(f'Write the data split file to "{csv_path.resolve()}".')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'type'])
        for path in sorted(train_folds):
            path = args.resampled_data_dir.resolve() / 'training' / path.name
            writer.writerow([path, 'train'])
        for path in sorted(test_folds):
            writer.writerow([path.resolve(), 'test'])

    # Write testing data split file.
    patient_dirs = sorted(dir_ for dir_ in (args.data_dir / 'testing').iterdir() if dir_.is_dir())
    csv_path = output_dir / 'testing.csv'
    logging.info(f'Write the data split file to "{csv_path.resolve()}".')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'type'])
        for path in patient_dirs:
            writer.writerow([path, 'test'])


def _parse_args():
    parser = argparse.ArgumentParser(description="The LiTS data split script.")
    parser.add_argument('data_dir', type=Path, help='The directory of the data.')
    parser.add_argument('resampled_data_dir', type=Path, help='The directory of the resampled data.')
    parser.add_argument('output_dir', type=Path, help='The output directory of the data split files.')
    parser.add_argument('--k', type=int, choices=[3, 5], default=3,
                        help='The number of folds for cross-validation.')
    parser.add_argument('--test_size', type=int, default=41, help='The number of testing data.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(name)-4s | %(levelname)-4s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
