import argparse
import csv
import logging
import random
from collections.abc import Iterable
from pathlib import Path


def main(args):
    # Randomly and evenly split the data into k folds based on the diseases.
    data_paths = sorted((args.data_dir / 'training').glob('**/Info.cfg'))
    group_dict = {
        'NOR': [],  # healthy patients
        'MINF': [],  # patients with previous myocardial infarction
        'DCM': [],  # patients with dilated cardiomyopathy
        'HCM': [],  # patients with an hypertrophic cardiomyopathy
        'RV': []  # patients with abnormal right ventricle
    }
    for data_path in data_paths:
        with open(data_path, 'r') as f:
            for line in f:
                key, value = line.rstrip('\n').split(': ')
                if key == 'Group':
                    group_dict[value].append(data_path.parent.resolve())
                    break
    _, groups = zip(*sorted(group_dict.items()))
    random.seed(0)
    groups = tuple(random.sample(group, k=len(group)) for group in groups)

    # Got twenty folds because there are twenty patients in each group,
    # and a fold contains five types of patients.
    folds = tuple(zip(*groups))

    output_dir = args.output_dir
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    ratio = len(folds) // (args.k + 1)
    test_start = args.k * ratio
    test_folds = folds[test_start:]
    for i in range(args.k):
        valid_start, valid_end = i * ratio, (i + 1) * ratio
        valid_folds = folds[valid_start:valid_end]
        train_folds = tuple(set(folds) - (set(test_folds) | set(valid_folds)))

        csv_path = output_dir / f'{i}.csv'
        logging.info(f'Write the data split file to "{csv_path.resolve()}".')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'type'])
            for path in sorted(_flatten(train_folds)):
                writer.writerow([path, 'train'])
            for path in sorted(_flatten(valid_folds)):
                writer.writerow([path, 'valid'])
            for path in sorted(_flatten(test_folds)):
                writer.writerow([path, 'test'])

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
    parser = argparse.ArgumentParser(description="The ACDC data split script.")
    parser.add_argument('data_dir', type=Path, help='The directory of the data.')
    parser.add_argument('output_dir', type=Path, help='The output directory of the data split files.')
    parser.add_argument('--k', type=int, choices=[5], default=5,
                        help='The number of folds for cross-validation.')
    args = parser.parse_args()
    return args


def _flatten(nested_iterable):
    for elem in nested_iterable:
        if not isinstance(elem, Iterable):
            yield elem
        else:
            yield from _flatten(elem)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(name)-4s | %(levelname)-4s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
