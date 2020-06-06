import argparse
import csv
import logging
import random
from pathlib import Path


def main(args):
    # Randomly and evenly split the data into 5 folds based on the diseases.
    HGG_patient_dirs = sorted(dir_ for dir_ in (args.resampled_data_dir / 'HGG').iterdir() if dir_.is_dir())
    LGG_patient_dirs = sorted(dir_ for dir_ in (args.resampled_data_dir / 'LGG').iterdir() if dir_.is_dir())
    random.seed(0)
    HGG_patient_dirs = tuple(random.sample(HGG_patient_dirs, k=len(HGG_patient_dirs)))
    LGG_patient_dirs = tuple(random.sample(LGG_patient_dirs, k=len(LGG_patient_dirs)))
    HGG_unit = len(HGG_patient_dirs) // 5
    LGG_unit = len(LGG_patient_dirs) // 5

    output_dir = args.output_dir
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    folds = HGG_patient_dirs + LGG_patient_dirs
    test_folds = HGG_patient_dirs[-HGG_unit:] + LGG_patient_dirs[-LGG_unit:]
    for i in range(args.k):
        HGG_start, HGG_end = i * HGG_unit, (i + 1) * HGG_unit
        LGG_start, LGG_end = i * LGG_unit, (i + 1) * LGG_unit
        valid_folds = HGG_patient_dirs[HGG_start:HGG_end] + LGG_patient_dirs[LGG_start:LGG_end]
        train_folds = tuple(set(folds) - (set(test_folds) | set(valid_folds)))

        csv_path = output_dir / f'{i}.csv'
        logging.info(f'Write the data split file to "{csv_path.resolve()}".')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'type'])
            for path in sorted(train_folds):
                writer.writerow([path, 'train'])
            for path in sorted(valid_folds):
                writer.writerow([path, 'valid'])
            for path in sorted(test_folds):
                writer.writerow([path, 'test'])


def _parse_args():
    parser = argparse.ArgumentParser(description="The BraTS17 data split script.")
    parser.add_argument('resampled_data_dir', type=Path, help='The directory of the resampled data.')
    parser.add_argument('output_dir', type=Path, help='The output directory of the data split files.')
    parser.add_argument('--k', type=int, choices=[3], default=3,
                        help='The number of folds for cross-validation.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(name)-4s | %(levelname)-4s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
