import argparse
import csv
import logging
import random
import numpy as np
import nibabel as nib
from pathlib import Path


def main(args):
    # Randomly split the data into k folds.
    patient_dirs = sorted(dir_ for dir_ in (args.resampled_data_dir / 'training').iterdir() if dir_.is_dir())

    # Testing fold, which is the official validation fold
    test_size = args.test_size
    test_folds = patient_dirs[:test_size]
    folds = patient_dirs[test_size:]

    np.random.seed(0)
    output_dir = args.output_dir
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # Calculate the total tumor size of each patient
    patient_info = {}
    for dir_ in folds:
        patient_id = dir_.name
        label_path = dir_ / f'{patient_id}_label.nii.gz'
        metadata = nib.load(label_path)
        pixel_size = np.prod(metadata.header['pixdim'][1:4])
        label = metadata.get_fdata()
        tumor_size = np.sum(label[label > 0]) * pixel_size
        patient_info[patient_id] = tumor_size

    # Randomly shuffle the splited patient group
    sorted_patient_list = sorted(patient_info.keys(), key=patient_info.__getitem__)
    sorted_patient_list = sorted_patient_list[:-(len(sorted_patient_list) % args.k)]
    splited_patient_list = np.array(sorted_patient_list).reshape(-1, args.k)
    np.take(splited_patient_list, np.random.permutation(splited_patient_list.shape[1]), axis=1, out=splited_patient_list)
    
    for i in range(args.k):
        valid_patient_list = splited_patient_list[:, i].reshape(-1)
        valid_folds = [args.resampled_data_dir / 'training' / pid for pid in valid_patient_list]
        train_folds = tuple(set(folds) - (set(test_folds) | set(valid_folds)))
        csv_path = output_dir / f'{i}.csv'
        logging.info(f'Write the data split file to "{csv_path.resolve()}".')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'type'])
            for path in sorted(train_folds):
                writer.writerow([path.resolve(), 'train'])
            for path in sorted(valid_folds):
                writer.writerow([path.resolve(), 'valid'])
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
    parser.add_argument('--test_size', type=int, default=35, help='The number of testing data.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(name)-4s | %(levelname)-4s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
