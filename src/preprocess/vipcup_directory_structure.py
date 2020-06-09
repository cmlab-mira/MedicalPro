import shutil
import logging
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path


def main(args):
    training_dir = args.data_dir / 'training/'
    testing_dir = args.data_dir / 'testing/'
    label_dir = args.label_dir

    # Move the label nifty file to the data directory
    paths = list(training_dir.glob('*.nii.gz'))
    for path in paths:
        name = path.name.replace('.nii.gz', 'label_GTV1.nii.gz')
        label_path = label_dir / name
        output_path = path.parent / name.replace('label_GTV1', '_label')
        shutil.copy(label_path.as_posix(), output_path.as_posix())

    # Rotate the ground truth label to align with image orientation
    paths = sorted(training_dir.glob('*label*'))
    for path in paths:
        name = path.name.replace('_label', "")
        img = nib.load(training_dir / name)
        label = nib.load(path)
        new_label = nib.Nifti1Image(np.rot90(label.get_fdata(), 3), affine=img.affine, header=img.header)
        nib.save(new_label, path)

    # Organize the directory structure
    # - training folder
    paths = sorted(training_dir.glob('*_label*'))
    for path in paths:
        pid = path.parts[-1].split('_')[0]
        patient_dir = path.parent / pid
        if patient_dir.exists() is False:
            patient_dir.mkdir()
        shutil.move((path.parent / f'{pid}.nii.gz').as_posix(),
                    (patient_dir / f'{pid}_img.nii.gz').as_posix())
        shutil.move(path.as_posix(), (patient_dir / pid).as_posix())
    # - testing folder
    paths = sorted(testing_dir.glob('*.nii.gz'))
    for path in paths:
        pid = path.parts[-1].split('.')[0]
        patient_dir = path.parent / pid
        if patient_dir.exists() is False:
            patient_dir.mkdir()
        shutil.move((path.parent / f'{pid}.nii.gz').as_posix(),
                    (patient_dir / f'{pid}_img.nii.gz').as_posix())


def _parse_args():
    parser = argparse.ArgumentParser(description="Process and organize the data directory.")
    parser.add_argument('data_dir', type=Path, help='The directory containing the data nifty files.')
    parser.add_argument('label_dir', type=Path, help='The directory containing the label nifty files.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
