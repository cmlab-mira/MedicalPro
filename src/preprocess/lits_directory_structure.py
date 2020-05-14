import argparse
import logging
import shutil
from pathlib import Path


def main(args):
    data_dir = args.data_dir / 'training'
    logging.info(f'Structure the data in "{data_dir.resolve()}".')
    for ct_path, gt_path in zip(
        sorted(data_dir.glob('volume-*.nii')),
        sorted(data_dir.glob('segmentation-*.nii'))
    ):
        idx = ct_path.stem.replace('volume-', '')
        patient_dir = ct_path.parent / f'patient{idx}'
        if not patient_dir.is_dir():
            patient_dir.mkdir(parents=True)
        shutil.move(ct_path, patient_dir / ct_path.name)
        shutil.move(gt_path, patient_dir / gt_path.name)

    data_dir = args.data_dir / 'testing'
    logging.info(f'Structure the data in "{data_dir.resolve()}".')
    for ct_path in sorted(data_dir.glob('test-volume-*.nii')):
        idx = ct_path.stem.replace('test-volume-', '')
        patient_dir = ct_path.parent / f'patient{idx}'
        if not patient_dir.is_dir():
            patient_dir.mkdir(parents=True)
        shutil.move(ct_path, patient_dir / ct_path.name)


def _parse_args():
    parser = argparse.ArgumentParser(description="The LiTS directory structure script.")
    parser.add_argument('data_dir', type=Path, help='The directory of the data.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(name)-4s | %(levelname)-4s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
