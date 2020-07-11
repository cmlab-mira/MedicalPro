import argparse
import logging
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.data.transforms import Resample


def main(args):
    input_dir = args.input_dir
    patient_dirs = (
        sorted(dir_ for dir_ in (input_dir / 'HGG').iterdir() if dir_.is_dir())
        + sorted(dir_ for dir_ in (input_dir / 'LGG').iterdir() if dir_.is_dir())
    )
    output_dir = args.output_dir
    if output_dir.exists() is False:
        output_dir.mkdir(parents=True)
    resample_fn = Resample(output_spacing=[2., 2., 2.])

    for patient_dir in tqdm(patient_dirs, desc='resampling', ascii=True):
        for path in patient_dir.glob('*.nii.gz'):
            nii_img = nib.load(path.as_posix())
            img = nii_img.get_fdata().astype(
                np.int64 if 'seg' in path.as_posix() else np.float32
            )[..., np.newaxis]
            input_spacing = nii_img.header['pixdim'][1:4]
            resampled_img, = resample_fn(img,
                                         input_spacings=(input_spacing,),
                                         orders=(0,) if 'seg' in path.as_posix() else (1,))
            resampled_img = resampled_img.squeeze(axis=-1)
            affine = nii_img.affine  # The affine doesn't modified.
            header = nii_img.header
            header['pixdim'][1:4] = [2., 2., 2.]
            output_path = output_dir / path.relative_to(input_dir)
            if output_path.parent.exists() is False:
                output_path.parent.mkdir(parents=True)
            nib.save(
                nib.Nifti1Image(resampled_img, affine, header),
                output_path.as_posix()
            )


def _parse_args():
    parser = argparse.ArgumentParser(description="The BraTS17 data preprocessing.")
    parser.add_argument('input_dir', type=Path, help='The directory of the data.')
    parser.add_argument('output_dir', type=Path, help='The directory of the processed data.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
