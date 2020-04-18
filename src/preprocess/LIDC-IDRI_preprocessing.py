import logging
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk

import dicom2nifti


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if output_dir.exists() is False:
        output_dir.mkdir(parents=True)
        
    data_paths = sorted(input_dir.iterdir())
    for path in tqdm(data_paths):
        itk_img = sitk.ReadImage(path.as_posix())
        resampled_img = resample_to_isotropic(itk_img)
        
        filename = path.parts[-1]
        output_path = output_dir / filename
        sitk.WriteImage(resampled_img, output_path.as_posix())


def resample_to_isotropic(itk_img):
    w_res, h_res, d_res = itk_img.GetSpacing()[:]
    w, h, d = itk_img.GetSize()[:]
    
    resized_height = h * h_res // 1.0
    resized_width = w * w_res // 1.0
    resized_depth = d * d_res // 1.0
    target_shape = [resized_depth, resized_height, resized_width]
    new_spacing = [1.0, 1.0, 1.0]

    target_space = sitk.GetImageFromArray(np.ones(np.int32(list(target_shape) + [1]), dtype=np.float32), sitk.sitkFloat32)
    target_space.SetDirection(itk_img.GetDirection())
    target_space.SetSpacing(new_spacing)
    target_space.SetOrigin(itk_img.GetOrigin())

    affine = sitk.AffineTransform(3)
    affine.Scale((1.0, 1.0, 1.0))

    itk_img_resized = sitk.Resample(itk_img, target_space, affine.GetInverse())
    return itk_img_resized
    

def _parse_args():
    parser = argparse.ArgumentParser(description="The LIDC-IDRI data preprocessing.")
    parser.add_argument('input_dir', type=Path, help='The directory of the nifty files.')
    parser.add_argument('output_dir', type=Path, help='The directory of the processed data.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
