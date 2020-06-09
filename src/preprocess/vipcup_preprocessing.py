import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    sub_folder = [input_dir / 'training', input_dir / 'testing']
    target_spacing = [3.0, 3.0, 3.0]

    if output_dir.exists() is False:
        output_dir.mkdir(parents=True)

    for folder in sub_folder:
        folder_name = folder.parts[-1]
        data_paths = sorted(folder.iterdir())
        for path in tqdm(data_paths):
            patient_id = path.parts[-1]
            output_subdir = output_dir / folder_name / patient_id
            if output_subdir.exists() is False:
                output_subdir.mkdir(parents=True)

            if folder_name == 'training':
                img_path = path / f'{patient_id}_img.nii.gz'
                label_path = path / f'{patient_id}_label.nii.gz'
                img = sitk.ReadImage(img_path.as_posix())
                label = sitk.ReadImage(label_path.as_posix())
                if img.GetSpacing() != label.GetSpacing():
                    label.SetSpacing(img.GetSpacing())

                resampled_img = resample(img, target_spacing, sitk.sitkLinear)
                resampled_label = resample(label, target_spacing, sitk.sitkNearestNeighbor)
                sitk.WriteImage(resampled_img, (output_subdir / img_path.parts[-1]).as_posix())
                sitk.WriteImage(resampled_label, (output_subdir / label_path.parts[-1]).as_posix())

            elif folder_name == 'testing':
                img_path = path / f'{patient_id}_img.nii.gz'
                img = sitk.ReadImage(img_path.as_posix())
                resampled_img = resample(img, target_spacing, sitk.sitkLinear)
                sitk.WriteImage(resampled_img, (output_subdir / img_path.parts[-1]).as_posix())


def resample(itk_img, new_spacing, interpolator):
    h_res, w_res, d_res = itk_img.GetSpacing()[:]
    h, w, d = itk_img.GetSize()[:]

    resized_height = h * h_res // new_spacing[0]
    resized_width = w * w_res // new_spacing[1]
    resized_depth = d * d_res // new_spacing[2]
    target_shape = [resized_depth, resized_height, resized_width]

    target_space = sitk.GetImageFromArray(
        np.ones(np.int32(list(target_shape) + [1]), dtype=np.float32),
        sitk.sitkFloat32
    )
    target_space.SetDirection(itk_img.GetDirection())
    target_space.SetSpacing(new_spacing)
    target_space.SetOrigin(itk_img.GetOrigin())

    affine = sitk.AffineTransform(3)
    affine.Scale((1.0, 1.0, 1.0))

    itk_img_resized = sitk.Resample(itk_img, target_space, affine.GetInverse(), interpolator)
    return itk_img_resized


def _parse_args():
    parser = argparse.ArgumentParser(description="The vipcup data preprocessing.")
    parser.add_argument('input_dir', type=Path, help='The directory of the data.')
    parser.add_argument('output_dir', type=Path, help='The directory of the processed data.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
