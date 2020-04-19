import argparse
import h5py
import logging
import xmltodict
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    if output_dir.exists() is False:
        output_dir.mkdir(parents=True)

    data_paths = sorted(input_dir.iterdir())
    for path in tqdm(data_paths):
        filename = path.parts[-1].split('.')[0]
        mri_type = filename.split('_')[2]
        sid = filename.split('_')[-1]

        output_subdir = output_dir / mri_type
        if output_subdir.exists() is False:
            output_subdir.mkdir(parents=True)

        hf = h5py.File(path.as_posix())
        header = xmltodict.parse(hf['ismrmrd_header'][()])
        data = hf['reconstruction_rss'][()]
        vmax, vmin = data.max(), data.min()
        data = (data - vmin) / (vmax - vmin)

        itk_img = sitk.GetImageFromArray(data)
        resolution_x = (float(header['ismrmrdHeader']['encoding']['reconSpace']['fieldOfView_mm']['x']) /
                        float(header['ismrmrdHeader']['encoding']['reconSpace']['matrixSize']['x']))
        resolution_y = (float(header['ismrmrdHeader']['encoding']['reconSpace']['fieldOfView_mm']['y']) /
                        float(header['ismrmrdHeader']['encoding']['reconSpace']['matrixSize']['y']))
        resolution_z = (float(header['ismrmrdHeader']['encoding']['reconSpace']['fieldOfView_mm']['z']) /
                        float(header['ismrmrdHeader']['encoding']['reconSpace']['matrixSize']['z']))
        itk_img.SetSpacing([resolution_x, resolution_y, resolution_z])
        resampled_img = resample_to_isotropic(itk_img)

        size = np.array(resampled_img.GetSize())
        filename = path.parts[-1]
        sitk.WriteImage(resampled_img, (output_subdir / f'{sid}.nii.gz').as_posix())


def resample_to_isotropic(itk_img):
    w_res, h_res, d_res = itk_img.GetSpacing()[:]
    w, h, d = itk_img.GetSize()[:]

    resized_height = h * h_res // 1.0
    resized_width = w * w_res // 1.0
    resized_depth = d * d_res // 1.0
    target_shape = [resized_depth, resized_height, resized_width]
    new_spacing = [1.0, 1.0, 1.0]

    target_space = sitk.GetImageFromArray(
        np.ones(np.int32(list(target_shape) + [1]), dtype=np.float32),
        sitk.sitkFloat32
    )
    target_space.SetDirection(itk_img.GetDirection())
    target_space.SetSpacing(new_spacing)
    target_space.SetOrigin(itk_img.GetOrigin())

    affine = sitk.AffineTransform(3)
    affine.Scale((1.0, 1.0, 1.0))

    itk_img_resized = sitk.Resample(itk_img, target_space, affine.GetInverse())
    return itk_img_resized


def _parse_args():
    parser = argparse.ArgumentParser(description="The FastMRI brain data preprocessing.")
    parser.add_argument('input_dir', type=Path, help='The directory of the .h5 files.')
    parser.add_argument('output_dir', type=Path, help='The directory of the processed data.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
