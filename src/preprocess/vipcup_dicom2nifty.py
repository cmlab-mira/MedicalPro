import logging
import argparse
from pathlib import Path

import dicom2nifti


def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir

    if output_dir.exists() is False:
        output_dir.mkdir(parents=True)

    for sub_dir in data_dir.iterdir():
        dir_name = sub_dir.name
        (output_dir / 'training').mkdir(exist_ok=True)
        (output_dir / 'testing').mkdir(exist_ok=True)
        
        patient_paths = sorted(dir_ for dir_ in sub_dir.iterdir() if dir_.is_dir())
        for path in patient_paths:
            folders = [folder for folder in path.iterdir() if folder.is_dir()]
            for folder in folders:
                dcm_files = list(folder.glob('*/*.dcm'))

                if len(dcm_files) == 1:
                    # contour dicom
                    continue
                else:
                    case_id = folder.parts[-2]
                    if folder.parts[-3] == 'VIP_CUP18_TestData':
                        output_path = output_dir / 'testing' / f"{case_id}.nii.gz"
                    else:
                        output_path = output_dir / 'training' / f"{case_id}.nii.gz"
                    try:
                        dicom2nifti.dicom_series_to_nifti(folder.as_posix(),
                                                          output_path.as_posix(),
                                                          reorient_nifti=True)
                    except Exception:
                        print(f"Failed: case {case_id}.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Convert the data of VIPCUP from dicom to nifti format.")
    parser.add_argument('data_dir', type=Path, help='The directory of the dataset.')
    parser.add_argument('output_dir', type=Path, help='The directory of the processed data.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
