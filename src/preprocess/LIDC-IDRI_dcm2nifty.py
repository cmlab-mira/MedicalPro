import logging
import argparse
import nibabel as nib
from pathlib import Path

import dicom2nifti


def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    
    if output_dir.exists() is False:
        output_dir.mkdir(parents=True)
    
    patient_paths = sorted(list((data_dir).iterdir()))
    for path in patient_paths:
        folders = list(path.iterdir())
        for folder in folders:
            dcm_files = list(folder.glob('*/*.dcm'))

            if len(dcm_files) < 10:
                # X-ray series
                continue
            else:
                case_id = folder.parts[-2]
                output_path = output_dir / f"{case_id}.nii.gz" 
                try:
                    dicom2nifti.dicom_series_to_nifti(folder.as_posix(), output_path.as_posix(), reorient_nifti=True)
                except:
                    print(f"Failed: case {case_id}.")
                

def _parse_args():
    parser = argparse.ArgumentParser(description="Convert the data of LIDC-IDRI from dicom to nifti format.")
    parser.add_argument('data_dir', type=Path, help='The directory of the dataset.')
    parser.add_argument('output_dir', type=Path, help='The directory of the processed data.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
