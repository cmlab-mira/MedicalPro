import torch
import logging
import argparse
import nibabel as nib
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    if output_dir.exists() is False:
        output_dir.mkdir(parents=True)
    
    all_pred_paths = np.array([sorted(path for path in (pred_dir / 'prediction').iterdir())
                               for pred_dir in input_dir.glob('data_split_*') if pred_dir.is_dir()])
    
    for i in tqdm(range(all_pred_paths.shape[1])):
        filename = all_pred_paths[0, i].name
        metadata = nib.load(all_pred_paths[0, i].as_posix())
        pred = metadata.get_fdata()
        for j in range(1, all_pred_paths.shape[0]):
            pred += nib.load(all_pred_paths[j, i].as_posix()).get_fdata()
        _, pred = torch.softmax(torch.tensor(pred), dim=-1).max(-1)
        nib.save(nib.Nifti1Image(pred.numpy(), metadata.affine, metadata.header), (output_dir / filename).as_posix())


def _parse_args():
    parser = argparse.ArgumentParser(description="The lits ensemble script.")
    parser.add_argument('input_dir', type=Path, help='The directory of the root folder of all split predictions.')
    parser.add_argument('output_dir', type=Path, help='The output directory of the ensemble prediction.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(name)-4s | %(levelname)-4s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
