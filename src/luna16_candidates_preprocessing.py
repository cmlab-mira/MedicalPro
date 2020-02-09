import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk


def main(args):
    logging.info(f"Read the csv file from {args.candidates_path}.")
    df = pd.read_csv(args.candidates_path)
    positive_df = df[df['class'] == 1]
    negative_df = df[df['class'] == 0]
    
    for i in range(10):
        subset_path = args.data_dir / f'subset{i}'
        positive_dir = args.output_dir / 'positive' / f'subset{i}'
        negative_dir = args.output_dir / 'negative' / f'subset{i}'
        
        if positive_dir.exists() == False:
            positive_dir.mkdir(parents=True)
        if negative_dir.exists() == False:
            negative_dir.mkdir(parents=True)

        data_paths = list(subset_path.glob('./*.mhd'))
        for path in data_paths:
            # Load the image and resample it to be istropic
            itk_img = sitk.ReadImage(str(path))
            resampled_img = resample_to_istropic(itk_img)
            img_array = sitk.GetArrayFromImage(resampled_img)
            img_array = img_array.transpose(2, 1, 0)
            
            # Execute the preprocessing according to the ModelGenesis guildlines
            # - all the intensity values be clipped on the min (-1000) and max (+1000)
            img_array = img_array.clip(-1000, 1000)
            # - scale between 0 and 1
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            
            origin = np.array(list(resampled_img.GetOrigin()))
            spacing = np.array(list(resampled_img.GetSpacing()))
            
            sid = path.name.replace('.mhd', '')
            patient_df = df[df.seriesuid == sid]
            """
            _positive_df = positive_df[positive_df.seriesuid == sid]
            _negative_df = negative_df[negative_df.seriesuid == sid].sample(n=int(_positive_df.size / args.positive_ratio))
            """
            patient_df = tqdm(patient_df[patient_df.seriesuid == sid].iterrows(), desc=f'subset{i}_{sid}')
            for index, row in patient_df:
                word_coord = np.array(row.values[1:4])
                voxel_coord = world_to_voxel_coord(word_coord, origin, spacing)
                candidate = crop(img_array, voxel_coord, args.crop_size)
                if (candidate.shape != np.array(args.crop_size)).any():
                    continue
                
                if row['class'] == 1:
                    output_path = positive_dir / f'{sid}_{index}.npy'
                else:
                    output_path = negative_dir / f'{sid}_{index}.npy'
                with open(output_path, 'wb') as fout:
                    np.save(fout, candidate)

def resample_to_istropic(itk_img):
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
   
def world_to_voxel_coord(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = (stretched_voxel_coord / spacing)
    voxel_coord = [np.round(coord).astype(np.int) for coord in voxel_coord]
    return voxel_coord
    
def crop(img, coord, crop_size):
    x_start, x_end = coord[0]-crop_size[0]//2, coord[0]+crop_size[0]//2
    y_start, y_end = coord[1]-crop_size[1]//2, coord[1]+crop_size[1]//2
    z_start, z_end = coord[2]-crop_size[2]//2, coord[2]+crop_size[2]//2
    cropped_img = img[x_start:x_end, y_start:y_end, z_start:z_end]
    return cropped_img
    
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser(description="The main pipeline script.")
    parser.add_argument('-i', '--candidates_path', type=Path, help='The path of the official candidates file.')
    parser.add_argument('-d', '--data_dir', type=Path, help='The path of the LUNA16 dataset.')
    parser.add_argument('-o', '--output_dir', type=Path, help='The path of output preprocessed patches.')
    parser.add_argument('-pr', '--postive_ratio', type=float, help='The postive samples ratio out of the whole dataset.')
    parser.add_argument('--crop_size', type=int, nargs='+', help='The crop size of each dim.')
    args = parser.parse_args()
    main(args)