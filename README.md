# Rethinking Pre-training in Medical Imaging

# Environment
To create the environment, please install `anaconda`/`miniconda` and run the following command

```
conda env create -f env.yml
```

# Datasets
We select four public tasks as our experimental challenges, namely ACDC, Lits, BraTS'17, VIPCUP. We do the pre-processing on each dataset separetely. Please refer the following step to pre-process the corresponding dataset.

## ACDC
1. Download ACDC dataset from [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html) and save it at `ACDC_DIR`.
2. Generate the data split files for the cross validation
```
python src/preprocess/acdc_data_split.py {ACDC_DIR} {ACDC_DIR}/data_split
```

## LiTS
1. Download LiTS dataset from [here](https://competitions.codalab.org/competitions/17094) and save it at `LITS_DIR`.
2. Change the data directory structure
```
python src/preprocess/lits_directory_structure.py {LITS_DIR}
```
3. Pre-process and save at `LITS_PROCESSED_DIR`
```
python src/preprocess/lits_preprocessing.py.py {LITS_DIR} {LITS_PROCESSED_DIR}
```
4. Generate the data split files for the cross validation
```
python src/preprocess/lits_data_split.py {LITS_PROCESSED_DIR} {LITS_PROCESSED_DIR}/data_split
```

## BraTS
1. Download BraTS'17 dataset from [here](https://www.med.upenn.edu/sbia/brats2017/data.html) and save it at `BRATS_DIR`.
2. Pre-process and save at `BRATS_PROCESSED_DIR`
```
python src/preprocess/brats17_preprocessing.py {BRATS_DIR} {BRATS_PROCESSED_DIR}
```
3. Generate the data split files for the cross validation
```
python src/preprocess/brats17_data_split.py {BRATS_PROCESSED_DIR} {BRATS_PROCESSED_DIR}/data_split
```

## VIPCUP
1. Download VIPCUP dataset and save it at `VIPCUP_DIR`.
2. Generate `.nii.gz` files from raw dicom files
```
python src/preprocess/vipcup_dicom2nifty.py {VIPCUP_DIR} {VIPCUP_NII_DIR}
```
3. Pre-process and save at `VIPCUP_PROCESSED_DIR`
```
python src/preprocess/vipcup_preprocessing.py {VIPCUP_DIR} {VIPCUP_PROCESSED_DIR}
```
4. Generate the data split files for the cross validation
```
python src/preprocess/vipcup_preprocessing.py {VIPCUP_PROCESSED_DIR} {VIPCUP_PROCESSED_DIR}/data_split
```

# Model training
We provide training and testing configurations for baselines and our proposed network.
Please note that the paths in configurations should be modified.

## Train
To reproduce the proposed `Network Alchemy` algorithm, please follow these commands. If you want to apply the method on another task, you can refer to the experimental configs.

### BraTS'17
```
python -m src.main configs/train/brats17_seg/network_alchemy_pre_trained_ct_fine_tuned/identification/data_split_{0/1/2}_config.yaml
python -m src.main configs/train/brats17_seg/network_alchemy_pre_trained_ct_fine_tuned/modification/data_split_{0/1/2}_config.yaml
python -m src.main configs/train/brats17_seg/network_alchemy_pre_trained_ct_fine_tuned/maximization/data_split_{0/1/2}_config.yaml
```

### VIPCUP
```
python -m src.main configs/train/vipcup_seg/network_alchemy_pre_trained_mr_fine_tuned/identification/data_split_{0/1/2}_config.yaml
python -m src.main configs/train/vipcup_seg/network_alchemy_pre_trained_mr_fine_tuned/modification/data_split_{0/1/2}_config.yaml
python -m src.main configs/train/vipcup_seg/network_alchemy_pre_trained_mr_fine_tuned/maximization/data_split_{0/1/2}_config.yaml
```

## Test
### BraTS'17
```
python -m src.main configs/test/brats17_seg/network_alchemy_pre_trained_ct_fine_tuned/data_split_{0/1/2}_config.yaml
```

### VIPCUP
```
python -m src.main configs/test/vipcup_seg/network_alchemy_pre_trained_mr_fine_tuned/data_split_{0/1/2}_config.yaml
```
