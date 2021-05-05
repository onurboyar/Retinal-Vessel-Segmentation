# Retinal Vessel Segmentation

Successful segmentation of the retinal vessel segmentation has widely studied and it is still one of the hot research areas. Various different architectures are tailored for this specific problem and numerous existed deep learning architectures are used in order to perform the segmentation task. The scarcity of the annotated data pushed researchers to use data augmentation to the certain amount in order to avoid the overfitting problem. However, the usage of the data augmentation is limited in these studies. If data augmentation strategy can address the problems of the input data, successful segmentation model can be obtained. In our study, we are looking for the performance gains that can be obtained by the excessive data augmentation using U-Net architecture for retinal vessel segmentation problem. We use DRIVE dataset that has become one of the standard benchmarks in the retinal vessel segmentation studies. 


## Documentation

### Path Structure

#### DRIVE

```
"LOG_PATH": "./data/logs", (contains log file as pickle)
"RESULT_PATH": "./data/test_results", (contains results as a folder named $save_name, final results are in here.
If you want to make a submission, use images in"/download" folder)
"MODEL_PATH": "./data/models", (contains model checkpoints)
"TRAIN_PATH": "./data/train",  (contains directories /images and /labels with related images)
"VAL_PATH": "./data/test", (contains directories /images and /labels with related images) 
"TEST_PATH": "./data/test", (contains directories /images and /labels with related images)
"TMP_TRAIN": "./data/tmp_train", (contains padded training images and labels, script creates all automatically)
"TMP_TEST": "./data/tmp_test", (contains padded test images and labels, script creates all automatically)
"TMP_VAL": "./data/tmp_val", (contains padded validation images and labels, script creates all automatically)
"TMP_RESULT": "./data/tmp_result", (contains raw predictions, script creates all automatically)
```

#### STARE
```
"TRAIN_PATH": "./data/train",  (contains directories /images and /labels with related images)
"VAL_PATH": "./data/test", (contains directories /images and /labels with related images) 
"TEST_PATH": "./data/test", (contains directories /images and /labels with related images)
"KFOLD_TEMP_TRAIN": "./kfold/temp_train", (contains padded training images and labels, script creates all automatically)
"KFOLD_TEMP_TEST": "./kfold/temp_test", (contains padded test images and labels, script creates all automatically)
"LOG_PATH_KFOLD": "./kfold/logs", (contains log file as pickle)
"CKPTS_PATH_KFOLD": "./kfold/checkpoints", (contains model checkpoints)
"RESULTS_PATH_KFOLD": "./kfold/results" (contains results as a folder named $save_name, final results are in here)
```

### Training DRIVE

Images are preprocessed (padding, normalizing etc.) in training function, so executing training files is enough. If you want to train [DRIVE: Digital Retinal Images for Vessel Extraction](https://drive.grand-challenge.org/) dataset follow these steps:

- Since DRIVE dataset gives training images with ".tif" and labels with ".gif" extension, you must give ".png" files to model.
- Other image preprocessing is done at training loop (binary masking, RGB to gray scale etc.).
- Your images must be multiples of 32 (our choice is 608x576 since DRIVE has resolution 584x565). If your images have been already padded, give --already_padded=True.
- If you want to save models at each epoch, give --train_at_once=False, else only the best model will be saved.


```bash
python3 train_drive.py --train_at_once True \
                       --save_name "experiment_1" \
                       --initial_model_path "/path/to/ckpts.hdf5" \
                       --model_name "vanilla" \
                       --epochs 15 \
                       --train_batch 3 \
                       --val_batch 3 \
                       --already_padded False
```

### Traning STARE

If you want to train [STARE: STructured Analysis of the Retina](https://cecas.clemson.edu/~ahoover/stare/) dataset follow these steps:

- Since STARE dataset gives training images with ".ppm" and labels with ".ppm" extension, you must give ".png" files to model.
- STARE dataset doesn't give you test images, so we follow k-fold procedure while training.
- STARE scripts doesn't contain padding script, so you must give padded images to training (examples are shown in [preprocess_for_DRIVE_dataset.ipynb](https://github.com/onurboyar/Retinal-Vessel-Segmentation/blob/main/notebooks/preprocess_for_DRIVE_dataset.ipynb)).
- If you interrupt k-fold training, you can specify starting fold with --start, after.
- If you specified starting fold, give --show_samples False.

```bash
python3 train_stare.pt --initial_model_path "/path/to/ckpts.hdf5"\
                       --model_name "vanilla" \
                       --epochs 15 \
                       --train_batch 3 \
                       --val_batch 3 \
                       --n_fold 5 \
                       --start_fold 5 \
                       --show_samples False
```

## Augmentations

We used [imgaug](https://github.com/aleju/imgaug), [CLoDSA](https://github.com/joheras/CLoDSA), [albumentations](https://github.com/albumentations-team/albumentations), [imagecorruptions](https://github.com/bethgelab/imagecorruptions) libraries for excessive augmentations. Before using augmentation scripts, run setup_augmentation.sh file "/augmentations". Since we used diverse augmentation techniques, we couldn't provide arguments for python files to augmentations. We provide this diverse augmentation techniques in methods.py file, you must edit augmentation_main.py file before you run.

## Results On DRIVE

...                                                         | Accuracy       | AUC           | Mean Dice Coef    | Challenge Ranking      |
:---                                                        | :---:          |    :----:     |          :---:    |  :---:                 |
Rotation (30*k) And Flipping                                | 0,970          | 0,971         | 0,809             | 621                    |
\+ Zoom Out                                                  | 0,971          | 0,983         | 0,820             | 323                   |
\+ White Noise/Elastic Deformations/Shift                    | 0,970          | 0,985         | 0,822             | 257                   |
\+ Gamma Correction/Random Crop/Grid and Optical Distortion  | 0,971          | 0,983         | 0,824             | 171                   |
\+ Blurring/Dropout/Eq. Histogram                            | 0,971          | **0,985**     | **0,826**         | **127**               |


Method                   |     AUC       |     Accuracy     |
:---                     |    :---:      |     :---:        |
UNet (2018*)             |0,9752         |0,9555            | 
Residual UNet (2018)     |0,9779         |0,9553            |      
IterNet (2019)           |0,9816         |0,9571            |
SUD-GAN (2020)           |0,9786         |0,9560            |
RV-GAN (2020)            |*0,9887*       |*0,9790*          |
Ours (2021)              |**0,9848**     |**0,9712**        |

## Authors
- Enes Sadi Uysal
- M. Şafak Bilici
- Billur Selin Zaza
- Mehmet Yiğit Özgenç
- Onur Boyar
