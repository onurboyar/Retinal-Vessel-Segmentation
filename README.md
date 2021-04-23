# Retinal Vessel Segmentation

Successful segmentation of the retinal vessel segmentation has widely studied and it is still one of the hot research areas. Various different architectures are tailored for this specific problem and numerous existed deep learning architectures are used in order to perform the segmentation task. The scarcity of the annotated data pushed researchers to use data augmentation to the certain amount in order to avoid the overfitting problem. However, the usage of the data augmentation is limited in these studies. If data augmentation strategy can address the problems of the input data, successful segmentation model can be obtained. In our study, we are looking for the performance gains that can be obtained by the excessive data augmentation using U-Net architecture for retinal vessel segmentation problem. We use DRIVE dataset that has become one of the standard benchmarks in the retinal vessel segmentation studies. 


## Documentation

Images are preprocessed (padding, normalizing etc.) in training function, so executing training files is enough. If you want to train [DRIVE: Digital Retinal Images for Vessel Extraction](https://drive.grand-challenge.org/) dataset,

```bash
python3 train_drive.py --train_at_once="True" \
                       --save_name="experiment 1" \
                       --initial_model_path="/path/to/ckpts.hdf5" \
                       --model_name="vanilla" \
                       --epochs=15 \
                       --train_batch=3 \
                       --val_batch=3 \
                       --already_padded=False
```


## Authors
- Enes Sadi Uysal
- M. Şafak Bilici
- Billur Selin Zaza
- Mehmet Yiğit Özgenç
- Onur Boyar
