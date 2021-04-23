# Retinal Vessel Segmentation

TODO



## Documentation

Images are preprocessed (padding, normalizing etc.) in training function, so executing traininn files is enough. If you want to train [DRIVE: Digital Retinal Images for Vessel Extraction](https://drive.grand-challenge.org/) dataset,

```bash
python3 train_drive.py --train_at_once="True"
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
