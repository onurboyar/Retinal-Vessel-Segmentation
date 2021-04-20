from augmentation.methods import *
import os

do_augment_path = "./data/train/"


wn_10 = "./augmentation/"
if not os.path.exists(wn_10):
    os.mkdir(wn_10)
    apply_white_noise(do_augment_path,wn_10,[10])
else:
    print("This augmentation technique has been already done.")



