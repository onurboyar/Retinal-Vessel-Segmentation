import os
import shutil
import skimage.io as io
import skimage.transform as trans
import cv2
import matplotlib.pyplot as plt
import pickle
import time
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from tensorflow.keras.models import load_model as load_initial_model
import gc

from unet.attention_unet import AttentionUNet
from unet.vanilla_unet import UNet
from unet.utils.metrics import dice, mean_dice
from unet.utils.preprocess import adjustData, pad, crop, threshold
from unet.utils.file import get_dirs, set_order, saveResult_drive
from unet.data.generators import trainGenerator, testGenerator, testGenerator2


dirs = get_dirs()
LOG_PATH     = dirs["files"][0]["LOG_PATH"]
RESULT_PATH  = dirs["files"][0]["RESULT_PATH"]
MODEL_PATH   = dirs["files"][0]["MODEL_PATH"]

TRAIN_PATH   = dirs["files"][0]["TRAIN_PATH"]
TEST_PATH    = dirs["files"][0]["TEST_PATH"]
VAL_PATH     = dirs["files"][0]["VAL_PATH"]

TMP_TRAIN    = dirs["files"][0]["TMP_TRAIN"]
TMP_TEST     = dirs["files"][0]["TMP_TEST"]
TMP_VAL      = dirs["files"][0]["TMP_VAL"]
TMP_RESULT   = dirs["files"][0]["TMP_RESULT"]



def train_once(save_name, num_train, num_test, initial_model_path,\
          train_batch = 3, test_batch = 3, epoch = 5, already_padded = False,\
          model_name = "vanilla"):

    if already_padded == False:
        shutil.rmtree(TMP_TRAIN, ignore_errors=False, onerror=None)
        os.mkdir(TMP_TRAIN)
        os.mkdir(TMP_TRAIN + "/images")
        os.mkdir(TMP_TRAIN + "/labels")

        shutil.rmtree(TMP_TEST, ignore_errors=False, onerror=None)
        os.mkdir(TMP_TEST)
        os.mkdir(TMP_TEST + "/images")
        os.mkdir(TMP_TEST + "/labels")

        shutil.rmtree(TMP_VAL, ignore_errors=False, onerror=None)
        os.mkdir(TMP_VAL)
        os.mkdir(TMP_VAL + "/images")
        os.mkdir(TMP_VAL + "/labels")

        pad(TRAIN_PATH + '/images', TMP_TRAIN + '/images')
        pad(TRAIN_PATH + '/labels', TMP_TRAIN + '/labels')

        pad(TEST_PATH + '/images', TMP_TEST + '/images')
        pad(TEST_PATH + '/labels', TMP_TEST + '/labels')

        pad(VAL_PATH + '/images', TMP_VAL + '/images')
        pad(VAL_PATH + '/labels', TMP_VAL + '/labels')


    data_gen_args = dict()
    train_generator = trainGenerator(train_batch, TMP_TRAIN, 'images', 'labels', data_gen_args, save_to_dir = None, target_size=(608,576))
    test_generator = testGenerator2(test_batch, TMP_TEST, 'images', 'labels', data_gen_args, save_to_dir = None, target_size=(608,576))

    if model_name == "vanilla":
        model = UNet(input_size=(608,576,1))
    elif model_name == "attention":
        model = AttentionUNet(input_size=(608,576,1))

    if initial_model_path != None:
        model.load_weights(initial_model_path)

    model_checkpoint = ModelCheckpoint(MODEL_PATH + "/" + save_name +".hdf5", monitor='loss',verbose=1, save_best_only=True)

    model_history = model.fit_generator(train_generator, steps_per_epoch=num_train//train_batch, \
                                        epochs=epoch, callbacks=[model_checkpoint],\
                                        validation_data=test_generator, validation_steps=num_test//test_batch)

    log_file = open(LOG_PATH + "/log_{}.pkl".format(save_name), "wb")#history file
    pickle.dump(model_history.history, log_file)
    log_file.close()

    test_generator_2 = testGenerator(TMP_TEST, target_size=(608,576))
    results = model.predict_generator(test_generator_2,verbose=1)

    shutil.rmtree(TMP_RESULT, ignore_errors=False, onerror=None)
    os.mkdir(TMP_RESULT)
    saveResult_drive(TMP_RESULT, results)

    os.mkdir(RESULT_PATH + '/' + save_name)
    crop(TMP_RESULT, RESULT_PATH + '/' + save_name)

    threshold(RESULT_PATH + '/' + save_name)

    shutil.rmtree(RESULT_PATH + '/download', ignore_errors=False, onerror=None)
    os.mkdir(RESULT_PATH + '/download')
    set_order(RESULT_PATH + '/' + save_name, RESULT_PATH + '/download')


if __name__ == "__main__":
    train_sample_number = len(os.listdir(TRAIN_PATH + '/images'))
    test_sample_number  = len(os.listdir(TEST_PATH + '/images'))

    SAVE_NAME = 'my_model'
    INITIAL_MODEL_PATH = None
    EPOCH = 1

    train_once(SAVE_NAME, initial_model_path= INITIAL_MODEL_PATH, epoch= EPOCH, \
               train_batch = 3, test_batch = 3, num_train = train_sample_number, \
               num_test= test_sample_number)



