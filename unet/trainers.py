import os, argparse, shutil, cv2, pickle, time, logging, gc, json

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from tensorflow.keras.models import load_model as load_initial_model


from unet.attention_unet import AttentionUNet
from unet.vanilla_unet import UNet
from utils.metrics import dice, mean_dice
from utils.preprocess import adjustData, pad, crop, threshold
from utils.file import get_dirs, set_order, saveResult_drive, saveResult
from data.generators import trainGenerator, testGenerator, testGenerator2

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



TRAIN_PATH_IMG    = dirs["files"][0]["TRAIN_PATH"] + "/images"
TRAIN_PATH_MASK   = dirs["files"][0]["TRAIN_PATH"] + "/labels"
KFOLD_TEMP_TRAIN  = dirs["files"][0]["KFOLD_TEMP_TRAIN"]
KFOLD_TEMP_TEST   = dirs["files"][0]["KFOLD_TEMP_TEST"]

LOG_PATH_K        = dirs["files"][0]["LOG_PATH_KFOLD"]
CKPTS_PATH        = dirs["files"][0]["CKPTS_PATH_KFOLD"]
RESULTS_PATH      = dirs["files"][0]["RESULTS_PATH_KFOLD"]


def train_once(save_name, num_train, num_test, initial_model_path,\
               train_batch = 3, test_batch = 3, epoch = 5, already_padded = False,\
               model_name = "vanilla"):

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

    pad(TRAIN_PATH + '/images', TMP_TRAIN + '/images', already_padded)
    pad(TRAIN_PATH + '/labels', TMP_TRAIN + '/labels', already_padded)

    pad(TEST_PATH + '/images', TMP_TEST + '/images',already_padded)
    pad(TEST_PATH + '/labels', TMP_TEST + '/labels',already_padded)

    pad(VAL_PATH + '/images', TMP_VAL + '/images',already_padded)
    pad(VAL_PATH + '/labels', TMP_VAL + '/labels',already_padded)


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

    if os.path.isdir(RESULT_PATH + "/download"):
        shutil.rmtree(RESULT_PATH + '/download', ignore_errors=False, onerror=None)

    os.mkdir(RESULT_PATH + "/download")
    #shutil.rmtree(RESULT_PATH + '/download', ignore_errors=False, onerror=None)
    #os.mkdir(RESULT_PATH + '/download')
    set_order(RESULT_PATH + '/' + save_name, RESULT_PATH + '/download')



def train_loop(save_name, num_train, num_test, initial_model_path,\
               train_batch = 3, test_batch = 3, epoch = 5, already_padded = False,\
               model_name = "vanilla"):


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

    pad(TRAIN_PATH + '/images', TMP_TRAIN + '/images', already_padded)
    pad(TRAIN_PATH + '/labels', TMP_TRAIN + '/labels', already_padded)

    pad(TEST_PATH + '/images', TMP_TEST + '/images',already_padded)
    pad(TEST_PATH + '/labels', TMP_TEST + '/labels',already_padded)

    pad(VAL_PATH + '/images', TMP_VAL + '/images',already_padded)
    pad(VAL_PATH + '/labels', TMP_VAL + '/labels',already_padded)


    data_gen_args = dict()
    train_generator = trainGenerator(train_batch, TMP_TRAIN, 'images', 'labels', data_gen_args, save_to_dir = None, target_size=(608,576))
    test_generator = testGenerator2(test_batch, TMP_TEST, 'images', 'labels', data_gen_args, save_to_dir = None, target_size=(608,576))

    for epoch in range(epochs):

        if model_name == "vanilla":
            model = UNet(input_size=(608,576,1))
        elif model_name == "attention":
            model = AttentionUNet(input_size=(608,576,1))

        if epoch != 0:
            model.load_weights(f'{MODEL_PATH}/{save_name +"_"+ str(epoch-1)}.hdf5')
        else:
            if initial_model_name != None:
                model.load_weights(initial_model_path)


        model_history = model.fit_generator(train_generator, steps_per_epoch=num_train//train_batch, \
                                            epochs=1, callbacks=[model_checkpoint],\
                                            validation_data=test_generator, validation_steps=num_test//test_batch)


        log_file = open(LOG_PATH + "/log_{}.pkl".format(save_name + str(epoch)), "wb")#history file
        pickle.dump(model_history.history, log_file)
        log_file.close()

        test_generator_2 = testGenerator(TMP_TEST, target_size=(608,576))
        results = model.predict_generator(test_generator_2,verbose=1)


        shutil.rmtree(TMP_RESULT, ignore_errors=False, onerror=None)
        os.mkdir(TMP_RESULT)
        saveResult_drive(TMP_RESULT, results)

        os.mkdir(RESULT_PATH + '/' + save_name + str(epoch))
        crop(TMP_RESULT, RESULT_PATH + '/' + save_name + str(epoch))

        threshold(RESULT_PATH + '/' + save_name + str(epoch))

        os.mkdir(RESULT_PATH + '/download' + "_"+save_name + str(epoch))
        set_order(RESULT_PATH + '/' + save_name + str(epoch), RESULT_PATH + '/download' + "_"+save_name + str(epoch))

        mean_dice_coef = mean_dice(TEST_PATH   + '/labels',
                                   RESULT_PATH + '/download_' + save_name + str(epoch))

        print(f"\nMean dice coeff at epoch {epoch}: {mean_dice_coef}\n\n")
        print("-"*20)




def train_kfold_stare(epoch, start, train_batch_size, \
                      test_batch_size, train_sample_number, \
                      test_sample_number, initial_model_path, \
                      k=5,show_samples=False, model_name = "vanilla"):

    assert 20 % k ==0, "Number of images divided by fold number must be integer."
    NOF_PLOTS = 0

    for i in range(start, int(20/k), 1):
        test_images_temp = [j for j in range(k)]
        test_images_temp_2 = [a + i*k for a in test_images_temp]
        test_images = [str(a) for a in test_images_temp_2] #our test ids
        print("Test images: {}".format(test_images))


        shutil.rmtree(KFOLD_TEMP_TRAIN, ignore_errors=False, onerror=None)
        os.mkdir(KFOLD_TEMP_TRAIN)
        os.mkdir(KFOLD_TEMP_TRAIN + "/images")
        os.mkdir(KFOLD_TEMP_TRAIN + "/labels")

        shutil.rmtree(KFOLD_TEMP_TEST, ignore_errors=False, onerror=None)
        os.mkdir(KFOLD_TEMP_TEST)
        os.mkdir(KFOLD_TEMP_TEST + "/images")
        os.mkdir(KFOLD_TEMP_TEST + "/labels")

        for test_image in test_images: #allocates test images into the path
            src = TRAIN_PATH_IMG + "/" + test_image + ".png"
            shutil.copy(src, KFOLD_TEMP_TEST + "/images")

            src = TRAIN_PATH_MASK + "/" + test_image + ".png"
            shutil.copy(src, KFOLD_TEMP_TEST + "/labels")

        for img in sorted(os.listdir(TRAIN_PATH_IMG)): #allocates train images into the path
            img_splitted_1 = img.split("_")
            img_splitted_2 = img.split(".")
            if (img_splitted_1[-1].split(".")[0] not in test_images) and (img_splitted_2[0] not in test_images):
                src = TRAIN_PATH_IMG + "/" + img
                shutil.copy(src, KFOLD_TEMP_TRAIN + "/images")

                src = TRAIN_PATH_MASK + "/" + img
                shutil.copy(src, KFOLD_TEMP_TRAIN + "/labels")


        data_gen_args = dict()
        train_generator = trainGenerator(train_batch_size,KFOLD_TEMP_TRAIN,\
                                         'images','labels',data_gen_args,\
                                         save_to_dir = None, target_size = (608,704))

        test_generator = testGenerator2(test_batch_size,KFOLD_TEMP_TEST,\
                                        'images','labels',data_gen_args,\
                                        save_to_dir = None, target_size = (608,704))


        if model_name == "vanilla":
            model = UNet(input_size=(608,704,1))
        elif model_name == "attention":
            model = AttentionUNet(input_size=(608,704,1))

        if initial_model_path != None:
            model.load_weights(initial_model_path)

        model_checkpoint = ModelCheckpoint(CKPTS_PATH + "/fold_{}_unet_stare.hdf5".format(i), monitor='loss',verbose=1, save_best_only=True)

        model_history = model.fit_generator(train_generator,steps_per_epoch=train_sample_number//train_batch_size,\
                                            epochs=epoch, callbacks=[model_checkpoint],validation_data=test_generator,\
                                            validation_steps=test_sample_number//test_batch_size)

        log_file = open(LOG_PATH_K + "/log_fold_{}.pkl".format(i), "wb")#history file
        pickle.dump(model_history.history, log_file)
        log_file.close()

        test_generator_2 = testGenerator(KFOLD_TEMP_TEST, target_size = (608,704))
        results = model.predict_generator(test_generator_2,k,verbose=1)
        saveResult(i,k,RESULTS_PATH,results)

        del model

        if show_samples:
            fig, axs = plt.subplots(k + NOF_PLOTS,3,figsize=(17,17))
            for idx,item in enumerate(sorted(os.listdir(RESULTS_PATH))):
                item_without_predict = item.split("_")[0] + ".png"
                img_real = cv2.imread(TRAIN_PATH_IMG + "/" + item_without_predict)
                img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
                #print(TRAIN_PATH_IMG + "/" + item_without_predict)
                #print(os.path.isfile(TRAIN_PATH_IMG + "/" + item_without_predict))
                img_ground_truth = cv2.imread(TRAIN_PATH_MASK + "/" + item_without_predict)
                img_ground_truth = cv2.cvtColor(img_ground_truth, cv2.COLOR_BGR2RGB)
                img_predict = cv2.imread(RESULTS_PATH + "/" + item)
                img_predict = cv2.cvtColor(img_predict, cv2.COLOR_BGR2RGB)

                axs[idx,0].imshow(img_real)
                axs[idx,0].title.set_text('image_{}'.format(item_without_predict))
                axs[idx,1].imshow(img_ground_truth)
                axs[idx,1].title.set_text('ground truth')
                axs[idx,2].imshow(img_predict)
                axs[idx,2].title.set_text('predicted')
            plt.show()
            NOF_PLOTS += k
