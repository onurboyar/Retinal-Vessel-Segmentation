from keras.preprocessing.image import ImageDataGenerator
from keras import backend as keras
import numpy as np
import cv2, os
import skimage.transform as trans
import skimage.io as io



def trainGenerator(batch_size,train_path,image_folder,\
                   mask_folder,aug_dict,image_color_mode = "grayscale",\
                   mask_color_mode = "grayscale",image_save_prefix  = "image",\
                   mask_save_prefix  = "mask",flag_multi_class = False,\
                   seed = 1,save_to_dir = None,\
                   target_size = (608,576), num_class = 2):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def testGenerator(test_path, target_size = (608,576),flag_multi_class = False,as_gray = True):

    image_datagen = ImageDataGenerator(rescale=1./255)
    mask_datagen = ImageDataGenerator(rescale=1./255)

    for img_name in sorted(os.listdir(test_path + "/images")):
        img = io.imread(os.path.join(test_path + "/images",img_name),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def testGenerator2(batch_size,test_path,image_folder,mask_folder,\
                   aug_dict,image_color_mode = "grayscale",\
                   mask_color_mode = "grayscale",image_save_prefix  = "image",\
                   mask_save_prefix  = "mask",flag_multi_class = False, seed = 1,\
                   save_to_dir = None,target_size = (608,576), num_class = 2):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        test_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        test_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    test_generator = zip(image_generator, mask_generator)
    for (img,mask) in test_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

