import os, cv2
import skimage.io as io
import skimage.transform as trans
import numpy as np

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) \
            if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif (np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def pad(input_folder, output_folder,already_padded):
    for file in sorted(os.listdir(input_folder)):
        if '.png' in file:
            tmp = cv2.imread(input_folder + '/' + file, 0)
            if not already_padded:
                tmp = cv2.copyMakeBorder(tmp.copy(),12,12,5,6,cv2.BORDER_CONSTANT,value=(0,0,0))
            io.imsave(output_folder + '/' + file, tmp)
    print('Padding is done.')


def crop(input_folder, output_folder):
    for file in sorted(os.listdir(input_folder)):
        if '.png' in file:
            tmp = cv2.imread(input_folder + '/' + file, 0)
            tmp = tmp[12:-12, 5:-6]
            io.imsave(output_folder + '/' + file, tmp)
    print('Cropping is done.')


def threshold(folder):
    for img in sorted(os.listdir(folder)):
        tmp = cv2.imread(folder + '/' + img, 0)
        _, tmp = cv2.threshold(tmp,115,255,cv2.THRESH_BINARY)
        io.imsave(folder + '/' + img, tmp)


