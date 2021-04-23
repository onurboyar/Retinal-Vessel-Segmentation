import json, os, cv2
import skimage.io as io
import numpy as np


def get_dirs():
    if not os.path.isdir("./kfold"):
        os.mkdir("./kfold")
    with open("./file_config.json") as json_f:
        dirs = json.load(json_f)
    for dir in dirs["files"][0]:
        if not os.path.exists(dirs["files"][0][dir]):
            os.makedirs(dirs["files"][0][dir])
    return dirs

def set_order(input_folder, output_folder):

    for img in sorted(os.listdir(input_folder)):

        if int(img.split('.')[0]) == 0:
            tmp = cv2.imread(input_folder  + '/' + img, 0)
            io.imsave(output_folder + '/' + str(int(img.split('.')[0])+1) + '.png', tmp)
        elif int(img.split('.')[0]) >= 1 and int(img.split('.')[0]) <= 10:
            tmp = cv2.imread(input_folder  + '/' + img, 0)
            io.imsave(output_folder + '/' + str(int(img.split('.')[0])+9) + '.png', tmp)
        elif int(img.split('.')[0]) >= 13 and int(img.split('.')[0]) <= 19:
            tmp = cv2.imread(input_folder  + '/' + img, 0)
            io.imsave(output_folder + '/' + str(int(img.split('.')[0])-10) + '.png', tmp)
        elif int(img.split('.')[0]) == 11:
            tmp = cv2.imread(input_folder  + '/' + img, 0)
            io.imsave(output_folder + '/' + str(int(img.split('.')[0])-9) + '.png', tmp)
        elif int(img.split('.')[0]) == 12:
            tmp = cv2.imread(input_folder  + '/' + img, 0)
            io.imsave(output_folder + '/' + str(int(img.split('.')[0])+8) + '.png', tmp)


def saveResult_drive(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d.png"%(i)),img)


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(fold,k,save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%(i+fold*k)),img)



