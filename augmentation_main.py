import os, shutil
from augmentation.methods import *

def apply():
    do_augment_path = "./data/train/"
    wn_10 = "./augmentation/wn_10/"
    if not os.path.exists(wn_10):
        os.mkdir(wn_10)
        apply_white_noise(do_augment_path,wn_10,[10])
    else:
        print("This augmentation technique has been already done.")

def merge_augmentations(augment_dir, output_dir, list_of_aug_files):

    os.mkdir(output_dir + '/images')
    os.mkdir(output_dir + '/labels')

    for folder in list_of_aug_files:
        if folder != 'base':
            for file in sorted(os.listdir(augment_dir + '/' + folder + '/images')):
                shutil.copy(augment_dir + '/' + folder + '/images/' + file, output_dir + '/images/' + folder + '_' + file.split("_")[-1].split('.')[0] + '.png')
                shutil.copy(augment_dir + '/' + folder + '/labels/' + file, output_dir + '/labels/' + folder + '_' + file.split("_")[-1].split('.')[0] + '.png')
                #shutil.copy(augment_dir + '/' + folder + '/images/' + file, output_dir + '/images/' + file)
                #shutil.copy(augment_dir + '/' + folder + '/labels/' + file, output_dir + '/labels/' + file)
        else:
            for file in sorted(os.listdir(augment_dir + '/' + folder + '/images')):
                shutil.copy(augment_dir + '/' + folder + '/images/' + file, output_dir + '/images/' + file.split("_")[-1].split('.')[0] + '.png')
                shutil.copy(augment_dir + '/' + folder + '/labels/' + file, output_dir + '/labels/' + file.split("_")[-1].split('.')[0] + '.png')
                #shutil.copy(augment_dir + '/' + folder + '/images/' + file, output_dir + '/images/' + file)
                #shutil.copy(augment_dir + '/' + folder + '/labels/' + file, output_dir + '/labels/' + file)

        print(folder + ' folder has been merged...')
        print('Number of images in output: ' + str(len(os.listdir(output_dir + '/images'))))
    print('Merging is done successfully!')


if __name__ == "__main__":

    apply()

    augment_dir = "./augmentation"
    if not os.path.exists(augment_dir):
        os.mkdir(augment_dir)

    merge_augmentations_path = "./data/augmentation/augment_id_1"
    if not os.path.exists(merge_augmentations_path):
      if not os.path.exists("./data/augmentation"):
        os.mkdir("./data/augmentation")
      os.mkdir(merge_augmentations_path)

    if not os.path.exists("./augmentation/train"):
        os.mkdir("./augmentation/train/")
        #os.mkdir("./augmentation/train/images/")
        #os.mkdir("./augmentation/train/labels/")
        shutil.copytree("./data/train/images/", "./augmentation/train/images/")
        shutil.copytree("./data/train/labels/", "./augmentation/train/labels/")

    augment_list = ["train","wn_10"]

    merge_augmentations(augment_dir, merge_augmentations_path, augment_list)  



