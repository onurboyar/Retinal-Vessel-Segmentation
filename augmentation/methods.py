from matplotlib import pyplot as plt
from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique
import random
import cv2
import albumentations as A
import imageio
import imgaug as ia
import os 

def apply_dropout(input_path, output_path, percentages):

    PROBLEM = "semantic_segmentation"
    ANNOTATION_MODE = "folders"
    INPUT_PATH = input_path
    GENERATION_MODE = "linear"
    OUTPUT_MODE = "folders"
    OUTPUT_PATH= output_path
    LABELS_EXTENSION = ".png"

    augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH,"labelsExtension":LABELS_EXTENSION})

    transformer = transformerGenerator(PROBLEM)

    for percentage in percentages:
        dropout = createTechnique("dropout", {"percentage" : percentage})
        augmentor.addTransformer(transformer(dropout))

    augmentor.applyAugmentation()
    print("Rotation results were saved given directory.")



def apply_gamma_correction(input_path, output_path, gammas):

    PROBLEM = "semantic_segmentation"
    ANNOTATION_MODE = "folders"
    INPUT_PATH = input_path
    GENERATION_MODE = "linear"
    OUTPUT_MODE = "folders"
    OUTPUT_PATH= output_path
    LABELS_EXTENSION = ".png"

    augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH,"labelsExtension":LABELS_EXTENSION})

    transformer = transformerGenerator(PROBLEM)

    for gamma in gammas:
        gamma_t = createTechnique("gamma", {"gamma" : gamma})
        augmentor.addTransformer(transformer(gamma_t))

    augmentor.applyAugmentation()
    print("Rotation results were saved given directory.")


def apply_white_noise(input_path, output_path, sd):

    PROBLEM = "semantic_segmentation"
    ANNOTATION_MODE = "folders"
    INPUT_PATH = input_path
    GENERATION_MODE = "linear"
    OUTPUT_MODE = "folders"
    OUTPUT_PATH= output_path
    LABELS_EXTENSION = ".png"

    augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH,"labelsExtension":LABELS_EXTENSION})

    transformer = transformerGenerator(PROBLEM)

    for sigma in sd:
        white_noise = createTechnique("gaussian_noise", {"mean" : 0,"sigma":sigma})
        augmentor.addTransformer(transformer(white_noise))

    augmentor.applyAugmentation()
    print("Rotation results were saved given directory.")


 def apply_eqhisto(input_path, output_path):

     PROBLEM = "semantic_segmentation"
     ANNOTATION_MODE = "folders"
     INPUT_PATH = input_path
     GENERATION_MODE = "linear"
     OUTPUT_MODE = "folders"
     OUTPUT_PATH= output_path
     LABELS_EXTENSION = ".png"

     augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH,"labelsExtension":LABELS_EXTENSION})

     transformer = transformerGenerator(PROBLEM)
     equalize = createTechnique("equalize_histogram",{})
     augmentor.addTransformer(transformer(equalize)) 
     augmentor.applyAugmentation()

     print("equalize histogram results were saved given directory.")

def aug_blurring(input_path, output_path, blurr:list):

    PROBLEM = "semantic_segmentation"
    ANNOTATION_MODE = "folders"
    INPUT_PATH = input_path
    GENERATION_MODE = "linear"
    OUTPUT_MODE = "folders"
    OUTPUT_PATH= output_path
    LABELS_EXTENSION = ".png"

    augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH,"labelsExtension":LABELS_EXTENSION})
    transformer = transformerGenerator(PROBLEM)

    for ker in blurr:
        blur = createTechnique("blurring", {"kernel" : ker})
        augmentor.addTransformer(transformer(blur))
        print("Blurring for kernel = {} is done".format(ker))

    augmentor.applyAugmentation()
    print("Augmentation results were saved given directory.")


def apply_elastic_deformation(input_path, output_path, alpha = 5, sigma = 0.05):

    PROBLEM = "semantic_segmentation"
    ANNOTATION_MODE = "folders"
    INPUT_PATH = input_path
    GENERATION_MODE = "linear"
    OUTPUT_MODE = "folders"
    OUTPUT_PATH= output_path
    LABELS_EXTENSION = ".png"

    augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH,"labelsExtension":LABELS_EXTENSION})

    transformer = transformerGenerator(PROBLEM)

    rotate = createTechnique("elastic", {"alpha" : alpha, "sigma" : sigma})
    augmentor.addTransformer(transformer(rotate))

    augmentor.applyAugmentation()
    print("Elastic deformation results were saved given directory.")


def apply_flipping(input_path, output_path, flip):

    PROBLEM = "semantic_segmentation"
    ANNOTATION_MODE = "folders"
    INPUT_PATH = input_path
    GENERATION_MODE = "linear"
    OUTPUT_MODE = "folders"
    OUTPUT_PATH= output_path
    LABELS_EXTENSION = ".png"

    augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH,"labelsExtension":LABELS_EXTENSION})

    transformer = transformerGenerator(PROBLEM)

    rotate = createTechnique("flip", {"flip" : flip})
    augmentor.addTransformer(transformer(rotate))

    augmentor.applyAugmentation()
    print("Flipping results were saved given directory.")


def apply_shearing(input_path, output_path, a = 0.5):

    PROBLEM = "semantic_segmentation"
    ANNOTATION_MODE = "folders"
    INPUT_PATH = input_path
    GENERATION_MODE = "linear"
    OUTPUT_MODE = "folders"
    OUTPUT_PATH= output_path
    LABELS_EXTENSION = ".png"

    augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH,"labelsExtension":LABELS_EXTENSION})
    transformer = transformerGenerator(PROBLEM)

    rotate = createTechnique("shearing", {"a" : a})
    augmentor.addTransformer(transformer(rotate))

    augmentor.applyAugmentation()
    print("Shearing results were saved given directory.")

def apply_sharpen(input_path, output_path):

    PROBLEM = "semantic_segmentation"
    ANNOTATION_MODE = "folders"
    INPUT_PATH = input_path
    GENERATION_MODE = "linear"
    OUTPUT_MODE = "folders"
    OUTPUT_PATH= output_path
    LABELS_EXTENSION = ".png"

    augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH,"labelsExtension":LABELS_EXTENSION})

    transformer = transformerGenerator(PROBLEM)

    rotate = createTechnique("sharpen", {})
    augmentor.addTransformer(transformer(rotate))

    augmentor.applyAugmentation()
    print("Sharping results were saved given directory.")


def apply_raise_satur(input_path, output_path, power):

    PROBLEM = "semantic_segmentation"
    ANNOTATION_MODE = "folders"
    INPUT_PATH = input_path
    GENERATION_MODE = "linear"
    OUTPUT_MODE = "folders"
    OUTPUT_PATH= output_path
    LABELS_EXTENSION = ".png"

    augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH,"labelsExtension":LABELS_EXTENSION})

    transformer = transformerGenerator(PROBLEM)

    rotate = createTechnique("raise_saturation", {"power" : power})
    augmentor.addTransformer(transformer(rotate))

    augmentor.applyAugmentation()
    print("Raise saturation results were saved given directory.")


def apply_jpeg_compression(input_path, output_path, degrees):

    images = []
    labels = []

    for img_path in range(20):
        img = imageio.imread(input_path + 'images/' + str(img_path) + '.png')
        images.append(img)

        lbl = imageio.imread(input_path + 'labels/' + str(img_path) + '.png')
        labels.append(lbl)

    path = os.path.join(output_path, 'images')
    os.mkdir(path)

    path = os.path.join(output_path, 'labels')
    os.mkdir(path)

  for degree in degrees:
      aug = ia.augmenters.JpegCompression(compression=degree)
      images_aug = aug.augment_images(images=images)
      for indx, i in enumerate(images_aug):
          imageio.imwrite(output_path + 'images/' + str(degree) + '_' + str(indx) + '.png', i)

    labels_aug = aug.augment_images(images=labels)
    for indx, i in enumerate(labels_aug):
        imageio.imwrite(output_path + 'labels/' + str(degree) + '_' + str(indx) + '.png', i)

  print("JPEG Compression results were saved given directory.")

