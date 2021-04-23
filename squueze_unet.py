{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "name": "squueze-unet.py",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KnTnpOV0Bb_W"
      },
      "source": [
        "## Import Libraries"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FGUElCrUJiLR"
      },
      "source": [
        "from __future__ import print_function\n",
        "\n",
        "import os\n",
        "import skimage.io as io\n",
        "import skimage.transform as trans\n",
        "import shutil\n",
        "import cv2\n",
        "import matplotlib.pyplot as plt\n",
        "import pickle\n",
        "import time\n",
        "import glob\n",
        "\n",
        "import numpy as np\n",
        "import keras\n",
        "from keras.models import Model\n",
        "from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D\n",
        "from keras.optimizers import Adam\n",
        "from keras.optimizers import SGD\n",
        "from keras.callbacks import ModelCheckpoint, LearningRateScheduler\n",
        "from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout\n",
        "from keras.layers import concatenate, Conv2DTranspose, BatchNormalization\n",
        "from keras import backend as K\n",
        "\n",
        "from keras import backend as keras\n",
        "\n",
        "from keras.layers import Dropout\n",
        "\n",
        "from sklearn.externals import joblib\n",
        "import argparse\n",
        "from keras.callbacks import *\n",
        "import sys\n",
        "import theano\n",
        "import theano.tensor as T\n",
        "from keras import initializers\n",
        "from keras.layers import BatchNormalization\n",
        "import copy\n",
        "import tensorflow as tf\n",
        "\n",
        "from keras.models import *\n",
        "from keras.layers import *\n",
        "from keras.optimizers import *\n",
        "from tensorflow.keras.models import load_model as load_initial_model\n",
        "from keras.preprocessing.image import ImageDataGenerator\n",
        "from keras.losses import binary_crossentropy\n",
        "\n",
        "import gc"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2pYrbjx1Bi5d"
      },
      "source": [
        "## Handcrafted Metrics For Additional Evaluation and Loss Function"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "W3m-XgyZNECY"
      },
      "source": [
        "def dice_coef(y_true, y_pred):\n",
        "  smooth = 0.0\n",
        "  y_true_f = keras.flatten(y_true)\n",
        "  y_pred_f = keras.flatten(y_pred)\n",
        "  intersection = keras.sum(y_true_f * y_pred_f)\n",
        "  return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)\n",
        "\n",
        "def jacard(y_true, y_pred):\n",
        "\n",
        "  y_true_f = keras.flatten(y_true)\n",
        "  y_pred_f = keras.flatten(y_pred)\n",
        "  intersection = keras.sum ( y_true_f * y_pred_f)\n",
        "  union = keras.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)\n",
        "\n",
        "  return intersection/union\n",
        "\n",
        "def dice_coef_loss(y_true, y_pred):\n",
        "    return 1. - dice_coef(y_true, y_pred)\n",
        "\n",
        "def cross_entropy(p, q):\n",
        "\t  return -sum([p[i]*log2(q[i]) for i in range(len(p))])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AGPnFlmGHuqO"
      },
      "source": [
        "## Squeeze U-Net Model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wxjiXFWpHtuz"
      },
      "source": [
        "def fire_module(x, fire_id, squeeze=16, expand=64):\n",
        "    f_name = \"fire{0}/{1}\"\n",
        "    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1\n",
        "\n",
        "    x = Conv2D(squeeze, (1, 1), activation='relu', padding='same', name=f_name.format(fire_id, \"squeeze1x1\"))(x)\n",
        "    x = BatchNormalization(axis=channel_axis)(x)\n",
        "\n",
        "    left = Conv2D(expand, (1, 1), activation='relu', padding='same', name=f_name.format(fire_id, \"expand1x1\"))(x)\n",
        "    right = Conv2D(expand, (3, 3), activation='relu', padding='same', name=f_name.format(fire_id, \"expand3x3\"))(x)\n",
        "    x = concatenate([left, right], axis=channel_axis, name=f_name.format(fire_id, \"concat\"))\n",
        "    return x\n",
        "\n",
        "\n",
        "def SqueezeUNet(inputs, num_classes=None, deconv_ksize=3, dropout=0.5, activation='sigmoid'):\n",
        "    \"\"\"SqueezeUNet is a implementation based in SqueezeNetv1.1 and unet for semantic segmentation\n",
        "    :param inputs: input layer.\n",
        "    :param num_classes: number of classes.\n",
        "    :param deconv_ksize: (width and height) or integer of the 2D deconvolution window.\n",
        "    :param dropout: dropout rate\n",
        "    :param activation: type of activation at the top layer.\n",
        "    :returns: SqueezeUNet model\n",
        "    \"\"\"\n",
        "    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1\n",
        "    if num_classes is None:\n",
        "        num_classes = K.int_shape(inputs)[channel_axis]\n",
        "\n",
        "    x01 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', name='conv1')(inputs)\n",
        "    x02 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1', padding='same')(x01)\n",
        "\n",
        "    x03 = fire_module(x02, fire_id=2, squeeze=16, expand=64)\n",
        "    x04 = fire_module(x03, fire_id=3, squeeze=16, expand=64)\n",
        "    x05 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3', padding=\"same\")(x04)\n",
        "\n",
        "    x06 = fire_module(x05, fire_id=4, squeeze=32, expand=128)\n",
        "    x07 = fire_module(x06, fire_id=5, squeeze=32, expand=128)\n",
        "    x08 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5', padding=\"same\")(x07)\n",
        "\n",
        "    x09 = fire_module(x08, fire_id=6, squeeze=48, expand=192)\n",
        "    x10 = fire_module(x09, fire_id=7, squeeze=48, expand=192)\n",
        "    x11 = fire_module(x10, fire_id=8, squeeze=64, expand=256)\n",
        "    x12 = fire_module(x11, fire_id=9, squeeze=64, expand=256)\n",
        "\n",
        "    if dropout != 0.0:\n",
        "        x12 = Dropout(dropout)(x12)\n",
        "\n",
        "    up1 = concatenate([\n",
        "        Conv2DTranspose(192, deconv_ksize, strides=(1, 1), padding='same')(x12),\n",
        "        x10,\n",
        "    ], axis=channel_axis)\n",
        "    up1 = fire_module(up1, fire_id=10, squeeze=48, expand=192)\n",
        "\n",
        "    up2 = concatenate([\n",
        "        Conv2DTranspose(128, deconv_ksize, strides=(1, 1), padding='same')(up1),\n",
        "        x08,\n",
        "    ], axis=channel_axis)\n",
        "    up2 = fire_module(up2, fire_id=11, squeeze=32, expand=128)\n",
        "\n",
        "    up3 = concatenate([\n",
        "        Conv2DTranspose(64, deconv_ksize, strides=(2, 2), padding='same')(up2),\n",
        "        x05,\n",
        "    ], axis=channel_axis)\n",
        "    up3 = fire_module(up3, fire_id=12, squeeze=16, expand=64)\n",
        "\n",
        "    up4 = concatenate([\n",
        "        Conv2DTranspose(32, deconv_ksize, strides=(2, 2), padding='same')(up3),\n",
        "        x02,\n",
        "    ], axis=channel_axis)\n",
        "    up4 = fire_module(up4, fire_id=13, squeeze=16, expand=32)\n",
        "    up4 = UpSampling2D(size=(2, 2))(up4)\n",
        "\n",
        "    x = concatenate([up4, x01], axis=channel_axis)\n",
        "    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)\n",
        "    x = UpSampling2D(size=(2, 2))(x)\n",
        "    x = Conv2D(num_classes, (1, 1), activation=activation)(x)\n",
        "    model = Model(inputs, x)\n",
        "    #model.summary()\n",
        "    model.compile(optimizer=Adam(lr = 1e-4), loss=binary_crossentropy, metrics = ['accuracy',dice_coef,jacard,tf.keras.metrics.MeanIoU(num_classes=2),\n",
        "                                                                                      tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])\n",
        "                                                                                      \n",
        "    return model"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iP2fOdnMLDCl"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}