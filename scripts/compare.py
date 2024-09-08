from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage import io, color

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import sys
sys.path.append('..')

import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.utils import to_categorical
import tifffile

from keras.utils import normalize
from keras.metrics import MeanIoU
from keras.models import load_model

SIZE_X = 128
SIZE_Y = 128
n_classes=4
activation='softmax'
learning_rate=0.0001
optim=keras.optimizers.Adam(learning_rate) # Faster learning rates may not converge

BACKBONE_RESNET50 = 'resnet50'
BACKBONE_VGG19 = 'vgg19'

def main():
    # Capture training image info as a list
    train_images = []
    train_images_gray = []

    for directory_path in glob.glob("../sandstone_data_for_ML/full_labels_for_deep_learning/128_patches/images"):
        for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
            train_images_gray = io.imread(img_path, as_gray=False)

    for image in train_images_gray:
        train_images.append(color.gray2rgb(image))

    train_images = np.array(train_images)
    print(train_images.shape)

    # Capture mask/label info as a list
    train_masks = []

    for directory_path in glob.glob("../sandstone_data_for_ML/full_labels_for_deep_learning/128_patches/masks"):
        for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
            # mask = io.imread(mask_path)
            #train_masks.append(mask)
            train_masks = io.imread(mask_path)

    train_masks = np.array(train_masks)
    print(train_masks.shape)


    labelencoder = LabelEncoder()
    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    np.unique(train_masks_encoded_original_shape)

    print("train_images_shape: ", train_images.shape)
    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3) # (1600, 128, 128, 1); 1 channel for mask

    # Create a subset of data for quick testing
    # Picking 10% for testing and remaining for training
    x1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size=0.10, random_state=0)

    # Further split training data to validation data
    X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(x1, y1, test_size=0.2, random_state=0)

    # Check the data
    print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled regions

    # y_train is (720, 128, 128, 1) since there are 720 masks of 128x128 pixels, where each pixel has a label [0, 1, 2, 3]
    # We need to one-hot encode the labels; y_train must be (720, 128, 128, 4) for 4 classes

    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    print(train_masks_cat.shape)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes)) # Why reshape?
    print(y_train_cat.shape)


    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25]))

    # Focal loss: works great for class imbalance
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # Want to track intersection over union metric, which is a standard metric for semantic segmentation
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # Standard U-Net architecture

    



if __name__ == "__main__":
    main()