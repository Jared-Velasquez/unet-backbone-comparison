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
optim_unet=keras.optimizers.Adam(learning_rate) # Faster learning rates may not converge
optim_resnet=keras.optimizers.Adam(learning_rate)
optim_vgg=keras.optimizers.Adam(learning_rate)

BACKBONE_RESNET50 = 'resnet50'
BACKBONE_VGG19 = 'vgg19'

def main():
    """Code adapted from Sreenivas Bhattiprolu's Python for Microscopists tutorial on semantic segmentation.
    https://github.com/bnsreenu/python_for_microscopists
    """

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

    # Do same for test data
    test_masks_cat = to_categorical(y_test, num_classes=n_classes) # One-hot encode the labels
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))


    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25]))

    # Focal loss: works great for class imbalance
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # Want to track intersection over union metric, which is a standard metric for semantic segmentation
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # Standard U-Net architecture
    unet_model = sm.Unet(classes=n_classes, activation=activation)
    # U-net with a resnet50 backbone
    unet_resnet_backbone_model = sm.Unet(BACKBONE_RESNET50, classes=n_classes, activation=activation, encoder_weights='imagenet')
    preprocess_resnet_input = sm.get_preprocessing(BACKBONE_RESNET50)
    X_train_resnet = preprocess_resnet_input(X_train)
    X_test_resnet = preprocess_resnet_input(X_test)
    # U-net with a vgg19 backbone
    unet_vgg_backbone_model = sm.Unet(BACKBONE_VGG19, classes=n_classes, activation=activation, encoder_weights='imagenet')
    preprocess_vgg_input = sm.get_preprocessing(BACKBONE_VGG19)
    X_train_vgg = preprocess_vgg_input(X_train)
    X_test_vgg = preprocess_vgg_input(X_test)

    ######################################################################
    print("Training unet model...")
    # U-net model
    unet_model.compile(optimizer=optim_unet, loss=total_loss, metrics=metrics)

    print(unet_model.summary())

    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        ]

    # Fit model
    history1=unet_model.fit(
        X_train,
        y_train_cat,
        batch_size=8,
        epochs=50,
        verbose=1,
        validation_data=(X_test, y_test_cat),
        callbacks=callbacks,
    )

    unet_model.save('unet_50epochs.hdf5')

    # plot training and validation accuracy and loss at each epoch
    loss_unet = history1.history['loss']
    val_loss_unet = history1.history['val_loss']
    epochs = range(1, len(loss_unet) + 1)

    plt.figure()
    plt.plot(epochs, loss_unet, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_unet, 'b', label='Validation loss')
    plt.title('Training and validation loss for U-net model')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('unet_loss.png')

    # plot iou and f1 score at each epoch
    iou = history1.history['iou_score']
    f1 = history1.history['f1-score']
    epochs = range(1, len(iou) + 1)

    plt.figure()
    plt.plot(epochs, iou, 'bo', label='Training IOU')
    plt.plot(epochs, f1, 'b', label='Training F1')
    plt.title('Training IOU and F1 score for U-net model')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.savefig('unet_iou_f1_score.png')

    ######################################################################
    print("Training unet model with resnet50 backbone...")
    # U-net model with resnet50 backbone
    unet_resnet_backbone_model.compile(optimizer=optim_resnet, loss=total_loss, metrics=metrics)

    print(unet_resnet_backbone_model.summary())

    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        ]
    
    # Fit model
    history2=unet_resnet_backbone_model.fit(
        X_train_resnet,
        y_train_cat,
        batch_size=8,
        epochs=50,
        verbose=1,
        validation_data=(X_test_resnet, y_test_cat),
        callbacks=callbacks,
    )

    unet_resnet_backbone_model.save('resnet50_backbone_50epochs.hdf5')

    # plot training and validation accuracy and loss at each epoch
    loss_resnet = history2.history['loss']
    val_loss_resnet = history2.history['val_loss']
    epochs = range(1, len(loss_resnet) + 1)

    plt.figure()
    plt.plot(epochs, loss_resnet, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_resnet, 'b', label='Validation loss')
    plt.title('Training and validation loss for U-net model with resnet50 backbone')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('resnet50_backbone_loss.png')

    # plot iou and f1 score at each epoch
    iou_resnet = history2.history['iou_score']
    f1_resnet = history2.history['f1-score']
    epochs = range(1, len(iou_resnet) + 1)
    
    plt.figure()
    plt.plot(epochs, iou_resnet, 'bo', label='Training IOU')
    plt.plot(epochs, f1_resnet, 'b', label='Training F1')
    plt.title('Training IOU and F1 score for U-net model with resnet50 backbone')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.savefig('resnet50_backbone_iou_f1_score.png')

    ######################################################################
    print("Training unet model with vgg19 backbone...")
    # U-net model with vgg19 backbone
    unet_vgg_backbone_model.compile(optimizer=optim_vgg, loss=total_loss, metrics=metrics)

    print(unet_vgg_backbone_model.summary())

    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        ]
    
    # Fit model
    history3=unet_vgg_backbone_model.fit(
        X_train_vgg,
        y_train_cat,
        batch_size=8,
        epochs=50,
        verbose=1,
        validation_data=(X_test_vgg, y_test_cat),
        callbacks=callbacks,
    )

    unet_vgg_backbone_model.save('vgg19_backbone_50epochs.hdf5')

    # plot training and validation accuracy and loss at each epoch
    loss_vgg = history3.history['loss']
    val_loss_vgg = history3.history['val_loss']
    epochs = range(1, len(loss_vgg) + 1)

    plt.figure()
    plt.plot(epochs, loss_vgg, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_vgg, 'b', label='Validation loss')
    plt.title('Training and validation loss for U-net model with vgg19 backbone')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('vgg19_backbone_loss.png')

    # plot iou and f1 score at each epoch
    iou_vgg = history3.history['iou_score']
    f1_vgg = history3.history['f1-score']
    epochs = range(1, len(iou_vgg) + 1)

    plt.figure()
    plt.plot(epochs, iou_vgg, 'bo', label='Training IOU')
    plt.plot(epochs, f1_vgg, 'b', label='Training F1')
    
    plt.title('Training IOU and F1 score for U-net model with vgg19 backbone')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.savefig('vgg19_backbone_iou_f1_score.png')

    print("Training complete.")

    # IOU Score and F1 Score for each model:

    # U-net model
    print("U-net model:")
    print("IOU Score: ", iou[-1])
    print("F1 Score: ", f1[-1])

    # U-net model with resnet50 backbone
    print("U-net model with resnet50 backbone:")
    print("IOU Score: ", iou_resnet[-1])
    print("F1 Score: ", f1_resnet[-1])

    # U-net model with vgg19 backbone
    print("U-net model with vgg19 backbone:")
    print("IOU Score: ", iou_vgg[-1])
    print("F1 Score: ", f1_vgg[-1])

    
    



if __name__ == "__main__":
    main()