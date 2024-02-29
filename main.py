# import cv2
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization as batchNorm
from tensorflow.keras.layers import Activation, Add, Dense
from tensorflow.keras.layers import AveragePooling2D, Flatten


def getLabels(csv_with_labels, directory_with_images):
    # read csv to pandas dataframe
    df = pd.read_csv(csv_with_labels)

    # get the filenames
    filenames = df['Image'].tolist()

    # get the labels
    labels = df['Class'].tolist()

    # create dict from filenames
    dictTrain = dict.fromkeys(filenames)

    # create filenames - labels pairs
    for key, value in zip(dictTrain.keys(), labels):
        dictTrain[key] = value

    # return the labels ordered by a sorted listdir function
    image_labels = []
    for filename in sorted(os.listdir(directory_with_images)):
        image_labels.append(dictTrain[filename])

    # return the ordered image labels
    return image_labels


pathTrain = "gic-unibuc-dl-2023/train_images"
pathValid = "gic-unibuc-dl-2023/val_images"

pathTrain_labels = "gic-unibuc-dl-2023/train.csv"
pathValid_labels = "gic-unibuc-dl-2023/val.csv"

# get labels for training
train_labels = getLabels(pathTrain_labels, pathTrain)

# get labels for validation
valid_labels = getLabels(pathValid_labels, pathValid)

# set batch size
size_batches = 64

# create datasets from train and validation directories

train_ds = tf.keras.utils.image_dataset_from_directory(
    pathTrain,
    labels=train_labels,
    label_mode="int",
    color_mode='rgb',
    batch_size=size_batches,
    image_size=(64, 64),
    shuffle=True
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    pathValid,
    labels=valid_labels,
    label_mode="int",
    color_mode='rgb',
    batch_size=size_batches,
    image_size=(64, 64),
    shuffle=True
)


# create a normalize function to map the datasets to using the per_image_standardization
def normalize(image, label):
    norm_image = tf.image.per_image_standardization(image)
    return norm_image, label


train_ds = train_ds.map(normalize)
valid_ds = valid_ds.map(normalize)


# resnet functions

#initial convolution, per the paper
def initial_conv_block(inputs):
    conv1 = Activation('relu')(batchNorm()(Conv2D(16, (3, 3), strides=(1, 1), padding='same')(inputs)))
    return conv1

#the main blocks used, conrpised of a single projection block, followed by 'blocks' number of identity blocks
def resnet_block(x, no_filters, blocks, strides, multiplier=2):
    proj = proj_block(x, no_filters, strides=strides, multiplier=multiplier)
    for i in range(blocks):
        proj = ident_block(proj, no_filters, multiplier)
    return proj

#identity block, shortcut followed by 1x1 conv 3x3 conv and 1x1 conv (per the paper) and adding the shortcut
def ident_block(x, no_filters, multiplier=2):
    shortcut = x
    conv1 = Activation('relu')(batchNorm()(Conv2D(no_filters, (1, 1), strides=(1, 1))(x)))
    conv2 = Activation('relu')(batchNorm()(Conv2D(no_filters, (3, 3), strides=(1, 1), padding="same")(conv1)))
    conv3 = batchNorm()(Conv2D(no_filters * multiplier, (1, 1), strides=(1, 1))(conv2))
    added = Add()([conv3, shortcut])
    reluAdd = Activation('relu')(added)
    return reluAdd

#projection block, same as identity, but with a 2D conv applied to the input to be added
def proj_block(x, no_filters, strides, multiplier=2):
    shortcut = Conv2D(no_filters * multiplier, (1, 1), strides=strides)(x)
    conv1 = Activation('relu')(batchNorm()(Conv2D(no_filters, (1, 1), strides=(1, 1))(x)))
    conv2 = Activation('relu')(batchNorm()(Conv2D(no_filters, (3, 3), strides=strides, padding='same')(conv1)))
    conv3 = batchNorm()(Conv2D(no_filters * multiplier, (1, 1), strides=(1, 1))(conv2))
    added = Add()([shortcut, conv3])
    reluAdd = Activation('relu')(added)
    return reluAdd

#output layer, norm, avg pool and flatten to pass through softmax
def output_layers(x, num_classes):
    batchNormed = Activation('relu')(batchNorm()(x))
    pooled = AveragePooling2D(pool_size=8)(batchNormed)
    flattened = Flatten()(pooled)
    # dense1 = Dense(258, activation='relu')(flattened))
    # dense2 = Dense(258, activation='relu')(flattened))
    # model.add(layers.Dense(260, activation='leaky_relu'))
    outputs = Dense(num_classes, activation='softmax')(flattened)
    return outputs

#create the model, 16,64,128 resnet block filters
def resnet_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    processed = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = initial_conv_block(processed)
    # x = resnet_block(x, 32, 2, strides=(1, 1), multiplier=4)
    #x = resnet_block(x, 16, 4, strides=(1, 1), multiplier=4)
    x = resnet_block(x, 16, 2, strides=(1, 1), multiplier=4)
    x = resnet_block(x, 64, 6, strides=(2, 2))
    x = resnet_block(x, 128, 10, strides=(2, 2))
    outputs = output_layers(x, num_classes)
    model = Model(inputs, outputs)
    return model


model = resnet_model((64, 64, 3), 100)

# Print model summary
model.summary()

#checkpoint for restoring the best weights
mcheck = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_custom_model.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=True)

#early stopping to avoid validation performance loss
earlyS = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    restore_best_weights=False,
    patience=8
)

#reduce learning rate to avoid overshooting minima
ReduceLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5,
    patience=5, verbose=True
)

#define optimizer
AdamW = tf.keras.optimizers.AdamW()

#compile
model.compile(optimizer=AdamW,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, epochs=50, validation_data=valid_ds, callbacks=[mcheck, earlyS, ReduceLR])

#create another model for loading the weights
bestModel = resnet_model((64, 64, 3), 100)

#load best weights
bestModel.load_weights('best_custom_model.h5')

model.save("bestCustomModel.h5")

# for predicting
'''
predictions = []
for img in imgs:
  prediction = model.predict(img)
  predictions.append(prediction)

prediction_labels = tf.argmax(predictions, axis = 2)
'''
