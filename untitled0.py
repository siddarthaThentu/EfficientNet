from efficientnet.tfkeras import EfficientNetB0
#from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import cv2
from keras import backend as K

baseModel = EfficientNetB0(weights="imagenet")
#print(baseModel.summary())

dropout_rate = 0.2
batch_size = 64
epochs = 25

baseModel = EfficientNetB0(weights="imagenet",include_top=False)
#print(baseModel.summary())

model = models.Sequential()
model.add(baseModel)
model.add(layers.GlobalMaxPooling2D(name="gmp"))

if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate,name="dropout_out"))

model.add(layers.Dense(100,activation="softmax",name="fc_out"))

#print(model.summary())
#baseModel.trainable = False

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
y_train = to_categorical(y_train,100)
y_test = to_categorical(y_test,100)
print(type(x_train))
#x_train = resize(x_train,(40,40,3))

img_rows = 227
img_cols = 227

if K.image_data_format() == 'th':
        x_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in x_train[:len(x_train),:,:,:]])
        x_test = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in x_test[:len(x_test),:,:,:]])
else:
        x_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in x_train[:len(x_train),:,:,:]])
        x_test = np.array([cv2.resize(img, (img_rows,img_cols)) for img in x_test[:len(x_test),:,:,:]])

print(x_train.shape)

#train_datagen = ImageDataGenerator(
#	  rescale=1.0 / 255
#    rotation_range=40,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    fill_mode="nearest"
#)

#train_datagen.fit(x_train)
                  
#validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

#validation_datagen.fit(x_test)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(lr=2e-4),
    metrics=["acc"]
)

#hist =  model.fit_generator(
 #          train_datagen.flow(x_train, y_train, batch_size=batch_size),
  #         steps_per_epoch=len(x_train) // batch_size, 
  #         epochs=epochs,
  #         validation_data=validation_datagen.flow(x_test,y_test,batch_size=batch_size),
  #         validation_steps=len(x_test) // batch_size,
  #         verbose=1)
#           #use_multiprocessing=True,
#           #orkers=4)
#	

hist = model.fit(x_train, y_train, batch_size=32, epochs=50,validation_split=0.2)

#plot_model(baseModel,to_file="baseModel.png",show_shapes=True)
#Image(filename="baseModel.png")

model.save("efficientSample.h5")

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train','Validation'],loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'],loc='upper right')
plt.show()

#model.evaluate(x_test,y_test)[1]
#baseModel.trainable = True
#
##set_trainable = False
##for layer in baseModel.layers:
##    if layer.name == 'multiply_16':
##        set_trainable = True
##    if set_trainable:
##        layer.trainable = True
##    else:
##        layer.trainable = False
##
#
#
#
#
#






