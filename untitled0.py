from efficientnet.tfkeras import EfficientNetB0
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from keras.utils import to_categorical

baseModel = EfficientNetB0(weights="imagenet")
#print(baseModel.summary())

baseModel = EfficientNetB0(weights="imagenet",include_top=False)
#print(baseModel.summary())

dropout_rate = 0.2
batch_size = 48
width = 150
height = 150
epochs = 20

model = models.Sequential()
model.add(baseModel)
#model.add(layers.Flatten(name="flatten"))
model.add(layers.GlobalMaxPooling2D(name="gmp"))
print(model.summary())

if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate,name="dropout_out"))

model.add(layers.Dense(100,activation="softmax",name="fc_out"))

baseModel.trainable = False

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#type(x_test)
#
##print(x_train.shape,x_test.shape)
NUM_TRAIN = len(x_train)
NUM_TEST = len(x_test)
##print(NUM_TEST,NUM_TRAIN)
#train_datagen = ImageDataGenerator(
#    rescale=1.0 / 255,
#    rotation_range=40,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    fill_mode="nearest"
#)
#
#test_datagen = ImageDataGenerator(rescale=1.0 / 255)
#
#train_generator = train_datagen.flow_from_dataframe(
#    x_train,
#    # All images will be resized to target height and width.
#    target_size=(height, width),
#    batch_size=batch_size,
#    class_mode="categorical",
#)
#
#validation_generator = test_datagen.flow_from_directory(
#    './cifar100',
#    target_size=(height, width),
#    batch_size=batch_size,
#    class_mode="categorical",
#)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=["acc"],
)

history = model.fit(
#    train_generator,
    steps_per_epoch=NUM_TRAIN // batch_size,
    epochs=epochs,
#    validation_data=validation_generator,
    validation_steps=NUM_TEST // batch_size,
    verbose=1,
    use_multiprocessing=True,
    workers=4,
)

plot_model(baseModel,to_file="baseModel.png",show_shapes=True)
Image(filename="baseModel.png")

baseModel.trainable = True

set_trainable = False
for layer in baseModel.layers:
    if layer.name == 'multiply_16':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False












