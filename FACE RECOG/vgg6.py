import tensorflow as tf
from tensorflow import keras
import os
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.layers import Flatten,Dense,Activation
from keras.models import Model
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#classification into trainin and testing
train_human_dir1 = os.path.join("C:\\Users\\\mukho\\\Desktop\\FACE RECOG\\Datasets\\Train\\Ashok")
test_human_dir1=os.path.join("C:\\Users\\mukho\\Desktop\\FACE RECOG\\Datasets\\Test\\Ashok")

train_human_dir2 = os.path.join("C:\\Users\\mukho\\Desktop\\FACE RECOG\\Datasets\\Train\\Ayush")
test_human_dir2=os.path.join("C:\\Users\\mukho\\Desktop\\FACE RECOG\\Datasets\\Test\\Ayush")

train_human_dir3 = os.path.join("C:\\Users\\mukho\\Desktop\\FACE RECOG\\Datasets\\Train\\Ivy")
test_human_dir3=os.path.join("C:\\Users\\mukho\\Desktop\\FACE RECOG\\Datasets\\Test\\Ivy")

print('total training human images:', len(os.listdir(train_human_dir1)))
print('total testing human images:', len(os.listdir(test_human_dir1)))

print('total training human images:', len(os.listdir(train_human_dir2)))
print('total testing human images:', len(os.listdir(test_human_dir2)))

print('total training human images:', len(os.listdir(train_human_dir3)))
print('total testing human images:', len(os.listdir(test_human_dir3)))

#building a mode
vgg=VGG16(include_top=False,weights="imagenet",input_tensor=None,input_shape=(224,224,3))

for Layer in vgg.layers:
    Layer.trainable=False


x=Flatten()(vgg.output)
prediction1=Dense(1024,activation="relu")(x)
prediction=Dense(3,activation="softmax")(prediction1)
model =Model(inputs=vgg.input,outputs=prediction)
model.summary()
model.compile(optimizer=tf.optimizers.Adam(),loss="categorical_crossentropy",metrics="accuracy")



# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode="nearest")

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        "C:\\Users\\mukho\\Desktop\\FACE RECOG\\Datasets\\Train",  # This is the source directory for training images
        target_size=(224,224),  # All images will be resized to 150x150
        batch_size=20,
        class_mode='categorical')


test_data=ImageDataGenerator(rescale=1./255,
)

test_generator=test_data.flow_from_directory(
    "C:\\Users\\mukho\\Desktop\\FACE RECOG\\Datasets\\Test",
    target_size=(224,224),
    batch_size=20,
    class_mode="categorical"
)


history=model.fit(train_generator,
    steps_per_epoch=5,
    epochs=5,
    verbose=1,
    validation_data=test_generator,
    validation_steps=8
)



model.save('model_face.h5')



# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
