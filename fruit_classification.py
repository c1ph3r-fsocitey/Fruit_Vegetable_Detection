import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import PIL
import tensorflow as tf

#%%

train_dir = pathlib.Path("/home/c1ph3r/Machine_Vision_Projects/Strawberry_detection/fruits-360/Training")
test_dir = pathlib.Path("/home/c1ph3r/Machine_Vision_Projects/Strawberry_detection/fruits-360/Test")

image_count = len(list(train_dir.glob("*/*.jpg")))
print(image_count)

#%%

fruits = list(train_dir.glob("Strawberry/*.jpg"))
plt.figure(figsize=(10, 10))
for i in range(9):
        plt.subplot(3, 3, i+1)
        img = PIL.Image.open(str(fruits[i]))
        plt.imshow(img)
        plt.axis("off")
plt.show()
#%%

batch_size = 32
img_height = 100
img_width = 100

#%%

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

#%%

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
#%%
class_names = train_ds.class_names
num_class = len(class_names)

for images, labels in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()
    
#%%

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#%%
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
])


#%%

preprocess_input = tf.keras.applications.resnet.preprocess_input

#%%
base_model = tf.keras.applications.resnet.ResNet50(
    input_shape = (img_height, img_width, 3),
    include_top = False,
    weights = "imagenet"
)

#%%

base_model.trainable = False

#%%
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(num_class)

#%%

inputs = tf.keras.Input(shape=(100, 100, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs = inputs, outputs=outputs)

#%%

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

#%%

print(model.summary())

#%%

model.evaluate(val_ds)

#%%

epochs = 10
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data = val_ds
)

#%%

print(model.summary())