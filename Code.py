from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt

# Set the batch size and number of training steps per epoch
batch_size = 32
train_steps_per_epoch = 10

# Data augmentation setup
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Directories for training and validation datasets
train_dir = 'Path of your train folder'
test_dir = 'Path of your test folder'

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Initialize and compile the model
base_model = VGG16(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=10,
    validation_data=validation_generator
)

# Save the trained model
model.save('6_class_vgg16_model.h5')

# Load the model for prediction
model = load_model('6_class_vgg16_model.h5')
class_labels = ['Missing Capacitors', 'No Missing Components', 'Missing Processor', 'Missing USB', 'Missing Rails']

# Prompt the user for the directory of test images
test_images_dir = input("Please enter the directory of the test images to predict: ")

# Predict classes of new images
for img_name in os.listdir(test_images_dir):
    img_path = os.path.join(test_images_dir, img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]

    print(f"Image: {img_name} - Predicted class label: {predicted_class_label}")
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{img_name} - Predicted: {predicted_class_label}")
    plt.show()
