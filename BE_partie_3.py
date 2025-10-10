# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:51:50 2025

@author: ML
"""

# -*- coding: utf-8 -*-

import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import splitfolders

from tensorflow.keras.applications import VGG16

# #1. Découpage des données
# splitfolders.ratio(
#     r"C:\Users\ML\Desktop\Fleurs/Donnees",
#     output= r"C:\Users\ML\Desktop\Fleurs\flower_output",
#     seed=1337,
#     ratio=(.8, 0.1, 0.1)
# )

# 2. Création des Data Generators
train_dir = r"C:/Users/ML/Desktop/Fleurs/flower_output/train"
val_dir = r"C:/Users/ML/Desktop/Fleurs/flower_output/val"
test_dir = r"C:/Users/ML/Desktop/Fleurs/flower_output/test"


test_rescal=ImageDataGenerator(rescale=1./255)
test_gen=test_rescal.flow_from_directory(
    r"C:/Users/ML/Desktop/Fleurs/flower_output/test",                       
    target_size=(224, 224),          
    color_mode='rgb',                 
    class_mode='categorical',          
    batch_size=32,                  
    shuffle=False,
    seed=None,                      
  )
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32,
    shuffle=True,
    seed=2,
    class_mode='categorical'
)
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(224, 224),
    color_mode='rgb',
    shuffle=True,
    batch_size=32,
    seed=2,
    class_mode='categorical'
)

# Transfer learning (Réseaux pré-entrainés avec le réglage de certains paramètres (fine tuning) 

num_classes=102

base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# Les paramètres des 17 premières couches sont gelés
for layer in base_model.layers[:17]:
    layer.trainable = False
# Pour voir quelles sont les couches entraînables
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.trainable)
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Entraînement
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

#evaluate the model
test_loss, test_acc = model.evaluate(test_gen)
print('Test accuracy:', test_acc)






