# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:10:33 2025

@author: ML
"""
# -*- coding: utf-8 -*-

import pandas as pd
import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import splitfolders

from tensorflow.keras.applications import VGG16

#1. Découpage des données

splitfolders.ratio(
    r"F:\Université de Toulouse\Je suis en France\Cours\S8\BE Apprentissage automatique (Machine learning)\Projet\Donnees",
    output= r"F:\Université de Toulouse\Je suis en France\Cours\S8\BE Apprentissage automatique (Machine learning)\Projet\flower_output",
    seed=1337,
    ratio=(.8, 0.1, 0.1)
)

# 2. Création des Data Generators
train_dir = r"F:\Université de Toulouse\Je suis en France\Cours\S8\BE Apprentissage automatique (Machine learning)\Projet\flower_output\train"
val_dir = r"F:\Université de Toulouse\Je suis en France\Cours\S8\BE Apprentissage automatique (Machine learning)\Projet\flower_output\val"
test_dir = r"F:\Université de Toulouse\Je suis en France\Cours\S8\BE Apprentissage automatique (Machine learning)\Projet\flower_output\test"


test_rescal=ImageDataGenerator(rescale=1./255)
test_gen=test_rescal.flow_from_directory(
    directory=test_dir,                       
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
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=2,
    
)
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    batch_size=32,
    seed=2,
    
)

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
    epochs=20
)

#evaluate the model
test_loss, test_acc = model.evaluate(test_gen)
print('Test accuracy:', test_acc)


# Analyse des résultats

import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
y_pred_prob=model.predict(test_gen)
y_pred=np.argmax(y_pred_prob, 1)
labels = test_gen.classes
mat=confusion_matrix(labels, y_pred)
print(mat)
print(classification_report(labels, y_pred))
sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues,
fmt='.0f')
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.show()


# Sauvegarder le modèle pour le réutiliser plus tard

model_json = model.to_json()
with open('model_flower.json', 'w') as json_file:
    json_file.write(model_json)
    model.save_weights('model_flower.weights.h5')





















