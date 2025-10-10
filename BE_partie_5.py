# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:23:05 2025

@author: ML
"""
import pandas as pd
import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import splitfolders
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import model_from_json
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image, ImageOps
import cv2


model_architecture = 'model_flower.json'
model_weights = 'model_flower.weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
model.summary()

base_model=model.layers[0]


# Example (visualiser le filtre numéro 2 appris par le modèle VGG16 )

# Récupérer les filtres et les poids de la 1ère couche
filters, biases, *is_anything_else_being_returned = base_model.layers[1].get_weights()
# normaliser les filtres sur [0 , 1]
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# Prendre le deuxième filtre de cette couche (qui contient 3 sous-filtres pour les 3 canaux R,
# V et B) et visualiser ces 3 sous-filtres
f=filters[:,:,:,1]
plt.subplot(1,3,1)
plt.imshow(f[:, :, 0], cmap='gray')
plt.subplot(1,3,2)
plt.imshow(f[:, :, 1], cmap='gray')
plt.subplot(1,3,3)
plt.imshow(f[:, :, 2], cmap='gray')


# Example 2

# Définir un modèle intermédiaire contenant les 2 premières couches du modèle initial
from keras.models import Model
file= 'F:/Université de Toulouse/Je suis en France/Cours/S8/BE Apprentissage automatique (Machine learning)/Projet/flower_output/test/flower_05/image_05168.jpg'
image=Image.open(file)
image = np.asarray(image)
image=image.astype('float32')
image=cv2.resize(image,(224,224))
image /= 255
image = image.reshape([-1,224,224,3])
inter_model = Model(inputs=base_model.inputs, outputs=base_model.layers[1].output)
inter_model.summary()
feature_maps = inter_model.predict(image)
plt.imshow(feature_maps[0,:,:,1]) #♀ Si on change le 1 le filtre change


proba=model.predict(image)
pred_classe=np.argmax(proba, 1)
print("classe =", pred_classe)






