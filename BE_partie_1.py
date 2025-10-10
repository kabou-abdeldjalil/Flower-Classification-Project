import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import splitfolders

# 1. Découpage des données

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
    target_size=(180, 180),          
    color_mode='rgb',                 
    class_mode='categorical',          
    batch_size=32,                  
    shuffle=False,
    #Pourquoi on ne met pas shuffle=True pour le test ?
    # Parce qu’on veut comparer précisément chaque prédiction à sa vraie classe.
    seed=None,                      
  )
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(180, 180),
    batch_size=32,
    class_mode='categorical'
)
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(180, 180),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 3. Modèle CNN simple
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compilation du modèle
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






