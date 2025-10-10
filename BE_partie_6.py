# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:10:09 2025

@author: ML
"""

import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import model_from_json

nom_classe= ['pink, primrose', 'hard-leaved, pocket, orchid', 
             'canterbury, bells', 'sweet, pea', 'english, marigold', 
             'tiger, lily', 'moon, orchid', 'bird, of, paradise', 
             'monkshood', 'globe, thistle', 'snapdragon', "colt's, foot", 
             'king, protea', 'spear, thistle', 'yellow, iris', 
             'globe-flower', 'purple, coneflower', 'peruvian, lily', 
             'balloon, flower', 'giant, white, arum, lily', 'fire, lily', 
             'pincushion, flower', 'fritillary', 'red, ginger', 'grape, hyacinth', 
             'corn, poppy', 'prince, of, wales, feathers', 'stemless, gentian', 
             'artichoke', 'sweet, william', 'carnation', 'garden, phlox', 
             'love, in, the, mist', 'mexican, aster', 'alpine, sea, holly', 
             'ruby-lipped, cattleya', 'cape, flower', 'great, masterwort', 
             'siam, tulip', 'lenten, rose', 'barbeton, daisy', 'daffodil', 
             'sword, lily', 'poinsettia', 'bolero, deep, blue', 'wallflower', 
             'marigold', 'buttercup', 'oxeye, daisy', 'common, dandelion', 
             'petunia', 'wild, pansy', 'primula', 'sunflower', 'pelargonium', 
             'bishop, of, llandaff', 'gaura', 'geranium', 'orange, dahlia', 
             'pink-yellow, dahlia', 'cautleya, spicata', 'japanese, anemone', 
             'black-eyed, susan', 'silverbush', 'californian, poppy', 'osteospermum', 
             'spring, crocus', 'bearded, iris', 'windflower', 'tree, poppy', 'gazania', 
             'azalea', 'water, lily', 'rose', 'thorn, apple', 'morning, glory', 'passion, flower', 
             'lotus, lotus', 'toad, lily', 'anthurium', 'frangipani', 'clematis', 
             'hibiscus', 'columbine', 'desert-rose', 'tree, mallow', 'magnolia', 
             'cyclamen', 'watercress', 'canna, lily', 'hippeastrum', 'bee, balm', 
             'ball, moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 
             'mexican, petunia', 'bromelia', 'blanket, flower', 'trumpet, creeper', 'blackberry, lily']


@st.cache_data
def load_model():
    #model_architecture = "F:/Université de Toulouse/Je suis en France/Cours/S8/BE Apprentissage automatique (Machine learning)/Projet/Etapes de projet/model_flower.json"
    #model_weights = "F:/Université de Toulouse/Je suis en France/Cours/S8/BE Apprentissage automatique (Machine learning)/Projet/Etapes de projet/model_flower.weights.h5"
    #model_architecture = 'model_flower.json'
    #model_weights = 'model_flower.weights.h5'
    model_architecture = r"F:/Université de Toulouse/Je suis en France/Cours/S8/BE Apprentissage automatique (Machine learning)/Projet/Etapes de projet/model_flower.json"
    model_weights = r"F:/Université de Toulouse/Je suis en France/Cours/S8/BE Apprentissage automatique (Machine learning)/Projet/Etapes de projet/model_flower.weights.h5"

    model = model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)  
    return model
    
    # from tensorflow.keras.models import load_model
    # model = load_model("F:/Université de Toulouse/Je suis en France/Cours/S8/BE Apprentissage automatique (Machine learning)/Projet/Etapes de projet/model_flower.keras")


with st.spinner('Model is being loaded..'):
  model=load_model()
 
st.write("""
         # Classification des fleurs         """
         )
 
file = st.file_uploader("Upload the image to be classified", type=["jpg", "png", "jpeg"])
#st.set_option('deprecation.showfileUploaderEncoding', False)
 
def upload_predict(image, model):  
        image = np.asarray(image)
        image=image.astype('float32')
        image=cv2.resize(image,(224,224))
        image /= 255
        image = image.reshape([-1,224,224,3])
        prediction = model.predict(image)
        pred_class=np.argmax(prediction, axis=1)       
        return pred_class, prediction
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    pred_class, prediction = upload_predict(image, model)
    st.write("The image is classified as",nom_classe[pred_class[0]])
    st.write("The probability is", prediction[0][pred_class[0]])


# streamlit run "F:/Université de Toulouse/Je suis en France/Cours/S8/BE Apprentissage automatique (Machine learning)/Projet/Etapes de projet/BE_partie_6.py"
