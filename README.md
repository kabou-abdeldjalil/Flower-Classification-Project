🌼 Projet : Classification des Fleurs par Deep Learning

Réalisé par : KABOU Abdeldjalil

Présentation :

    Ce projet a pour objectif de concevoir un système de reconnaissance automatique d’espèces de fleurs à partir d’images, en exploitant la puissance des réseaux de neurones convolutionnels (CNN) et du transfert de connaissances.

Les principales étapes réalisées sont :

    - Découpage des données en ensembles d'entraînement, de validation et de test ;
    - Construction d’un modèle CNN pour la classification d’images ;
    - Amélioration des performances grâce au transfert de connaissances avec le modèle pré-entraîné VGG16 ;
    - Évaluation du modèle à l’aide de métriques classiques : précision, matrice de confusion, etc. ;
    - Développement d’une application web interactive avec Streamlit, permettant de tester le modèle en téléversant une image.

Modèles utilisés :

Deux modèles principaux ont été développés et comparés :

    1) Modèle CNN personnalisé  
        - Construit à partir de zéro.
        - Composé de plusieurs couches convolutionnelles, de pooling et de couches denses.
        - Sert de base pour évaluer les performances du réseau sans transfert d’apprentissage.
    2) Modèle basé sur VGG16 (Transfert de Connaissances)
        - Utilise VGG16 pré-entraîné sur ImageNet comme extracteur de caractéristiques.
        - Des couches personnalisées sont ajoutées en sortie pour adapter la classification au dataset de fleurs.
        - Permet d’obtenir de meilleures performances grâce aux représentations déjà apprises.
        
    Les modèles ont été évalués sur le jeu de test et comparés en termes de précision et de généralisation.

Modèle pré-entraîné :

Le fichier .h5 contenant le modèle entraîné est disponible en téléchargement ici :

https://drive.google.com/file/d/1fOBjp2VSxxTAvIbyVGY0ET1vtie-8LWK/view?usp=sharing
