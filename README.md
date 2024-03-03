# Style Transfer amb VGG19

VGG19 és un model de xarxa neuronal convolucional (CNN) desenvolupat per l'equip de Visual Geometry Group de la Universitat d'Oxford i es va presentar al concurs ILSVRC de 2014.<br><br>El model ha mostrat molt bons resultats quan s'entrena amb milions d'imatges del dataset ImageNet. Aquest model és molt utilitzat en tasques de visió per computador, com la detecció d'objectes i la **transferència d'estil**.<br><br>La transferència d'estil és una tècnica que combina l'estil visual d'una imatge (l'estil) amb el contingut d'una altra imatge, creant una imatge resultant que manté el contingut original però amb l'aspecte estètic de la imatge d'estil.<br><br>

## Estructura VGG19

El model pren aquest nom per la seva arquitectura profunda de 19 capes (16 capes convolucionals, 3 capes *fully-connected*, capes de *max-pooling* i funció d'activació ReLU). Les capes convolucionals es troben agrupades en 5 blocs convolucionals ( ***fig 1**: la columna E es correspon amb VGG19* ).<br><br>

![architectures de CNNs](CNN_architectures.png)
***Fig 1***: *Taula comparativa d'arquitectures de CNNs extreta del paper 'Very Deep Convolutional Networks for Large-Scale Image Recognition' (https://doi.org/10.48550/arXiv.1409.1556)* <br><br>

## Transfer Learning


