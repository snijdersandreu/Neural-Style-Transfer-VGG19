## Taula de continguts
* [Descripció del projecte](#neural-style-transfer-amb-vgg19)
* [Estructura de VGG19](#1-estructura-de-vgg19)
* [Transferència de l'estil](#2-transferència-de-lestil)
* [*Gram Matrix*](#3-gram-matrix)
  * [Perquè utilitzem la *Gram Matrix*?](#31-perquè-utilitzem-la-gram-matrix)
* [Funció de pèrdues](#4-funció-de-pèrdues)
  * [*Loss* de contingut](#41-loss-de-contingut)
  * [*Loss* d'estil](#42-loss-destil)  
  * [Respecte quines capes optimitzem?](#43-respecte-quines-capes-optimitzem)
* [Interfície d'usuari](#5-interfície-dusuari)
* [Codi](#6-codi)
* [Documentació](#7-documentació)
___

# Neural Style Transfer amb VGG19

Aquest projecte consisteix en utilitzar un model VGG19 per a la transferència d'estil entre imatges. A més, en aquest projecte intento explicar de manera entenedora conceptes com la ***Matriu Gram*** i el optimitzador ***L-BFGS***.<br><br>VGG19 és un model de **xarxa neuronal convolucional** (CNN) desenvolupat per l'equip de Visual Geometry Group de la Universitat d'Oxford i es va presentar al concurs **ILSVRC** ( *ImageNet Large Scale Visual Recognition Challenge* ) de 2014. Aquest model es va endur el 2n premi per darrere de GoogLeNet.<br><br>El model ha mostrat molt bons resultats quan s'entrena amb milions d'imatges del dataset **ImageNet**. És molt utilitzat en tasques de visió per computador, com la detecció i classificació d'objectes i la **transferència d'estil**.<br><br>La transferència d'estil és una tècnica que combina l'estil visual d'una imatge amb el contingut d'una altra imatge, creant una imatge resultant que manté el contingut original però amb l'aspecte estètic de la imatge d'estil.

## 1. Estructura de VGG19

El model pren aquest nom per la seva arquitectura profunda de 19 capes (**16** capes **convolucionals**, **3** capes ***fully-connected***, capes de ***max-pooling*** i funció d'activació **ReLU**). Les capes convolucionals es troben agrupades en 5 blocs convolucionals ( ***fig 1***: *la columna E es correspon amb VGG19* ).<br><br>Per al nostre projecte només necessitarem l'etapa d'**extracció de característiques**. Per tant, no hi afegirem les últimes 5 capes (*maxpool*, 3x *fully-connected*, *maxpool*).<br><br>Carregarem un conjunt de parametres (***weights***) preentrenats amb el dataset *ImageNet*. A més, congelarem totes les capes per tal d'assegurar-nos que no canviem cap d'aquest paràmetres ja entrenats.<br><br>

<p align="center">
  <img src="CNN_architectures.png" alt="arquitectures de CNNs">
</p>

***fig 1***: *Taula comparativa d'arquitectures de CNNs extreta del paper 'Very Deep Convolutional Networks for Large-Scale Image Recognition' (https://doi.org/10.48550/arXiv.1409.1556)* <br><br>

## 2. Transferència de l'estil

Podem utilitzar VGG19 per extreure característiques que ens permetin representar tant l'estil com el contingut d'una imatge: les **capes inicials** de la xarxa capturen detalls d'estil com **textures i colors**, mentre que les **capes més profundes** codifiquen informació de contingut, com ara **objectes i escenes**.<br><br>La idea és crear una imatge que anomenem ***target*** que sigui una combinació de dues. Podem iniciar la nova imatge com una copia de la imatge contingut. En el procés de transferència d'estil, **minimitzem les diferències** entre les representacions d'estil de la **imatge *target* i la imatge d'estil**, així com entre les representacions de contingut de la **imatge *target* i l'imatge de contingut**. Aixo ho aconseguim ajustant iterativament la imatge *target* mitjançant ***backprop***. A diferència del que es fa en l'entrament de xarxes neuronals, el *backprop* el realitzem per **modificar el input** (imatge *target*) enlloc dels pesos (*weights*). D'aquesta manera creem una nova imatge que manté el contingut original però pren l'estil de l'altre imatge.
___
## 3. *Gram Matrix*

En els models de transferència d'estil s'acostuma a utilitzar una operació matricial que té com a resultat una matriu que s'anomena ***Matriu Gram***.<br><br>Per exemple, si partim d'un conjunt de vectors (diguem $F$) i la seva transposada ($F^T$), la *Matriu Gram* de $F$ seria la multiplicació matricial d'aquestes dues:<br><br>$G = F \times F^T$<br><br>Aquesta matriu resultant codifica l'autocorrelació del conjunt de vectors $F$. És a dir, és una representació de la **correlació que existeix entre els diferents vectors** del conjunt. Això és degut a que cada element de $G$ és el producte escalar de dos vectors de $F$:<br><br>$G_{ij} = F_i \cdot F_j$

### 3.1 Perquè utilitzem la *Gram Matrix*?

Utilitzem aquesta matriu per obtenir una **representació de l'estil i contingut** de cada imatge. Per entendre millor, però, la utilitat d'aquesta representació, cal entendre com l'apliquem en el nostre projecte.<br><br>En el context de transferència d'estil, volem calcular una *Matriu Gram* per algunes capes del model (**una matriu per cadascuna** de les capes seleccionades). Cal recordar que diferents capes captaràn característiques diferents de la imatge: capes **inicials** representeràn **detalls petits** i capes més **profundes** captaràn **característiques més generals**. Per tant, el que farem serà trobar correlacions entre detalls petits de la imatge (**estil**) i també correlacions entre característiques generals (**contingut**).<br><br>El conjunt de vectors que anteriorment hem anomenat $F$ seria el conjunt de mapes de característiques  que genera **una sola capa** del model VGG19. Per calcular una de les Matrius Gram ho fariem de la següent manera:
* $F_n$ seria el conjunt de mapes de característiques resultant de la *n-éssima* capa convolucional amb dimensions $(D, H, W)$ on: $D$ = numero de filtres, $(H)$ = alçada del mapa de característiques i $(W)$ = amplada del mapa de característiques.
* Cada mapa de característiques $F_n$ es transforma en un conjunt de vectors (*matriu*) de dimensions $(D, H \times W)$. Podem dir que en aquest procés *estirem* els mapes de característiques generats per cada filtre perque prenguin forma de vector (de tamany $H \times W$).
* A partir d'aquesta matriu $F_n$ calculem la corresponent *Matriu Gram*: $G_n = F_n \times F_n^T$

Simplificarem conceptualment el significat dels mapes de característiques per explicar com extraiem un representació de l'estil a partir d'aquests:<br><br> *Podem entendre que cada filtre d'una capa s'encarrega de detectar diferents característiques. Per exemple, un filtre pot estar buscant linies diagonals i un altre pot estar buscant zones de color vermell. Els seus respectius mapes de característiques prendran valors alts en les zones on hi hagi linies diagonals i, per l'altre filtre, valors alts en zones on hi hagi color vermell. En el nostre exemple bàsic podriem buscar, mitjançant una **Matriu Gram**, una correlació entre aquests dos mapes. És a dir, podriem trobar si les zones amb liníes diagonals acostumen a ser de color vermell. Aquesta relació de característiques és el que visualment entenem com estil.*
<br>L'anterior exemple redueix molt la complexitat real del funcionament de les CNNs però ens serveix com a métode d'explicació. Els filtres realment no busquen linies o colors, sinó que ***aprenen*, mitjançant *backprop***, filtres que codifiquen la informació de la imatge en una altra dimensionalitat.<br><br>Per resumir, la correlació entre mapes de característiques ens ajuda a comprendre com les textures, colors i patrons es repeteixen o varien dins de la imatge. Així es com definim l'estil visual.

___
## 4. Funció de pèrdues

Una funció de pèrdues (***Loss Function***) mesura l'eficàcia d'un model, on resultats alts indiquen baixa precisió. En la classificació incorrecta d'un objecte, per exemple, obtindriem un valor molt alt.<br><br>En el procés d'entrenament de xarxes neurals, com VGG19, s'utilitza el *backprop* per ajustar els pesos (*weights*) i optimitzar la funció de pèrdues, millorant així el model.<br><br> En transferència d'estil, en canvi, el *backprop* s'utilitza per **ajustar l'imatge *target***, modificant-la per optimitzar la funció de pèrdues i aconseguir l'estil desitjat. Amb aquesta aclaració podem pasar a explicar la funció de pèrdues que utilitzem ja que no l'habitual.<br><br>La funció que utilitzem aqui realment està formada per dues altres. Per mesurar la ***loss*** ($L$) total del nostre model ho fem amb la següent formula:

$L_{total} = W_{content} \times L_{content} \ + \ W_{style} \times L_{style} $

on $W$ són pesos que ens permeten ajustar en quina mesura volem aplicar l'estil i contingut a la nova imatge.

### 4.1 *Loss* de contingut $L_{content}$

.........
La pèrdua de contingut mesura quant difereix el contingut de la imatge objectiu del contingut de la imatge de contingut. En el camp de la transferència d'estil, "contingut" es refereix a les característiques d'alt nivell de la imatge, com les formes i l'estructura general que defineixen els elements principals de l'escena.

Com es calcula: La pèrdua de contingut es calcula típicament com l'error quadràtic mitjà (MSE) entre les representacions de característiques de la imatge objectiu i la imatge de contingut. Aquestes representacions de característiques s'extreuen d'una o més capes d'una xarxa neuronal convolucional (CNN) preentrenada, com VGG19 en el teu cas. En el teu codi, la pèrdua de contingut es calcula utilitzant els mapes de característiques de la capa 'conv4_2', coneguda per capturar bé el contingut.

### 4.2 *Loss* d'estil $L_{style}$

........
La pèrdua d'estil mesura com de diferent és l'estil de la imatge objectiu de l'estil de la imatge d'estil. "Estil" inclou textures, colors i patrons visuals de la imatge però no l'estructura d'alt nivell o el contingut.

Com es calcula: La pèrdua d'estil és més complexa i es calcula utilitzant la matriu Gram dels mapes de característiques de múltiples capes a través de la CNN. La matriu Gram captura les correlacions entre diferents mapes de característiques en una capa, representant efectivament la informació de textura. Per a cadascuna d'aquestes capes, la pèrdua d'estil és l'MSE entre les matrius Gram de la imatge objectiu i la imatge d'estil. La pèrdua d'estil a través de múltiples capes captura una gamma de detalls estilístics, des de textures més fines fins a patrons més abstractes.

### 4.3 Respecte quines capes optimitzem?

........
En la teva funció calculate_losses, primer calcules la pèrdua d'estil per a cada capa especificada comparant les matrius Gram de la imatge objectiu i de l'estil. La pèrdua d'estil per a cada capa es pondera per style_weights[layer] i es normalitza per la mida dels mapes de característiques per assegurar que la magnitud de la pèrdua d'estil no domini la pèrdua total a causa de diferències en les dimensions de la capa.

___
## 7. Documentació
* VGG -> https://doi.org/10.48550/arXiv.1409.1556

* Style Transfer -> https://arxiv.org/abs/1701.01036