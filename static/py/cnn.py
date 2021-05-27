# *** Clasificación de imagenes de retinopatia diabetica utilizando redes neuronales convolucionales *** #

# En este documento se implementará una red convolucional capaz de clasificar los distintos grados de severidad de retinopatia diabetica.
# Para ello se usará el set de datos obtenido desde la página oficial https://www.kaggle.com/datasets,
# que contiene un total de 1928 imágenes de prueba y 3662 imagenes de entrenamiento de fondo de ojos de distintos pacientes que padecen la enfermedad mencionada.
# El tamaño de este dataset es de 9.51 GB aprox., con lo cual se recomienda descargar las imagenes a un diretorio local para su analisis posterior.

##################################################################
# IMPORTANCIÓN DE LAS LIBRERIAS A UTILIZAR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm #barra de progreso
import cv2

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras as kr
from keras.utils import np_utils

import sklearn
from skimage.transform import resize
from skimage import io
from skimage.color import rgb2gray

import random
import datetime
import re

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.optimizers import SGD, Adam

from tensorflow import keras
from keras.layers import LeakyReLU, Dropout
##################################################################
# DIRECTORIOS QUE CONTIENEN LAS IMAGENES
train_dir_colab = "static/py/Pre_Procesamiento/train_images"
test_dir_colab = "static/py/Pre_Procesamiento/test_images"

# DEFINICIÓN DE FUNCIONES A UTILIZAR PARA EL PREPROCESAMIENTO

IMG_SIZE = 250

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img

def load_ben_color(path, sigmaX=10):
    image = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE),cv2.INTER_CUBIC)
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    return image

def circle_crop(img):
    #img = cv2.imread(img)
    #img = crop_image_from_gray(img)

    height, width, depth = img.shape

    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))

    #Se construye una mascara de ceros con las nuevas dimensiones (x,y)
    circle_img = np.zeros((height, width), np.uint8)

    #Sintaxis: cv2.circle (imagen, coordenadas_centrales, radio, color, grosor)
    #Se utiliza la funcion cv2 circle para enerar una mascara circular
    #https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    #Se aplica la mascara definida en el paso anterior
    #https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img

def circle_crop_v2(img):
    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img

def load_image(image):
    img = cv2.imread(image)#/255.0
    img = circle_crop(img)
    img = load_ben_color(img, sigmaX=10)
    return img

def load_images_from_folder(folder):
    images = []
    for filename in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))#/255.0
        img = circle_crop(img)
        img = load_ben_color(img, sigmaX=10)
        if img is not None:
          img_arr = np.asarray(img)
          images.append(img_arr)
    images = np.asarray(images)
    return images

def load_images_from_folder_filter(folder, filter_img):
    images = []
    stopFormat = '.png'
    for filename in tqdm(os.listdir(folder)):
        reduced = filter(lambda w: w not in stopFormat, re.split(r'\W+', filename))
        result = list(reduced)
        validation = filter_img[filter_img["id_code"] == result[0]]
        if not validation.empty:
            img = cv2.imread(os.path.join(folder,filename))#/255.0
            img = circle_crop(img)
            img = load_ben_color(img, sigmaX=10)
            img_arr = np.asarray(img)
            images.append(img_arr)
    images = np.asarray(images)
    return images

##################################################################
# IMPORTAMOS EL TARGET del conjunto de datos de entrenamiento

path_train = "static/py/Pre_Procesamiento/train.csv"

train_dataset = pd.read_csv(path_train, sep=',')

y_train = train_dataset['diagnosis']

##################################################################
# PROCEDEMOS A VISUALIZAR LA CANTIDAD DE IMAGENES POR CLASES

def cantidadClases(df):
    clase0 = len(df[df==0])
    clase1 = len(df[df==1])
    clase2 = len(df[df==2])
    clase3 = len(df[df==3])
    clase4 = len(df[df==4])

    totalImgs = len(df)
    countClassImg = [clase0,clase1,clase2,clase3,clase4]
    porcentajeClassImg = [round(x/totalImgs*100,2) for x in countClassImg]

    return countClassImg, porcentajeClassImg

df = pd.DataFrame(data = None, index = [0,1,2,3,4],columns=["Id_Clase", "Clase","Cantidad de imagenes", "% de imagenes"])
df["Id_Clase"].iloc[:] = [0,1,2,3,4]
df["Clase"].iloc[:] = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
df["Cantidad de imagenes"].iloc[:] = cantidadClases(y_train)[0]
df["% de imagenes"].iloc[:] = cantidadClases(y_train)[1]
#print(df)

df2 = df["% de imagenes"]
#ax = df2.plot.bar(y='% de imagenes', rot=0)

##################################################################
# BALANCEAMOS LAS CLASES TOMANDO UNA MUESTRA ALEATORIA
x0 = train_dataset[train_dataset["diagnosis"] == 0].sample(200, random_state = 0)
x1 = train_dataset[train_dataset["diagnosis"] == 1].sample(200, random_state = 0)
x2 = train_dataset[train_dataset["diagnosis"] == 2].sample(200, random_state = 0)
x3 = train_dataset[train_dataset["diagnosis"] == 3].sample(193, random_state = 0)
x4 = train_dataset[train_dataset["diagnosis"] == 4].sample(200, random_state = 0)

X = pd.concat([x0,x1,x2,x3,x4])
#print(X)

# Visualización balanceo de clases de entrenamiento
X_0 = len(X[X["diagnosis"] == 0])
X_1 = len(X[X["diagnosis"] == 1])
X_2 = len(X[X["diagnosis"] == 2])
X_3 = len(X[X["diagnosis"] == 3])
X_4 = len(X[X["diagnosis"] == 4])

totalImgs = len(X)

countClassBalanc = [X_0, X_1, X_2, X_3, X_4]
porcClassBalanc = [round(x/totalImgs*100,2) for x in countClassBalanc]

df_train_class_balanc = pd.DataFrame(data = None, index = [0,1,2,3,4],columns=["Id_Clase", "Clase","Cantidad de imagenes", "% de imagenes"])
df_train_class_balanc["Id_Clase"].iloc[:] = [0,1,2,3,4]
df_train_class_balanc["Clase"].iloc[:] = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
df_train_class_balanc["Cantidad de imagenes"].iloc[:] = countClassBalanc
df_train_class_balanc["% de imagenes"].iloc[:] = porcClassBalanc
#print(df_train_class_balanc)

# Se visualiza el resultado de las clases balanceadas
#df_train_class_balanc.plot.bar(y='Cantidad de imagenes', color = 'g', rot=0)

##################################################################
# GENERACION DEL CONJUNTO DE DATOS DE PRUEBA (Y_TEST)
y_train_copy = train_dataset
y_train_copy.drop(y_train_copy.index[[X.index]])

y0 = y_train_copy[y_train_copy["diagnosis"] == 0].sample(100, random_state = 0)
y1 = y_train_copy[y_train_copy["diagnosis"] == 1].sample(100, random_state = 0)
y2 = y_train_copy[y_train_copy["diagnosis"] == 2].sample(100, random_state = 0)
y3 = y_train_copy[y_train_copy["diagnosis"] == 3].sample(100, random_state = 0)
y4 = y_train_copy[y_train_copy["diagnosis"] == 4].sample(100, random_state = 0)

y_test = pd.concat([y0,y1,y2,y3,y4])
#print(y_test)

##################################################################
# CARGA Y FILTRADO DE IMAGENES
# Se procede a filtrar las imagenes balanceadas creadas en los pasos anteriores
# y cargando solamente estas imagenes en memoria. Para ejecutar este paso,
# debe setearse la variable **"LOAD_FROM_IMAGES"** en **False**.

LOAD_FROM_IMAGES = True

if(not(LOAD_FROM_IMAGES)):
    x_train_filter = load_images_from_folder_filter(train_dir_colab, X)
    x_test_filter = load_images_from_folder_filter(train_dir_colab, y_test)

# save numpy array as npy file
#https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/
from numpy import asarray
from numpy import save
from numpy import load

LOAD_STATUS = True

if(not(LOAD_STATUS)):
    save('static/py/backup/backup_train_filter.npy', x_train_filter)
    save('static/py/backup/backup_test_filter.npy', x_test_filter)

    save('static/py/backup/backup_train_y_filter.npy', X["diagnosis"])
    save('static/py/backup/backup_test_y_filter.npy', y_test["diagnosis"])

x_train_filter = np.load('static/py/backup/backup_train_filter.npy',allow_pickle=False,fix_imports=True,encoding='latin1')
y_train_filter = np.load('static/py/backup/backup_train_y_filter.npy',allow_pickle=False,fix_imports=True,encoding='latin1')

x_test_filter = np.load('static/py/backup/backup_test_filter.npy',allow_pickle=False,fix_imports=True,encoding='latin1')
y_test_filter = np.load('static/py/backup/backup_test_y_filter.npy',allow_pickle=False,fix_imports=True,encoding='latin1')

#print("\nDATA_TRAIN_X_FILTER: \n")
#print(x_train_filter.shape)

#print("\nDATA_TRAIN_Y_FILTER: \n")
#print(y_train_filter.shape)

#print("\nDATA_TEST_X_FILTER: \n")
#print(x_test_filter.shape)

#print("\nDATA_TEST_Y_FILTER: \n")
#print(y_test_filter.shape)

##################################################################
# VISUALIZACIÓN DE IMAGEN APLICANDO EL PREPROCESAMIENTO

demo_path = 'static/py/Pre_Procesamiento/demo'
demo_filter = load_images_from_folder(demo_path)

img3_sinFiltrar = cv2.imread('static/py/Pre_Procesamiento/demo/0f96c358a250.png')
img3_sinFiltrar = cv2.cvtColor(img3_sinFiltrar, cv2.COLOR_BGR2RGB)
circleImg3 = circle_crop(img3_sinFiltrar)


img4_sinFiltrar = cv2.imread('static/py/Pre_Procesamiento/demo/2cdcc910778d.png')
img4_sinFiltrar = cv2.cvtColor(img4_sinFiltrar, cv2.COLOR_BGR2RGB)
circleImg4 = circle_crop(img4_sinFiltrar)
'''
f = plt.figure(figsize=(20,20))
f.add_subplot(2,3, 1)
plt.imshow(img3_sinFiltrar, cmap=plt.cm.binary)


f.add_subplot(2,3, 2)
plt.imshow(circleImg3, cmap=plt.cm.binary)


f.add_subplot(2,3, 3)
plt.imshow(demo_filter[2], cmap=plt.cm.binary)

f.add_subplot(2,3, 4)
plt.imshow(img4_sinFiltrar, cmap=plt.cm.binary)


f.add_subplot(2,3, 5)
plt.imshow(circleImg4, cmap=plt.cm.binary)


f.add_subplot(2,3, 6)
plt.imshow(demo_filter[3], cmap=plt.cm.binary)

#plt.show()
'''
##################################################################
# Se normaliza el target del conjunto de entrenamiento y prueba
# Se convierte el array y_train_filter y el array y_test_cod
# en en una matriz de clase binaria de 5 categorias (0,1,2,3,4)
# https://keras.io/api/utils/python_utils/#to_categorical-function

nclases = 5
y_train_cod = np_utils.to_categorical(y_train_filter, nclases)
y_test_cod = np_utils.to_categorical(y_test_filter, nclases)
# print(y_test_cod.shape)

##################################################################
# IMPLEMENTACION RED NEURONAL CONVOLUCIONAL (CNN)

# Creacion del modelo
modelo = Sequential()
# CONV1 Y MAX-POOLING1
modelo.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(250,250,3)))
modelo.add(MaxPooling2D(pool_size=(2,2)))

# CONV2 Y MAX-POOLING2
modelo.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))

# Aplanar, FC1, FC2 y salida
modelo.add(Flatten())
modelo.add(Dense(120,activation='relu'))
modelo.add(Dense(84,activation='relu'))
modelo.add(Dense(nclases,activation='softmax'))

#sgd = SGD(lr=0.1)
modelo.compile(loss='categorical_crossentropy',
               optimizer=tf.keras.optimizers.Adam(lr=0.001),
               metrics=['accuracy'])

#---------------------------
# Se omite la compilacion del modelo y el entrenamiento,
# ya que se realizó previamente y se utilizará los pesos entrenados.
#------#

##################################################################
# SE CARGA EL JSON Y SE CREAR EL MODELO
# SE DEBE ESPECIFICAR LA RUTA DEL ARCHIVO .JSON Y .H5 QUE SE DESEA UTILIZAR

from keras.models import model_from_json

model_output = "static/py/backup/model  4-7-2020(dd-mm-yyyy).json"
pesos_model_output = "static/py/backup/pesos model  4-7-2020(dd-mm-yyyy).h5"

json_file = open(model_output, 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# SE CARGA LOS PESOS EN EL NUEVO MODELO
loaded_model.load_weights(pesos_model_output)

print("Se cargó correctamente el modelo y los pesos desde la app")

##################################################################
# DESEMPEÑO DEL MODELO

from sklearn.metrics import confusion_matrix
y_pred = loaded_model.predict_classes(x_test_filter)
y_ref = np.argmax(y_test_cod,axis=1)
etiquetas = ['0','1','2','3','4']

#print(confusion_matrix(y_ref, y_pred))

##################################################################
# REALIZANDO PREDICCIONES de un nuevo dataset

test1_path = 'static/py/test_example_sin_etiquetar'
test1 = load_images_from_folder(test1_path)
clasesPredichas = loaded_model.predict_classes(test1)
# print(clasesPredichas)

'''
plt.style.use("seaborn-dark")
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
plt.figure(figsize=(200,200))
for i in range(4):
    plt.subplot(50,50,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test1[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[clasesPredichas[i]])
plt.show()
'''

##################################################################
# Realizando una prediccion de datos de prueba utilizando los pesos del modelo entrenado
# EVALUANDO EL MODELO CARGADO CON LOS DATOS DE PRUEBA
loaded_model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(lr=0.001),
                     metrics=['accuracy'])
score = loaded_model.evaluate(x_test_filter, y_test_cod, verbose=1)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
##################################################################
