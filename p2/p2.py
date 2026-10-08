import cv2
import matplotlib.pyplot as plot
import numpy as np
from skimage import data

imagen = data.astronaut() #No se encontró la imagen adjuntada a la tarea, por lo que se uso una de la librería ski image
imageng = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY) #Imagen en escala de grises
#Para normalizar, debemos encontrar el máximo
maximo = imageng.max()
#print(maximo) #255
imagennormalizada = imageng / 255 #Normalizamos por el máximo
nuevomin = imagennormalizada.min()
nuevomax = imagennormalizada.max()
#Imprimimos mínimo y máximo para corroborar que se encuentre en [0, 1]
#print(nuevomin) #0.0 
#print(nuevomax) #1.0

plot.imshow(imagennormalizada, cmap="gray")
plot.show()
