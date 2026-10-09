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

#plot.imshow(imagennormalizada, cmap="gray")
#plot.show()

np.random.seed(12) #semilla fija para el generador aleatorio
ruidogaussiano = np.random.normal(0, 0.05, imagennormalizada.shape)
#definimos el ruido gaussiano con los parámetros solicitados media = 0, desviación estándar = 0.05, y tamaño de la imagen normalizada
imagenconruido = imagennormalizada + ruidogaussiano #Le aplicamos el ruido gaussiano a la imagen normalizada
#plot.imshow(imagenconruido, cmap="gray") 
#plot.title("Imagen con ruido Gaussiano")
#plot.show()

def h(x, K):
    h = 1 / (1 + (x/K)**2) #Para sacar los coeficientes de difusión
    return h

def difasintropica(Iijt, iteraciones, l, K):
    It = Iijt
    filas = It.shape[0]
    cols = It.shape[1]
    #Para aplicar difusión asintrópica utilizamos las fórmulas vistas en la clase 10, comenzamos calculando 
    #las derivadas direccionales y los coeficientes de difusión
    for t in range(iteraciones):
        for i in range(1, filas - 1):
            for j in range(1, cols - 1):
                DN = It[i - 1, j] -  It[i, j]
                DS = It[i + 1, j] -  It[i, j]
                DE = It[i, j + 1] -  It[i, j]
                DW = It[i, j - 1] -  It[i, j]

                cN = h(np.abs(DN), K)
                cS = h(np.abs(DS), K)
                cE = h(np.abs(DE), K)
                cW = h(np.abs(DW), K)

                #Luego, usamos la fórmula del operador de actualización de forma discreta
                It[i, j] = It[i, j] + l * (cN * DN + cS * DS + cE * DE + cW * DW)

    return It