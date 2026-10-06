import numpy as np
import matplotlib.pyplot as plt

#Creamos arreglo, matriz para la imagen sintética (de 256x256)
ims = np.zeros((256, 256))

#Hacemos una función para determinar si un pixel debe ser considerado como parte del cuadrado (y asignado 0.45), 
#o fondo (con intensidad 0.15).
def dentrodelcuadrado(y, x):
    if (x >= 64 and x <= 192) and (y >= 64 and y <= 192):
        intensidad = 0.45
    else:
        intensidad = 0.15

    return intensidad

#Ahora con un for rellenaremos el arreglo con las intensidades 0.15, 0.45 o 0.8 dependiendo de la zona
#a la cual pertenezca.
for y in range(256):
    for x in range(256):
        #Para el círculo utilizaremos la condición de distancia, si la distancia es menor o igual al radio será
        #incluido en el círculo y por ende se le asigna intensidad 0.8.
        distancia = np.sqrt((x - 128)**2 + (y -128)**2)
        if distancia <= 32:
            ims[y,x] = 0.8
        #De no ser parte del círculo entra a la función creada para definir si será fondo o parte del cuadrado.
        elif distancia > 32:
            ims[y,x] = dentrodelcuadrado(y,x)

np.random.seed(12) #semilla fija para el generador aleatorio
ruidopoisson = np.random.poisson(40* ims) / 40 #definimos el ruido Poisson con la información del enunciado
