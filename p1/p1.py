import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

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

#Para crear las máscaras hacemos una función con la misma lógica que cuando hicimos la imagen sintéticas, 
#definimos 3 arreglos m1, m2 y m3, y dependiendo de las condiciones ya mencionadas determinamos a que máscara
#corressponde
def mascaras(arreglo):
    m1 = np.zeros((256, 256))
    m2 = np.zeros((256, 256))
    m3 = np.zeros((256, 256))
    for y in range(256):
        for x in range(256):
            distancia = np.sqrt((x - 128)**2 + (y-128)**2)
            if distancia <= 32:
                m1[y,x] = 1
            elif distancia > 32:
                i = dentrodelcuadrado(y,x)
                if i == 0.45:
                    m2[y,x] = 1
                else:
                    m3[y,x] = 1
    return m1,m2,m3

#Luego, las 3 máscaras quedan:
m1 = mascaras(ims)[0] #circulo
m2 = mascaras(ims)[1]#cuadrado excluyendo el círculo
m3 = mascaras(ims)[2] #fondo

def kernelgaussianoyfiltrado(imagen, sigma):
    #En las cápsulas pudimos ver un ejemplo de kernel 5x5 (-2, 2, 5) y sigma = 1, en este caso es necesario
    #adaptar el kernel en función de sigma (no necesariamente 1), con la cantidad de puntos necesarios (considerando
    #que esta sea simétrica e impar)
    x = np.linspace(-2 * sigma, 2 * sigma, int((2 * sigma)+ 1))
    y = np.linspace(-2 * sigma, 2 * sigma, int((2 * sigma)+ 1))
    kx, ky = np.meshgrid(x, y)

    #Usamos lo visto en las cápsulas y la función gaussiana de la clase 8
    kernelgauss = np.exp(-(kx**2 + ky**2) / (2 *(sigma**2)))
    #Se normaliza para mantener la energía de la imagen
    kernelgaussnormalizado = kernelgauss / np.sum(kernelgauss)

    #Para aplicar el kernel realizado, lo convolucionamos con la imagen. Para esto utilizamos la función convolve2d de
    #SciPy
    imagenfiltrada = convolve2d(imagen, kernelgaussnormalizado, mode = "same")

    return imagenfiltrada

#Aquí se puede modificar el parámetro sigma 
#sigma = 2
#imagenfiltrada = kernelgaussianoyfiltrado(ruidopoisson, sigma= sigma)

#Para la experimentación y análisis, se define un areglo con todos los sigmas probados para graficar. 
sigmas = [0.5, 0.8, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7, 10] 
#Creamos 4 arreglos para almacenar los resultados de los RMSE (máscara 1, máscara 2, máscara 3 y global)
rmsem1 = []
rmsem2 = []
rmsem3 = []
rmseg = []

#Recorremos los sigmas, calculando la imagen filtrada con el valor correspondiente. Luego calculamos el error para finalmente
#calcular los 4 RMSE solicitados (añadiendolos al respectivo arreglo para facilitar luego la gráfica)
for sigma in sigmas:
    imagenfiltrada = kernelgaussianoyfiltrado(ruidopoisson, sigma= sigma)
    error = ims - imagenfiltrada
    rmsem1.append((np.sqrt(np.sum((error * m1)**2)/np.sum(m1))))
    rmsem2.append(np.sqrt(np.sum((error * m2)**2)/np.sum(m2)))
    rmsem3.append(np.sqrt(np.sum((error * m3)**2)/np.sum(m3)))
    rmseg.append(np.sqrt(np.mean(error**2)))

    #plt.figure(figsize=(12, 4))
    #plt.subplot(1, 2, 1)
    #plt.imshow(ruidopoisson, cmap="gray")
    #plt.title("Imagen con ruido Poisson")

    #plt.subplot(1, 2, 2)
    #plt.imshow(imagenfiltrada, cmap="gray")
    #plt.title(f"Imagen Filtrada (sigma = {sigma})")

    #plt.show()


#Se imprimen los valores solicitados
for i in range (len(sigmas)):
    print(f"Sigma = {sigmas[i]}")
    print(f"RMSE m1 = {np.round(rmsem1[i],3)}")
    print(f"RMSE m2 = {np.round(rmsem2[i],3)}")
    print(f"RMSE m3 = {np.round(rmsem3[i],3)}")
    print(f"RMSE global = {np.round(rmseg[i],3)}")
    print("__________________________________________________")

#plt.figure(figsize=(12, 4))
#plt.subplot(1, 2, 1)
#plt.imshow(ruidopoisson, cmap="gray")
#plt.title("Imagen con ruido Poisson")

#plt.subplot(1, 2, 2)
#plt.imshow(imagenfiltrada, cmap="gray")
#plt.title(f"Imagen Filtrada (sigma = {sigma})")

#plt.show()

