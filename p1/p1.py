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

#AQUÍ SE PUEDE MODIFICAR PARA VER UN SOLO SIGMA, DESCOMENTAR
#sigma = 2
#imagenfiltrada = kernelgaussianoyfiltrado(ruidopoisson, sigma= sigma)

#ACA SE ENTREGA UN ARREGLO ARBITRARIO DE SIGMAS 
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

#-----------------------------------------------------------------------------------------
    #SE PUEDE DESCOMENTAR PARA VER LA IMAGEN FILTRADA PRODUCIDA PARA CADA SIGMA (esta comentado para no retrasar otros resultados)
    #plt.figure(figsize=(12, 4))
    #plt.subplot(1, 2, 1)
    #plt.imshow(ruidopoisson, cmap="gray")
    #plt.title("Imagen con ruido Poisson")

    #plt.subplot(1, 2, 2)
    #plt.imshow(imagenfiltrada, cmap="gray")
    #plt.title(f"Imagen Filtrada (sigma = {sigma})")

    #plt.show()
#-----------------------------------------------------------------------------------------

#Se imprimen los valores solicitados
for i in range (len(sigmas)):
    print(f"Sigma = {sigmas[i]}")
    print(f"RMSE m1 = {np.round(rmsem1[i],3)}")
    print(f"RMSE m2 = {np.round(rmsem2[i],3)}")
    print(f"RMSE m3 = {np.round(rmsem3[i],3)}")
    print(f"RMSE global = {np.round(rmseg[i],3)}")
    print("__________________________________________________")

#-----------------------------------------------------------------------------------------
#SE PUEDE DESCOMENTAR EN CONJUNTO A LA LÍNEA CON SOLO UN VALOR DE SIGMA PARA VER SU GRÁFICA
#plt.figure(figsize=(12, 4))
#plt.subplot(1, 2, 1)
#plt.imshow(ruidopoisson, cmap="gray")
#plt.title("Imagen con ruido Poisson")

#plt.subplot(1, 2, 2)
#plt.imshow(imagenfiltrada, cmap="gray")
#plt.title(f"Imagen Filtrada (sigma = {sigma})")

#plt.show()
#-----------------------------------------------------------------------------------------

#___________________________________exp y análisis__________________________
#Vemos en donde están los mínimos de cada RMSE para poder graficar el punto
min1 = np.argmin(rmsem1)
min2 = np.argmin(rmsem2)
min3 = np.argmin(rmsem3)
ming = np.argmin(rmseg)


#-----------------------------------------------------------------------------------------
#SE PUEDE DESCOMENTAR PARA VER EL GRÁFICO DE RMSE 
#Graficamos las curvas de cada RMSE
#plt.figure(figsize =(12, 4))
#plt.plot(sigmas, rmsem1, label = "Máscara 1 (círculo)")
#plt.plot(sigmas, rmsem2, label = "Máscara 2 (cuadrado exc. círculo)")
#plt.plot(sigmas, rmsem3, label = "Máscara 3 (fondo)")
#plt.plot(sigmas, rmseg, label = "Global")

#Ocupamos los índices calculados anteriormente, y graficamos los mínimos como puntos del mismo color de las curvas
#plt.plot(sigmas[min1], rmsem1[min1], color="blue", marker="o", label=f"Mín m1 ({np.round(sigmas[min1],3)}, {np.round(rmsem1[min1],3)})")
#plt.plot(sigmas[min2], rmsem2[min2], color="orange", marker="o", label=f"Mín m2 ({np.round(sigmas[min2],3)}, {np.round(rmsem2[min2],3)})")
#plt.plot(sigmas[min3], rmsem3[min3], color="green", marker="o", label=f"Mín m3 ({np.round(sigmas[min3],3)}, {np.round(rmsem3[min3],3)})")
#plt.plot(sigmas[ming], rmseg[ming], color="red", marker="o", label=f"Mín global ({np.round(sigmas[ming],3)}, {np.round(rmseg[ming],3)})")

#plt.xlabel("Sigma")
#plt.ylabel("RMSE")
#plt.legend()
#plt.title("Gráfica de RMSE para cada sigma definido")
#plt.show()
#-----------------------------------------------------------------------------------------

#Para el filtro adaptativo generaremos una función por tramos en donde se asigne un sigma cercano al optimo, 
#pues se vio que en los 4 casos era sigma = 3, entonces para no asignar el mismo sigma a todas las zonas
#se decidió utilizar los más cercanos sin aumentar de gran manera el RMSE
def sigmasadaptativos(imagen):
    primerfiltrado = kernelgaussianoyfiltrado(ruidopoisson, sigma = 1)
    #El filtrado auxiliar se hace con sigma = 1, tal como se explica en el informe, no se escoge sigma = 3
    #debido a que el hacer el mejor filtrado gaussiano de una vez y luego aplicar el adaptativo genera distorsiones
    #en los bordes de zonas como el círculo.
    sigma = np.zeros((256, 256))
    for y in range(256):
        for x in range(256):
            refintensidad = primerfiltrado[y,x]
            if refintensidad <= 0.3:
                sigma[y,x] = 2 #para zonas como el fondo, el sigma optimo establecido será 2
            elif (refintensidad > 0.3 and refintensidad <= 0.7):
                sigma[y,x] = 3 #para zonas como el cuadrado, el sigma optimo establecido será 3 (el óptimo real)
            else:
                sigma[y, x] = 3.5 #para zonas como el círculo, el sigma optimo establecido será 3.5
    return sigma, primerfiltrado

arreglosigma = sigmasadaptativos(ruidopoisson)[0] 

#Definimos una función para el filtro adaptativo
def filtradoadaptativo(imagen, arreglosigma):
    imagenfiltradoadaptativo = np.zeros((256, 256))
    imi1 = kernelgaussianoyfiltrado(imagen, sigma = 2)
    imi2 = kernelgaussianoyfiltrado(imagen, sigma = 3)
    imi3 = kernelgaussianoyfiltrado(imagen, sigma = 3.5)
    for y in range(256):
        for x in range(256):
            scorresponde = arreglosigma[y,x]
            if scorresponde == 2:
                imagenfiltradoadaptativo[y,x] = imi1[y,x]
            elif scorresponde == 3:
                imagenfiltradoadaptativo[y,x] = imi2[y,x]
            elif scorresponde == 3.5:
                imagenfiltradoadaptativo[y,x] = imi3[y,x]

    return imagenfiltradoadaptativo

imagenfiltradoadaptativo = filtradoadaptativo(ruidopoisson, arreglosigma)
mejorfiltrogaussiano = kernelgaussianoyfiltrado(ruidopoisson, 3)
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(ruidopoisson, cmap="gray")
plt.title("Imagen con ruido Poisson")

plt.subplot(2, 2, 2)
plt.imshow(imagenfiltradoadaptativo, cmap="gray")
plt.title("Imagen con filtrado adaptativo")

#-----------------------------------------------------------------------------------------
#SE PUEDE DESCOMENTAR PARRA VER COMO 3ERA IMAGEN LA ORIGINAL SIN RUIDO EN VEZ DEL MEJOR FILTRADO GAUSSIANO
#plt.subplot(2, 1, 2)
#plt.imshow(ims, cmap="gray")
#plt.title("Imagen Original (sin ruido)")
#-----------------------------------------------------------------------------------------
plt.subplot(2, 1, 2)
plt.imshow(mejorfiltrogaussiano, cmap="gray")
plt.title("Mejor filtrado gaussiano, sigma = 3")

plt.show()
#_____________
plt.figure(figsize=(6, 6))
plt.imshow(arreglosigma, cmap="viridis") 
plt.colorbar(label="Valor de sigma asignado")
plt.title("Mapa de sigmas $\\sigma(x,y)$")
plt.show()

#Se imprimen los valores solicitados, esta vez para el filtro adaptativo
#print("Valores para Filtro Adaptativo")
error = ims - imagenfiltradoadaptativo
rmsem1a = np.sqrt(np.sum((error * m1)**2)/np.sum(m1))
rmsem2a = np.sqrt(np.sum((error * m2)**2)/np.sum(m2))
rmsem3a = np.sqrt(np.sum((error * m3)**2)/np.sum(m3))
rmsega = np.sqrt(np.mean(error**2))
#-----------------------------------------------------------------------------------------
#SE PUEDE DESCOMENTAR PARA VER LOS VALORES DE RMSE PARA EL FILTRADO ADAPTATIVO
#print(f"RMSE m1 = {np.round(rmsem1a,3)}")
#print(f"RMSE m2 = {np.round(rmsem2a,3)}")
#print(f"RMSE m3 = {np.round(rmsem3a,3)}")
#print(f"RMSE global = {np.round(rmsega,3)}")
#-----------------------------------------------------------------------------------------

print("Valores para sigma en un píxel de cada región")
print("Fondo:", arreglosigma[12,12])
print("Cuadrado:", arreglosigma[70, 70])
print("Círculo:", arreglosigma[128, 128])

print("_____________________________________________")
print("Valores para mu gorro (filtrado auxiliar) en un píxel de cada región")
print("Fondo:", np.round(sigmasadaptativos(ruidopoisson)[1][12,12],3))
print("Cuadrado:", np.round(sigmasadaptativos(ruidopoisson)[1][70, 70],3))
print("Círculo:", np.round(sigmasadaptativos(ruidopoisson)[1][128, 128],3))
