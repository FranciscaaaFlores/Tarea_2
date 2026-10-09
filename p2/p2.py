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

def difanisotropica(Iijt, iteraciones, l, Koeps, modo):
    It = Iijt.copy()
    filas = It.shape[0]
    cols = It.shape[1]
    #Para aplicar difusión anisotropica utilizamos las fórmulas vistas en la clase 10, comenzamos calculando 
    #las derivadas direccionales y los coeficientes de difusión
    for t in range(iteraciones):
        if modo == "cTV":
            epsilon = Koeps
            #Tomamos I = u 
            gradx = np.gradient(It)[0]
            grady = np.gradient(It)[1]
            gradu2 = gradx**2 + grady**2

            #Aplicamos la fórmula del enunciado para el coeficiente difusión:
            cTV = 1/ np.sqrt(gradu2 + epsilon**2)

        else:
            cmax = 0

        for i in range(1, filas - 1):
            for j in range(1, cols - 1):
                DN = It[i - 1, j] -  It[i, j]
                DS = It[i + 1, j] -  It[i, j]
                DE = It[i, j + 1] -  It[i, j]
                DW = It[i, j - 1] -  It[i, j]

                if modo == "explicito":
                    K = Koeps
                    cN = h(np.abs(DN), K)
                    cS = h(np.abs(DS), K)
                    cE = h(np.abs(DE), K)
                    cW = h(np.abs(DW), K)

                    #Luego, usamos la fórmula del operador de actualización de forma discreta
                    It[i, j] = It[i, j] + l * (cN * DN + cS * DS + cE * DE + cW * DW)

                elif modo == "cTV":
                    It[i, j] = It[i, j] + l * cTV[i, j] * (DN + DS + DE + DW)
                    
    return It, cmax

#Procedemos de la misma manera que en la py con los sigmas, esta vez con los epsilons. Pensando que epsilon evita la división por cero,
#definiremos algunos de los valores cercanos a 0 para ver el comportamiento

#epsilons = [0.005, 0.05, 0.1, 1, 10]
epsilons = [0.1]
for epsilon in epsilons:
    I = difanisotropica(Iijt = imagenconruido, iteraciones = 10, l = 10, Koeps= epsilon, modo = "cTV")[0]
    cTV = difanisotropica(Iijt = imagenconruido, iteraciones = 10, l = 10, Koeps= epsilon, modo = "cTV")[1]

    plot.figure(figsize=(12,4))
    plot.subplot(1,2,1)
    plot.imshow(cTV, cmap = "viridis")
    plot.colorbar()
    plot.title("Mapa CTV obtenido")

    plot.subplot(1,2,2)
    plot.imshow(I, cmap= "gray")
    plot.title(f"Imagen obtenida con ($\epsilon$ = {epsilon})")
    plot.show()

print("Listo")