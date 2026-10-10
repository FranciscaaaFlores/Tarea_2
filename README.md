# Tarea 2
En este repositorio se encuentran 2 carpetas (P1 y P2) con sus respectivos archivos .py
Los códigos no cuentan con interfaz gráfica, por lo que en las líneas correspondientes en donde se llama a la función es posible modificar los parámetros.

## P1 (Pregunta 1 — Ruido Poisson y filtrado Gaussiano adaptativo)
En esta pregunta se ha creado una imagen sintética de 256x256 píxeles con 3 regiones: el fondo (intensidad 0.15), un cuadrado centrado de 128x128 pixeles （intensidad 0.45）
y un círculo centrado (intensidad 0.8). 

Para cumplir con el objetivo principal de esta pregunta, el cual era construir un kernel Gaussiano isotrópico a partir de σ,
y luego filtrar la imagen sintética con ruido Poisson se implementó la función kernelgaussianoyfiltrado(imagen, sigma) la cual 
recibe 2 parámetros:
1. **"imagen"**: Imagen sintética ya creada.
2. **"sigma"**: Un valor de sigma para definir el kernel y su número de puntos de forma proporcional.
(Para pasarle a la función un arreglo de sigmas se encuentra en el código un ciclo for que recorre el arreglo "sigmas", en donde se puede agregar la cantidad que se desee). 

### Cómo usar:
Para usar este código, se debe realizar lo antes mencionado, modificar el arreglo "sigmas" con valores arbitrarios a elección. Luego, 
el código se encarga de llamar a la función de kernelgaussianoyfiltrado(imagen, sigma) en donde aplica el filtro gaussiano a la imagen
según cada uno de los parámetros. 
En este código se encuentran desarrolladas varias acciones, por lo que la mayoría se encuentran comentadas para que no ocurra todo 
al mismo tiempo. Se encuentra detallado en el archivo .py que se debe descomentar para cada cosa. Entre estas encontramos:
1. Descomentar para ver el plot de la imagen filtrada producida para cada sigma:
```python
    #plt.figure(figsize=(12, 4))
    #plt.subplot(1, 2, 1)
    #plt.imshow(ruidopoisson, cmap="gray")
    #plt.title("Imagen con ruido Poisson")

    #plt.subplot(1, 2, 2)
    #plt.imshow(imagenfiltrada, cmap="gray")
    #plt.title(f"Imagen Filtrada (sigma = {sigma})")

    #plt.show()
```

3. Descomentar para ver el Plot de las curvas de cada RMSE:
```python
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
```

3. Descomentar para ver como 3era imagen la original sin ruido en vez del mejor filtro gaussiano.
```python
  #plt.subplot(2, 1, 2)
  #plt.imshow(ims, cmap="gray")
  #plt.title("Imagen Original (sin ruido)")
```

5. Descomentar para ver los valors de RMSE para el filtrado adaptativo
```python
  #print(f"RMSE m1 = {np.round(rmsem1a,3)}")
  #print(f"RMSE m2 = {np.round(rmsem2a,3)}")
  #print(f"RMSE m3 = {np.round(rmsem3a,3)}")
  #print(f"RMSE global = {np.round(rmsega,3)}")
```

### Resultados:
La salida del código es:
1. Una imagen, la cual contiene: a) La imagen con ruido Poisson, b) Imagen con filtro adaptativo, c) Mejor filtro gaussiano, sigma = 3.
2. Mapa de sigmas del filtro adaptativo.

Es posible además obtener la gráfica de la imagen resultado de un arreglo de sigmas, la gráfica de las 3 curvas RMSE, los valores RMSE para el mejor 
filtro gaussiano, entre otros, para lo cual se deben descomentar las respectivas líneas especificadas arriba.

## P2 (Pregunta 2 — Difusión anisotrópica, Variación Total y diseño del coeficiente de difusión)
Para esta pregunta se pedía usar una imagen adjuntada con la tarea, pero no se encontró un archivo de imagen en canvas. Se ha utilizado una imagen de ski image (astronaut), la cual mediante las librerías de cv2 se ha pasado a escala de grises.
En este código 

Para cumplir con el objetivo de esta pregunta que era aplicar difusión anisotrópica, explícita y variación total se ha 
definido la función difanisotropica(Iijt, iteraciones, l, Koeps, modo) la cual recibe 5 parámetros:
1. **"Iijt"**: Imagen con ruido gaussiano ya aplicado.
2. **"iteraciones"**: Número de iteraciones.
3. **"l"**: Paso temporal. 
4. **"Koeps"**: En el caso de querer aplicar difusión anisotrópica explícita este parámetro corresponderá a K (el cual se reemplaza en la función h, y en el caso de ser variación total el parámetro corresponde a epsilon. 
5. **"modo"**: La función cuenta con 2 modos: "explicito" para aplicar difusión anisotrópica explícita y "cTV" para variación total.

### Cómo usar:
Para usar el código basta con modificar el arreglo "epsilons" (se ha definido de esta manera ya que en lo que se alcanzó a hacer de esta pregunta fue la mayoría variación total, sin embargo si se quiere usar el modo "explicito" se reemplaza en este arreglo los valores deseados para K.
Posteriormente, el código se encarga de llamar la función, y mediante un for va calculando las imágenes para cada parámetro ingresado en el arreglo. 

### Resultados:
Tras correr el código se obtiene una serie de imágenes (tantas como parámetros epsilon/K se hayan ingresado), en donde cada una cuenta con 2 imágenes: a) Mapa cVT obtenido y b) Imagen obtenida con el epsilon determinado.
