# Reconocimiento de cara con cubrebocas usando Cascade trainer GUI


##Iniciare mostrando el codigo con el cual saque mis fotos para mi dataset:





```python
import cv2
import matplotlib.pyplot as plt
import os 
import imutils
from mtcnn.mtcnn import MTCNN
Fotos = 'ConCubrebocas'
direccion = '/Users/rafaelfabiansilva/IA2/CubrebocasFotos'
carpeta = direccion + '/' + 'ConCubrebocas'

if not os.path.exists(carpeta):
    print('Carpeta creada: ', carpeta)
    os.makedirs(carpeta)

detector = MTCNN()
cap = cv2.VideoCapture(0)
count = 550

while True:
    ret, frame = cap.read()
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    copia = frame.copy()

    caras = detector.detect_faces(copia)

    for i in range (len(caras)):
        x1, y1, ancho, alto = caras [i]['box']
        x2, y2 = x1 + ancho, y1 + alto
        cara_reg = gris[y1:y2, x1:x2]
        cara_reg = cv2.resize(cara_reg, (150,200), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(carpeta +"/rostro_{}.jpg".format(count), cara_reg)
        count = count + 1
    cv2.imshow("Entrenamiento", frame)

    t= cv2.waitKey(1)
    if t == 27 or count >= 700:
        break

cap.release()
cv2.destroyAllWindows()


```




####Adjunto video de muestra y prueba
[Video de muestra y prueba](https://youtu.be/60lzmNb0Zpw)




#Prueba Final
Número de positivas: 700

Número de negativas: 1,273


Tamaño de las fotos: 30x30


scaleFactor = 12 minNeighbors = 180



##Codigo

```python
import cv2 as cv
rostro = cv.CascadeClassifier('/Users/rafaelfabiansilva/IA2/cascade.xml')
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 12, 180, minSize=(700,780))
    for(x,y,w,h) in rostros:
        frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv.imshow('rostros',frame)
    k = cv.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

```








####Adjunto video del resultado final del reconocimiento facial con cubre bocas

[Video del resultado final](https://youtu.be/pGbbaaJwqWQ?si=NdFqShAj5o3rZj8e)













