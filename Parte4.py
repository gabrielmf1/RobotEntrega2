import cv2
import numpy as np
from math import acos, degrees, sqrt

# Iniciando a captura de video:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Definindo os limites das cores:
# HSV -> Hue Sat Value
lower_cyan = np.array([95, 90, 20])
upper_cyan = np.array([110, 255, 255])

lower_magenta = np.array([165, 100, 20])
upper_magenta = np.array([180, 255, 255])

# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

#Mede a distancia entre a folha e a camera
def dist_folha_camera(n):
    entre_circ = 14
    f = 525

    distancia = f * entre_circ/n

    return distancia

#mede a distancia entre dois pontos
def dist_pontos(xa, ya, xb, yb):

    d = (xb - xa)**2+(yb - ya)**2
    dist = sqrt(d)

    return dist
# Loop principal:
while True:
    # Pegando um frame da webcam:
    _, frame = cap.read()

    # Convertendo o frame para HSV:
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fazendo a mascara das duas cores:
    mask_cyan = cv2.inRange(frame_hsv, lower_cyan, upper_cyan)
    mask_magenta = cv2.inRange(frame_hsv, lower_magenta, upper_magenta)

    # Juntando as duas mascaras:
    mask_both = cv2.bitwise_or(mask_cyan, mask_magenta)

    # Sobrepondo as mascaras com a imagem original para pegar as cores
    # da imagem original:
    output = cv2.bitwise_and(frame, frame, mask=mask_both)

    # Convertendo o frame para GRAY:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tirando os ruidos da imagem:
    frame_blur = cv2.GaussianBlur(frame_gray,(5,5),0)
    
    # Retirando as bordas do frame:
    edges = auto_canny(frame_blur)

    # Bordas com cor:
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 2, 40, param1=100, param2=100, minRadius=5, maxRadius=60)

    # Verificando se pelomenos 1 circulo foi encontrado:
    if circles is not None:
        # Converter as coordenadas (x, y) para numeros inteiros:
        circles = np.round(circles[0, :]).astype("int")


        # Desenhando uma linha entre os dois primeiros circulos
        if len(circles) == 2:
            # Pegando as coordenadas de cada circulo:
            xa, ya, ra = circles[0]
            xb, yb, rb = circles[1]

            distancia = dist_pontos(xa, ya, xb, yb)

            # Calculando o angulo da linha com a horizontal:
            angulo = acos(abs(xb - xa) / distancia)
            angulo = degrees(angulo)

            # Colocanto a distancia e angulo na tela:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(output, 'Distancia : {0:.2f}cm'.format(dist_folha_camera(distancia)), (0,25), font, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(output, 'Angulo : {0:.2f} graus'.format(angulo), (0,60), font, 1, (255,255,255), 2, cv2.LINE_AA)


            #Colocando a linha entre as bolas
            cv2.line(output, (xa, ya), (xb, yb), (255,0,0), 4)


    # Mostrando o resultado:
    cv2.imshow("Output", output)

    # Apertar Q para sair do loop principal:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechando as janelas e desligando a webcam:
cv2.destroyAllWindows()
cap.release()