import cv2
import numpy as np

# Iniciando a captura de video:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Definindo os limites das cores:
# HSV -> Hue Sat Value
lower_cyan = np.array([95, 90, 20])
upper_cyan = np.array([110, 255, 255])

lower_magenta = np.array([155, 100, 30])
upper_magenta = np.array([180, 255, 255])

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
    mask_both = cv2.add(mask_cyan, mask_magenta)

    # Sobrepondo as mascaras com a imagem original para pegar as cores
    # da imagem original:
    output = cv2.bitwise_and(frame, frame, mask=mask_both)

    # Mostrando o resultado:
    #cv2.imshow("Mask", mask_both)
    cv2.imshow("Output", output)

    # Apertar Q para sair do loop principal:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fechando as janelas e desligando a webcam:
cv2.destroyAllWindows()
cap.release()
