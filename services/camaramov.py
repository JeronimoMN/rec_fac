import cv2
import numpy as np
import os
import time
import subprocess
    
# Configuración de la resolución
ANCHO = 640  # Cambia según lo que necesites
ALTO = 480

# Definir el área de recorte (más pequeño para reducir visibilidad)
RECORTE_X1, RECORTE_Y1 = 60, 60  # Esquina superior izquierda
RECORTE_X2, RECORTE_Y2 = 460, 300  # Esquina inferior derecha

# Función para obtener el nombre de la pantalla
def obtener_pantalla():
    try:
        salida = subprocess.check_output("xrandr | grep ' connected'", shell=True).decode()
        return salida.split()[0]  # Extrae el primer valor (nombre de la pantalla)
    except Exception as e:
        print("Error al obtener la pantalla:", e)
        return "HDMI-1"

# Nombre de la pantalla detectada
pantalla = obtener_pantalla()
print(f"Usando pantalla: {pantalla}")

# Funciones para encender/apagar pantalla
def encender_pantalla():
    os.system(f"xrandr --output {pantalla} --auto")

def apagar_pantalla():
    os.system(f"xrandr --output {pantalla} --off")

# Inicializar la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
time.sleep(2)  # Esperar a que la cámara se estabilice

# Leer el primer cuadro
ret, frame1 = cap.read()
frame1 = frame1[RECORTE_Y1:RECORTE_Y2, RECORTE_X1:RECORTE_X2]  # Recortar imagen
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

# Variables de control
pantalla_encendida = False
ultimo_movimiento = time.time()
ventana_abierta = False

while True:
    # Leer nuevo cuadro
    ret, frame2 = cap.read()
    if not ret:
        break

    # Aplicar recorte para limitar el área visible
    frame2 = frame2[RECORTE_Y1:RECORTE_Y2, RECORTE_X1:RECORTE_X2]

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Calcular la diferencia entre cuadros
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Si hay movimiento detectado
    if len(contours) > 0:
        print("Movimiento detectado. Encendiendo pantalla...")
        ultimo_movimiento = time.time()

        # Encender pantalla si estaba apagada
        if not pantalla_encendida:
            encender_pantalla()
            pantalla_encendida = True

        # Abrir ventana si estaba cerrada
        if not ventana_abierta:
            cv2.namedWindow("Camara", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Camara", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            ventana_abierta = True

    # Si no hay movimiento por más de 3 segundos, cerrar ventana y apagar pantalla
    elif pantalla_encendida and (time.time() - ultimo_movimiento > 3):
        print("No hay movimiento. Apagando pantalla...")
        apagar_pantalla()
        pantalla_encendida = False

        # Cerrar ventana si está abierta
        if ventana_abierta:
            cv2.destroyWindow("Camara")
            ventana_abierta = False

    # Mostrar video en ventana solo si está abierta
    if ventana_abierta:
        cv2.imshow("Camara", frame2)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Actualizar el cuadro anterior
    gray1 = gray2

cap.release()
cv2.destroyAllWindows()
