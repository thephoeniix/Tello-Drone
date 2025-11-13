import robomaster
from robomaster import robot
import cv2
import numpy as np

def nothing(x):
    pass

# Crear ventana de sliders
cv2.namedWindow("Sliders")
cv2.createTrackbar("H min", "Sliders", 80, 179, nothing)
cv2.createTrackbar("H max", "Sliders", 100, 179, nothing)
cv2.createTrackbar("S min", "Sliders", 100, 255, nothing)
cv2.createTrackbar("S max", "Sliders", 255, 255, nothing)
cv2.createTrackbar("V min", "Sliders", 100, 255, nothing)
cv2.createTrackbar("V max", "Sliders", 255, 255, nothing)

# Inicializar dron
tl_drone = robot.Drone()
tl_drone.initialize()
camera = tl_drone.camera
camera.start_video_stream(display=False)

try:
    while True:
        frame = camera.read_cv2_image(strategy="newest", timeout=5)
        frame = cv2.resize(frame, (640, 480))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Leer valores HSV de sliders
        h_min = cv2.getTrackbarPos("H min", "Sliders")
        h_max = cv2.getTrackbarPos("H max", "Sliders")
        s_min = cv2.getTrackbarPos("S min", "Sliders")
        s_max = cv2.getTrackbarPos("S max", "Sliders")
        v_min = cv2.getTrackbarPos("V min", "Sliders")
        v_max = cv2.getTrackbarPos("V max", "Sliders")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Mostrar imágenes
        cv2.imshow("Original", frame)
        cv2.imshow("Mascara", mask)
        cv2.imshow("Resultado", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    camera.stop_video_stream()
    tl_drone.close()
    cv2.destroyAllWindows()
    print("Dron desconectado")
