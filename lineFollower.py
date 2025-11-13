import robomaster
from robomaster import robot
import time
import cv2 as cv2
import numpy as np

ROI_HEIGHT_RATIO = 0.75

def process_frame(frame, center_x_ref):
    """Procesa el frame para encontrar la línea blanca y calcular el error."""
    h, w, _ = frame.shape
    
    # 1. Definir la Región de Interés (ROI) en la parte inferior
    y0 = int(h * ROI_HEIGHT_RATIO)
    roi = frame[y0:, :].copy()
    roi_h, roi_w = roi.shape[:2]

    # 2. Preprocesado: gris -> desenfoque -> Umbralización (Otsu para blanco)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # THRESH_BINARY + THRESH_OTSU para detectar OBJETOS CLAROS (pipa blanca)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morfología para limpiar ruido
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 3. Buscar Contornos y Centroide
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_error = None # Indica que la línea se perdió
    
    if contours:
        # Encontrar el contorno más grande (asumiendo que es la línea)
        best_cnt = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(best_cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            
            # Dibujar centroide (solo para debug visual)
            cv2.circle(roi, (cx, int(M['m01'] / M['m00'])), 5, (0, 255, 0), -1)

            # Error: Distancia del centroide (cx) al centro de la imagen (w/2)
            # Nota: w es el ancho completo del frame, no del ROI
            best_error = float((w / 2) - cx)
            
    # Opcional: Mostrar imagen binaria para debug
    cv2.imshow("Binary ROI", binary)
    cv2.waitKey(1)
    
    return best_error, roi

if __name__ == '__main__':

    # Initialize the robot
    tl_drone = robot.Drone()
    tl_drone.initialize()
    tl_flight = tl_drone.flight

    # Get battery status
    tl_battery = tl_drone.battery
    battery_info = tl_battery.get_battery() 
    print("Drone battery soc: {0}".format(battery_info))

    # Get camera
    tl_camera = tl_drone.camera
    tl_camera.start_video_stream(display=False)

    try:
        while True:
            # Get the image from the camera
            frame = tl_camera.read_cv2_image(strategy="newest", timeout=5)


            frame = cv2.resize(frame, (640, 480))

            h, w, _ = frame.shape
            center_x_ref = w / 2 # Centro de la imagen
            
            error, _ = process_frame(frame, center_x_ref)

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    except Exception as e:
        print("An error occurred: ", e)
    finally:
        # Stop the camera and close the window
        tl_camera.stop_video_stream()
        cv2.destroyAllWindows()
        tl_drone.close()
        print("Drone closed.")
    
    tl_drone.close()