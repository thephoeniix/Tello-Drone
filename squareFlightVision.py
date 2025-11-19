import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
import time

# ============================================================
# CONFIGURACIÓN
# ============================================================
MODEL_PATH = 'best.pt'
TARGET_CLASS = 'Pipes'

# Control PD para corrección
KP = 0.30
KD = 0.15
DEADZONE = 30
MAX_SPEED = 40

# ============================================================
# SETUP
# ============================================================
print("Conectando...")
tello = Tello()
tello.connect()
print(f"Batería: {tello.get_battery()}%")

tello.streamon()
time.sleep(2)

print("Cargando modelo...")
model = YOLO(MODEL_PATH)

frame_read = tello.get_frame_read()

# Contador de segmentos
segment_counter = 0
total_segments = 2 * (6 + 7)  # 2 vueltas * (6 + 7 segmentos)

# ============================================================
# FUNCIÓN DE CORRECCIÓN SIMPLE
# ============================================================
def correction_after_forward(timeout=3.0):
    """
    Corrección lateral RÁPIDA después de cada forward.
    Solo usa el centroide del Pipe para centrar el dron.
    """
    global segment_counter
    
    prev_error = 0
    start_time = time.time()
    centered_frames = 0
    
    print(f"  → Corrigiendo segmento {segment_counter}/{total_segments}...")
    
    while time.time() - start_time < timeout:
        frame = frame_read.frame
        if frame is None:
            time.sleep(0.05)
            continue
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        center_x = w // 2
        
        # Detección YOLO optimizada
        results = model(frame_bgr, conf=0.3, verbose=False, imgsz=416)
        
        # Visualización
        vis_frame = frame_bgr.copy()
        cv2.line(vis_frame, (center_x, 0), (center_x, h), (0, 255, 0), 1)
        
        # Info en pantalla
        cv2.putText(vis_frame, f"Segmento: {segment_counter}/{total_segments}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        pipe_found = False
        
        if len(results) > 0 and results[0].masks is not None:
            result = results[0]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names
            
            # Filtrar SOLO Pipes
            pipe_idx = None
            for i, cls_id in enumerate(classes):
                if class_names[cls_id].lower() == TARGET_CLASS.lower():
                    pipe_idx = i
                    break
            
            if pipe_idx is not None:
                pipe_found = True
                
                # Procesar máscara
                mask = result.masks.data[pipe_idx].cpu().numpy()
                mask_resized = cv2.resize(mask, (w, h))
                mask_bin = (mask_resized > 0.5).astype(np.uint8)
                
                # Calcular centroide
                M = cv2.moments(mask_bin)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    error = cx - center_x
                    
                    # Dibujar
                    colored_mask = np.zeros_like(frame_bgr)
                    colored_mask[mask_bin > 0] = [0, 255, 0]
                    vis_frame = cv2.addWeighted(vis_frame, 0.7, colored_mask, 0.3, 0)
                    cv2.circle(vis_frame, (cx, cy), 8, (0, 0, 255), -1)
                    cv2.line(vis_frame, (center_x, cy), (cx, cy), (255, 0, 255), 2)
                    
                    # Info
                    cv2.putText(vis_frame, f"Error: {error}px", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Verificar si está centrado
                    if abs(error) < DEADZONE:
                        centered_frames += 1
                        cv2.putText(vis_frame, "CENTRADO!", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        if centered_frames >= 2:  # 2 frames centrado = listo
                            cv2.imshow("Tello - Ruta Hardcoded", vis_frame)
                            cv2.waitKey(1)
                            tello.send_rc_control(0, 0, 0, 0)
                            print("  ✓ Centrado")
                            return
                    else:
                        centered_frames = 0
                        
                        # Control PD
                        d_error = error - prev_error
                        control = KP * error + KD * d_error
                        control = int(np.clip(control, -MAX_SPEED, MAX_SPEED))  # ← CONVERSIÓN A INT
                        
                        direction = "←" if control < 0 else "→"
                        cv2.putText(vis_frame, f"Ajustando {direction} {abs(control)}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Enviar comando
                        tello.send_rc_control(control, 0, 0, 0)
                        prev_error = error
        
        if not pipe_found:
            cv2.putText(vis_frame, "Buscando Pipe...", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Tello - Ruta Hardcoded", vis_frame)
        cv2.waitKey(10)  # ← Aumentado a 10ms para dar tiempo al display
        
        time.sleep(0.05)
    
    # Timeout alcanzado
    tello.send_rc_control(0, 0, 0, 0)
    print("  ⏱ Timeout")

# ============================================================
# MAIN - RUTA HARDCODEADA
# ============================================================
if __name__ == "__main__":
    try:
        # Mostrar ventana inicial
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "Iniciando...", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Tello - Ruta Hardcoded", dummy_frame)
        cv2.waitKey(1000)  # Mostrar 1 segundo
        
        print("\n=== DESPEGUE ===")
        tello.takeoff()
        time.sleep(3)
        
        print("Posicionando...")
        tello.move_forward(40)
        time.sleep(1)
        
        # ========================================
        # RUTA HARDCODEADA: 2 VUELTAS
        # ========================================
        for vuelta in range(2):
            print(f"\n{'='*50}")
            print(f"VUELTA {vuelta + 1}/2")
            print(f"{'='*50}")
            
            # --- 6 SEGMENTOS ---
            print("\n[Tramo 1: 6 segmentos]")
            for i in range(6):
                segment_counter += 1
                print(f"\nSegmento {segment_counter}/{total_segments}")
                tello.move_forward(60)
                time.sleep(2.0)
                correction_after_forward()
            
            # --- GIRO 90° ---
            print("\n🔄 Girando 90° izquierda...")
            tello.rotate_counter_clockwise(90)
            time.sleep(4)
            
            # --- 7 SEGMENTOS ---
            print("\n[Tramo 2: 7 segmentos]")
            for i in range(7):
                segment_counter += 1
                print(f"\nSegmento {segment_counter}/{total_segments}")
                tello.move_forward(60)
                time.sleep(2.0)
                correction_after_forward()
            
            # --- GIRO 90° ---
            print("\n🔄 Girando 90° izquierda...")
            tello.rotate_counter_clockwise(90)
            time.sleep(4)
        
        print("\n✅ RUTA COMPLETADA")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nAterrizando...")
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()
        print("Finalizado")