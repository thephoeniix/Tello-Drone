import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
import time
import threading

# ============================================================
# CONFIGURACIÓN - AJUSTA ESTOS VALORES
# ============================================================
MODEL_PATH = 'best.pt'
TARGET_CLASS = 'Pipes'

# ========== CONTROLADOR PD + VELOCIDAD ==========
# 🎛️ AJUSTA AQUÍ:
KP = 0.25         # Ganancia proporcional (error)
KD = 0.2         # Ganancia derivativa (cambio de error)
KV = 0.0          # Ganancia de velocidad (feed-forward)

DEADZONE = 60      
MAX_SPEED = 18     

# Tiempos
CORRECTION_TIME = 8.0
STABILIZATION_TIME = 1.0     # ↓ Reducido de 2.0 a 1.0

# Optimización
INFERENCE_SIZE = 224
FRAME_SKIP = 2

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

segment_counter = 0
total_segments = 2 * (7 + 6)

latest_frame = None
display_running = True
frame_counter = 0

# ============================================================
# DISPLAY THREAD
# ============================================================
def display_thread():
    global latest_frame, display_running
    while display_running:
        if latest_frame is not None:
            try:
                cv2.imshow("Tello - Ruta Hardcoded", latest_frame)
                cv2.waitKey(1)
            except:
                pass
        time.sleep(0.01)

display_thread_obj = threading.Thread(target=display_thread, daemon=True)
display_thread_obj.start()

dummy = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(dummy, "Esperando despegue...", (150, 240), 
           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
latest_frame = dummy

# ============================================================
# FUNCIÓN: AVANZAR 60CM
# ============================================================
def move_forward_30cm():
    global latest_frame
    
    print(f"    → Avanzando 60cm...")
    
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy, "AVANZANDO...", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        latest_frame = dummy
        
        tello.move_forward(30)
        print(f"    ✓ Avance completo")
        
        # Estabilización REDUCIDA
        print(f"    ⏳ Estabilizando {STABILIZATION_TIME}s...")
        for i in range(int(STABILIZATION_TIME * 10)):
            frame = frame_read.frame
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                remaining = STABILIZATION_TIME - (i * 0.1)
                cv2.putText(frame_bgr, f"Estabilizando... {remaining:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                latest_frame = frame_bgr
            time.sleep(0.1)
        
        print(f"    ✓ Estabilizado - Iniciando control")
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    time.sleep(0.2)  # Reducido de 0.3

# ============================================================
# FUNCIÓN: ROTAR 90° IZQUIERDA
# ============================================================
def rotate_left_90():
    print(f"    🔄 Rotando 90° izquierda...")
    try:
        tello.rotate_counter_clockwise(90)
        print(f"    ✓ Rotación completa")
        time.sleep(1.0)  # Reducido de 1.5
    except Exception as e:
        print(f"    ✗ Error: {e}")

# ============================================================
# CORRECCIÓN CON PD + VELOCIDAD
# ============================================================
def correction_phase():
    """
    Control PD + Feed-forward de velocidad (KV).
    """
    global segment_counter, latest_frame, frame_counter
    
    prev_error = 0
    prev_time = time.time()
    start_time = time.time()
    
    print(f"    → Control PD+V ({CORRECTION_TIME}s) [Kp:{KP} Kd:{KD} Kv:{KV}]")
    
    error_history = []
    velocity_estimate = 0
    correction_count = 0
    last_control = 0
    
    while time.time() - start_time < CORRECTION_TIME:
        frame = frame_read.frame
        if frame is None:
            time.sleep(0.02)
            continue
        
        # Frame skip
        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            time.sleep(0.02)
            continue
        
        current_time = time.time()
        dt = current_time - prev_time
        if dt < 0.001:
            dt = 0.001
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        center_x = w // 2
        
        results = model(frame_bgr, conf=0.3, verbose=False, imgsz=INFERENCE_SIZE)
        
        vis_frame = frame_bgr.copy()
        cv2.line(vis_frame, (center_x, 0), (center_x, h), (0, 255, 0), 2)
        
        elapsed = time.time() - start_time
        remaining = CORRECTION_TIME - elapsed
        cv2.putText(vis_frame, f"Seg {segment_counter}/{total_segments} | T: {remaining:.1f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        pipe_found = False
        
        if len(results) > 0 and results[0].masks is not None:
            result = results[0]
            classes = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names
            
            pipe_idx = None
            best_conf = 0
            for i, cls_id in enumerate(classes):
                if class_names[cls_id].lower() == TARGET_CLASS.lower():
                    if result.boxes.conf[i] > best_conf:
                        pipe_idx = i
                        best_conf = result.boxes.conf[i]
            
            if pipe_idx is not None:
                pipe_found = True
                
                mask = result.masks.data[pipe_idx].cpu().numpy()
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_bin = (mask_resized > 0.5).astype(np.uint8)
                
                overlay = vis_frame.copy()
                overlay[mask_bin > 0] = [0, 255, 255]
                vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
                
                M = cv2.moments(mask_bin)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    error = cx - center_x
                    
                    cv2.circle(vis_frame, (cx, cy), 10, (0, 0, 255), -1)
                    cv2.circle(vis_frame, (cx, cy), 13, (255, 255, 255), 2)
                    cv2.line(vis_frame, (center_x, cy), (cx, cy), (255, 0, 255), 2)
                    
                    # Filtro de error
                    error_history.append(error)
                    if len(error_history) > 4:
                        error_history.pop(0)
                    error_filtered = int(np.mean(error_history))
                    
                    # Estimación de velocidad lateral (derivada del error)
                    velocity_estimate = (error_filtered - prev_error) / dt
                    
                    cv2.putText(vis_frame, f"Err:{error_filtered} Vel:{velocity_estimate:.1f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # ========== CONTROL PD + VELOCIDAD ==========
                    if abs(error_filtered) < DEADZONE:
                        cv2.putText(vis_frame, "CENTRADO!", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        control = int(error_filtered * 0.08)
                        control = int(np.clip(control, -8, 8))
                        tello.send_rc_control(control, 0, 0, 0)
                        
                    else:
                        # P: Proporcional al error
                        p_term = KP * error_filtered
                        
                        # D: Derivativo (amortiguamiento)
                        d_term = KD * (error_filtered - prev_error) / dt
                        
                        # V: Feed-forward de velocidad
                        v_term = KV * velocity_estimate
                        
                        # Control total
                        control = p_term + d_term - v_term  # Negativo para compensar
                        
                        # Suavizar cambios bruscos
                        control_change = abs(control - last_control)
                        if control_change > 10:
                            control = last_control + np.sign(control - last_control) * 10
                        
                        control = int(np.clip(control, -MAX_SPEED, MAX_SPEED))
                        
                        direction = "←" if control < 0 else "→"
                        cv2.putText(vis_frame, f"Corr {direction} {abs(control)} (P:{p_term:.1f} D:{d_term:.1f} V:{v_term:.1f})", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        tello.send_rc_control(control, 0, 0, 0)
                        correction_count += 1
                        prev_error = error_filtered
                        last_control = control
        
        if not pipe_found:
            cv2.putText(vis_frame, "BUSCANDO PIPE...", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            tello.send_rc_control(0, 0, 0, 0)
            error_history.clear()
            velocity_estimate = 0
        
        latest_frame = vis_frame
        prev_time = current_time
        time.sleep(0.05)  # Ligeramente más rápido
    
    tello.send_rc_control(0, 0, 0, 0)
    print(f"    ✓ Control completo ({correction_count} ajustes)")
    time.sleep(0.2)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    try:
        print("\n=== PREVIEW CÁMARA ===")
        for _ in range(30):
            frame = frame_read.frame
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, "PRESIONA ENTER", 
                           (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                latest_frame = frame_bgr
            time.sleep(0.1)
        
        input()
        
        print("\n=== DESPEGUE ===")
        tello.takeoff()
        time.sleep(3)
        
        # CRÍTICO: Mantener dron activo
        print("Estabilización inicial (5s)...")
        for i in range(50):
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.1)
        
        print("Posicionamiento inicial...")
        move_forward_30cm()
        
        # RUTA HARDCODED
        for i in range(2):
            move_forward_30cm()
            correction_phase()
            
        rotate_left_90()
        print("\n✅ RUTA COMPLETADA")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        display_running = False
        for _ in range(5):
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.1)
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()
        print("Finalizado")