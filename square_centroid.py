#!/usr/bin/env python3
"""
SQUARE PATTERN + CENTROID CORRECTION
"""

import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
import time


class SquareCentroidFollower:
    def __init__(self, model_path='best.pt', target_class='Pipes'):
        self.tello = Tello()
        self.model = None
        self.model_path = model_path
        self.target_class = target_class
        self.target_class_id = None
        
        # Control PD
        self.kp_x = 0.3
        self.kd_x = 0.8
        self.kp_y = 0.3
        self.kd_y = 0.8
        
        self.forward_speed = 15
        self.max_speed = 25
        self.conf_threshold = 0.3
        self.deadzone_x = 80
        self.deadzone_y = 80
        
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.prev_time = time.time()
        self.error_x_filtered = 0
        self.error_y_filtered = 0
        self.filter_alpha = 0.7
        
        # Recuperación de línea perdida
        self.frames_without_detection = 0
        self.max_frames_before_search = 15
        
        self.flying = False
        self.current_segment = 0
        self.pattern = self.build_pattern()
        
    def build_pattern(self):
        pattern = []
        # Mínimo 20cm para movimientos del Tello
        pattern.append(('down', 20))
        
        for pasada in range(2):
            for i in range(7):
                pattern.append(('forward', 60))
                if i % 2 == 1 and i < 6:
                    pattern.append(('rotate', 5))
            
            pattern.append(('rotate', 75))
            
            for i in range(6):
                pattern.append(('forward', 60))
                if i % 2 == 1 and i < 5:
                    pattern.append(('rotate', 5))
            
            if pasada < 1:
                pattern.append(('rotate', 75))
        
        return pattern
    
    def load_model(self):
        print(f'Cargando modelo: {self.model_path}')
        try:
            self.model = YOLO(self.model_path)
            
            for class_id, class_name in self.model.names.items():
                if class_name.lower() == self.target_class.lower():
                    self.target_class_id = class_id
                    break
            
            print(f'Modelo cargado')
            return True
        except Exception as e:
            print(f'Error: {e}')
            return False
    
    def connect(self):
        print('Conectando...')
        try:
            self.tello.connect()
            battery = self.tello.get_battery()
            print(f'Conectado - Bateria: {battery}%')
            return battery >= 20
        except Exception as e:
            print(f'Error: {e}')
            return False
    
    def start_stream(self):
        try:
            self.tello.streamon()
            print('Stream iniciado - esperando video...')
            time.sleep(4)
            return True
        except Exception as e:
            print(f'Error: {e}')
            return False
    
    def calculate_centroid(self, mask):
        moments = cv2.moments(mask)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return (cx, cy)
        return None
    
    def detect_line_and_centroid(self, frame):
        h, w = frame.shape[:2]
        
        if frame is None or frame.size == 0:
            return None, frame
        
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        if not results or len(results) == 0:
            return None, frame
        
        result = results[0]
        
        if result.masks is None or len(result.masks) == 0:
            return None, frame
        
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        valid_detections = []
        if self.target_class_id is not None:
            for idx, cls in enumerate(classes):
                if cls == self.target_class_id:
                    valid_detections.append(idx)
        
        if len(valid_detections) == 0:
            return None, frame
        
        valid_confidences = [confidences[idx] for idx in valid_detections]
        best_idx = valid_detections[np.argmax(valid_confidences)]
        
        mask = result.masks.data[best_idx].cpu().numpy()
        mask_resized = cv2.resize(mask, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
        
        centroid = self.calculate_centroid(mask_binary)
        
        annotated = frame.copy()
        colored_mask = np.zeros_like(frame)
        colored_mask[mask_binary > 0] = (0, 255, 0)
        annotated = cv2.addWeighted(annotated, 0.7, colored_mask, 0.3, 0)
        
        if centroid:
            cv2.circle(annotated, centroid, 10, (0, 0, 255), -1)
            center = (w // 2, h // 2)
            cv2.line(annotated, center, centroid, (255, 0, 255), 2)
        
        center = (w // 2, h // 2)
        cv2.drawMarker(annotated, center, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
        
        return centroid, annotated
    
    def calculate_control_commands(self, centroid, frame_shape):
        h, w = frame_shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt < 0.001:
            dt = 0.001
        
        if centroid is None:
            self.frames_without_detection += 1
            
            # Si llevamos muchos frames sin detección, subir para buscar
            if self.frames_without_detection > self.max_frames_before_search:
                print(f'  ! Perdida {self.frames_without_detection} frames - SUBIENDO')
                return 0, 15, 5, 0
            else:
                return 0, 0, int(self.forward_speed), 0
        
        if self.frames_without_detection > 0:
            print(f'  ✓ Linea recuperada ({self.frames_without_detection} frames)')
        self.frames_without_detection = 0
        
        cx, cy = centroid
        
        error_x = cx - center_x
        error_y = cy - center_y
        
        self.error_x_filtered = (self.filter_alpha * self.error_x_filtered + 
                                 (1 - self.filter_alpha) * error_x)
        self.error_y_filtered = (self.filter_alpha * self.error_y_filtered + 
                                 (1 - self.filter_alpha) * error_y)
        
        error_x = self.error_x_filtered
        error_y = self.error_y_filtered
        
        if abs(error_x) < self.deadzone_x:
            error_x = 0
        if abs(error_y) < self.deadzone_y:
            error_y = 0
        
        d_error_x = (error_x - self.prev_error_x) / dt
        d_error_y = (error_y - self.prev_error_y) / dt
        
        left_right = int(self.kp_x * error_x + self.kd_x * d_error_x)
        up_down = -int(self.kp_y * error_y + self.kd_y * d_error_y)
        
        left_right = int(np.clip(left_right, -self.max_speed, self.max_speed))
        up_down = int(np.clip(up_down, -self.max_speed, self.max_speed))
        
        forward = int(self.forward_speed)
        yaw = 0
        
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_time = current_time
        
        return left_right, up_down, forward, yaw
    
    def draw_overlay(self, frame, centroid, segment_info, commands):
        h, w = frame.shape[:2]
        
        seg_text = f'Seg: {segment_info}'
        cv2.putText(frame, seg_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if centroid:
            centroid_text = f'Centro: ({centroid[0]}, {centroid[1]})'
            color = (0, 255, 0)
        else:
            centroid_text = f'Sin deteccion ({self.frames_without_detection})'
            if self.frames_without_detection > self.max_frames_before_search:
                color = (0, 0, 255)
            else:
                color = (0, 165, 255)
        
        cv2.putText(frame, centroid_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        lr, ud, fw, _ = commands
        cmd_text = f'LR:{lr:+d} UD:{ud:+d} FW:{fw:+d}'
        cv2.putText(frame, cmd_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        return frame
    
    def execute_forward_segment(self, distance_cm, frame_read):
        duration = distance_cm / 20.0
        start_time = time.time()
        
        last_valid_frame = None
        self.frames_without_detection = 0
        
        while (time.time() - start_time) < duration:
            frame = frame_read.frame
            
            if frame is None or frame.size == 0:
                if last_valid_frame is not None:
                    frame = last_valid_frame
                else:
                    time.sleep(0.05)
                    continue
            else:
                last_valid_frame = frame.copy()
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            centroid, annotated = self.detect_line_and_centroid(frame_bgr)
            
            left_right, up_down, forward, yaw = self.calculate_control_commands(
                centroid, frame_bgr.shape
            )
            
            try:
                self.tello.send_rc_control(left_right, forward, up_down, yaw)
            except:
                pass
            
            display = self.draw_overlay(
                annotated, centroid, 
                f'{self.current_segment}/{len(self.pattern)} - FWD {distance_cm}cm',
                (left_right, up_down, forward, yaw)
            )
            
            cv2.imshow('Square Centroid', display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
            
            time.sleep(0.05)
        
        self.tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.3)
        
        return True
    
    def execute_pattern(self, frame_read):
        print(f'\nEjecutando patron: {len(self.pattern)} acciones\n')
        
        for action, value in self.pattern:
            self.current_segment += 1
            
            if action == 'forward':
                print(f'[{self.current_segment}/{len(self.pattern)}] Forward {value}cm')
                if not self.execute_forward_segment(value, frame_read):
                    return False
                    
            elif action == 'rotate':
                print(f'[{self.current_segment}/{len(self.pattern)}] Rotate {value}')
                if value > 0:
                    self.tello.rotate_clockwise(value)
                else:
                    self.tello.rotate_counter_clockwise(abs(value))
                time.sleep(1)
                
                self.prev_error_x = 0
                self.prev_error_y = 0
                self.error_x_filtered = 0
                self.error_y_filtered = 0
                self.frames_without_detection = 0
                
            elif action == 'down':
                print(f'[{self.current_segment}/{len(self.pattern)}] Down {value}cm')
                self.tello.move_down(value)
                time.sleep(2)
        
        print('\nPatron completado!\n')
        return True
    
    def takeoff(self):
        if not self.flying:
            print('Despegando...')
            try:
                self.tello.takeoff()
                time.sleep(5)
                
                self.flying = True
                print('En el aire - listo')
                return True
            except Exception as e:
                print(f'Error: {e}')
                return False
        return True
    
    def land(self):
        if self.flying:
            print('Aterrizando...')
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)
                self.tello.land()
                time.sleep(2)
                self.flying = False
                print('Aterrizado')
                return True
            except Exception as e:
                print(f'Error: {e}')
                return False
        return True
    
    def run(self):
        print('\nSQUARE PATTERN + CENTROID')
        print('[T]=Despegar [L]=Aterrizar [Q]=Salir')
        print('Asegurate de ver la ventana de video antes de despegar!\n')
        
        frame_read = self.tello.get_frame_read()
        
        print('Esperando stream de video...')
        valid_frames = 0
        attempts = 0
        max_attempts = 200
        
        while valid_frames < 20 and attempts < max_attempts:
            frame = frame_read.frame
            if frame is not None and frame.size > 0:
                valid_frames += 1
                # Mostrar frame para verificar que funciona
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('Square Centroid', frame_bgr)
                cv2.waitKey(1)
            attempts += 1
            time.sleep(0.1)
        
        if valid_frames < 20:
            print('ERROR: No se pudo obtener stream de video estable')
            return
        
        print(f'Stream OK - {valid_frames} frames validos\n')
        input('Presiona ENTER cuando veas la ventana de video...')
        
        pattern_started = False
        
        while True:
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.05)
                continue
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if not self.flying or not pattern_started:
                centroid, annotated = self.detect_line_and_centroid(frame_bgr)
                display = annotated.copy()
                
                h, w = display.shape[:2]
                
                if not self.flying:
                    status = 'EN TIERRA - Presiona T para despegar'
                else:
                    status = 'LISTO - Patron iniciara'
                
                cv2.putText(display, status, (10, h - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Square Centroid', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            
            elif key == ord('t'):
                if not self.flying:
                    if self.takeoff():
                        pattern_started = True
                        self.execute_pattern(frame_read)
                        self.land()
                        pattern_started = False
            
            elif key == ord('l'):
                if self.flying:
                    self.tello.send_rc_control(0, 0, 0, 0)
                    self.land()
                    pattern_started = False
        
        cv2.destroyAllWindows()
        if self.flying:
            self.land()
    
    def stop_stream(self):
        try:
            self.tello.streamoff()
        except:
            pass
    
    def disconnect(self):
        try:
            self.tello.end()
        except:
            pass


def main():
    follower = SquareCentroidFollower(model_path='best.pt', target_class='Pipes')
    
    try:
        if not follower.load_model():
            return
        if not follower.connect():
            return
        if not follower.start_stream():
            return
        
        follower.run()
        
    except KeyboardInterrupt:
        print('\nInterrumpido')
    except Exception as e:
        print(f'\nError: {e}')
    finally:
        if follower.flying:
            follower.land()
        follower.stop_stream()
        follower.disconnect()


if __name__ == '__main__':
    main()