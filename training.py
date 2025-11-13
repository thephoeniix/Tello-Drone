#!/usr/bin/env python3
"""
Script para entrenar YOLOv11 con dataset del Tello Drone
Optimizado para MSI Cyborg 15 - Intel Core Ultra 7 + RTX 4060 (8GB VRAM)
Dataset: 7 clases (Cooler, Gas_station, Gas_tank, Oxxo, Pipes, Tree, Truck)
"""

import os
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import psutil
from datetime import datetime

# ============================================================
# CONFIGURACIÓN DEL USUARIO - MODIFICA AQUÍ
# ============================================================

# Ruta al dataset (CAMBIAR en Windows)
DATASET_PATH_LOCAL = r"C:\Users\ejohn\Documents\Concentracion Drones\Tello\Datasets\Tello Drone.v1-first_tello.yolov11"

# Configuración de entrenamiento
EPOCHS = 150  # Épocas de entrenamiento
IMG_SIZE = 640  # Tamaño de imagen
MODEL_SIZE = 'yolo11n.pt'  # yolo11n.pt (rápido), yolo11s.pt, yolo11m.pt
PROJECT_NAME = "tello_detection"
RUN_NAME = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Clases del dataset (ORDEN IMPORTANTE - debe coincidir con data.yaml)
EXPECTED_CLASSES = [
    'Cooler',
    'Gas_station',
    'Gas_tank',
    'Oxxo',
    'Pipes',
    'Tree',
    'Truck'
]

# ============================================================
# FUNCIONES DE SISTEMA
# ============================================================

def check_system():
    """Verificar configuración del sistema"""
    print("\n" + "="*60)
    print("VERIFICACIÓN DEL SISTEMA")
    print("="*60)
    
    # CPU
    print(f"💻 CPU:")
    print(f"   - Núcleos físicos: {psutil.cpu_count(logical=False)}")
    print(f"   - Núcleos lógicos: {psutil.cpu_count(logical=True)}")
    print(f"   - Uso actual: {psutil.cpu_percent()}%")
    
    # RAM
    ram = psutil.virtual_memory()
    print(f"\n🎯 RAM:")
    print(f"   - Total: {ram.total / (1024**3):.1f} GB")
    print(f"   - Disponible: {ram.available / (1024**3):.1f} GB")
    print(f"   - Uso: {ram.percent}%")
    
    # GPU
    print(f"\n🎮 GPU:")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA disponible")
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - CUDA Version: {torch.version.cuda}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   - Memoria GPU: {gpu_mem:.1f} GB VRAM")
        torch.cuda.empty_cache()
        print(f"   - Caché GPU limpiado")
    else:
        print(f"   ✗ CUDA no disponible - Se usará CPU")
    
    print("="*60 + "\n")
    return torch.cuda.is_available()


def verify_dataset_structure(dataset_path):
    """Verificar estructura del dataset"""
    print("\n" + "="*60)
    print("VERIFICANDO ESTRUCTURA DEL DATASET")
    print("="*60)
    
    dataset_path = Path(dataset_path)
    
    # Verificar data.yaml
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        print(f"✗ Error: No se encontró data.yaml en {dataset_path}")
        return False, None
    
    # Leer data.yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"✓ data.yaml encontrado")
    print(f"  - Clases: {data.get('nc', 'N/A')}")
    print(f"  - Nombres: {data.get('names', {})}")
    
    # Verificar carpetas
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    valid_images = dataset_path / "valid" / "images"
    valid_labels = dataset_path / "valid" / "labels"
    
    print(f"\nVerificando carpetas:")
    
    if train_images.exists():
        num_train_imgs = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
        num_train_lbls = len(list(train_labels.glob("*.txt"))) if train_labels.exists() else 0
        print(f"  ✓ Train: {num_train_imgs} imágenes, {num_train_lbls} labels")
    else:
        print(f"  ✗ Train: No encontrado")
        return False, None
    
    if valid_images.exists():
        num_valid_imgs = len(list(valid_images.glob("*.jpg"))) + len(list(valid_images.glob("*.png")))
        num_valid_lbls = len(list(valid_labels.glob("*.txt"))) if valid_labels.exists() else 0
        print(f"  ✓ Valid: {num_valid_imgs} imágenes, {num_valid_lbls} labels")
    else:
        print(f"  ⚠️  Valid: No encontrado - Ejecuta dividir_dataset.py primero")
    
    # Advertencias
    if num_train_imgs == 0:
        print(f"\n✗ Error: No hay imágenes en train/images")
        return False, None
    
    if num_train_lbls == 0:
        print(f"\n⚠️  ADVERTENCIA: No hay labels - ¿Ya anotaste las imágenes?")
        print(f"   Usa LabelImg o Roboflow para etiquetar antes de entrenar")
    
    print("="*60 + "\n")
    return True, yaml_path


def train_tello_yolo(dataset_yaml_path, epochs=150, img_size=640):
    """
    Entrenar modelo YOLOv11 para detección de objetos con Tello
    Optimizado para RTX 4060 (8GB VRAM)
    """
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO YOLO V11 - TELLO DATASET")
    print("="*60)
    
    # Configuración optimizada para RTX 4060
    training_config = {
        # Dataset
        'data': str(dataset_yaml_path),
        'epochs': epochs,
        'imgsz': img_size,
        
        # Hardware - Optimizado para RTX 4060 (8GB VRAM)
        'batch': 16,  # Ajustar a 8 si hay OOM
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        
        # Eficiencia
        'cache': True,
        'amp': True,  # Mixed Precision para RTX 40 series
        
        # Guardado
        'project': PROJECT_NAME,
        'name': RUN_NAME,
        'exist_ok': True,
        'save': True,
        'save_period': 20,
        'patience': 50,  # Early stopping
        'plots': True,
        'verbose': True,
        
        # Optimizador
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Data Augmentation (optimizado para detección general de objetos)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,  # Rotación moderada
        'translate': 0.1,
        'scale': 0.5,
        'shear': 2.0,     # Pequeña deformación
        'perspective': 0.0,
        'flipud': 0.0,    # Sin flip vertical
        'fliplr': 0.5,    # Flip horizontal
        'mosaic': 1.0,
        'mixup': 0.1,     # Pequeño mixup
        'copy_paste': 0.0,
        
        # Otros
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
    }
    
    print(f"Modelo: {MODEL_SIZE}")
    print(f"Épocas: {epochs}")
    print(f"Tamaño imagen: {img_size}x{img_size}")
    print(f"Batch size: {training_config['batch']}")
    print(f"Dispositivo: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print(f"Workers: {training_config['workers']}")
    print(f"Clases a detectar: {', '.join(EXPECTED_CLASSES)}")
    print("="*60)
    
    try:
        print(f"\nCargando modelo base: {MODEL_SIZE}")
        model = YOLO(MODEL_SIZE)
        
        print("Iniciando entrenamiento...")
        print("(Presiona Ctrl+C para detener)\n")
        
        # Entrenar
        results = model.train(**training_config)
        
        print("\n" + "="*60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*60)
        print(f"📁 Mejores pesos: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
        print(f"📁 Últimos pesos: {PROJECT_NAME}/{RUN_NAME}/weights/last.pt")
        print(f"📊 Resultados: {PROJECT_NAME}/{RUN_NAME}/")
        print("="*60 + "\n")
        
        return model, results
        
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido")
        return None, None
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        
        if "out of memory" in str(e).lower():
            print("\n💡 Solución para OOM:")
            print("  - Reduce batch de 16 a 8")
            print("  - Reduce img_size de 640 a 416")
            print("  - Desactiva cache")
        
        import traceback
        traceback.print_exc()
        return None, None


def evaluate_model(model, dataset_yaml_path):
    """Evaluar modelo"""
    print("\n" + "="*60)
    print("EVALUANDO MODELO")
    print("="*60)
    
    try:
        results = model.val(data=str(dataset_yaml_path), plots=True)
        
        print("\n📊 MÉTRICAS:")
        print(f"  mAP50:       {results.box.map50:.4f} ({results.box.map50*100:.2f}%)")
        print(f"  mAP50-95:    {results.box.map:.4f} ({results.box.map*100:.2f}%)")
        print(f"  Precision:   {results.box.mp:.4f} ({results.box.mp*100:.2f}%)")
        print(f"  Recall:      {results.box.mr:.4f} ({results.box.mr*100:.2f}%)")
        
        if results.box.map50 > 0.8:
            print("\n  ✓ Excelente rendimiento!")
        elif results.box.map50 > 0.6:
            print("\n  ⚠️  Buen rendimiento, puede mejorar")
        else:
            print("\n  ✗ Rendimiento bajo, considera más épocas")
        
        print("="*60 + "\n")
        return results
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    """Función principal"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*8 + "ENTRENAMIENTO YOLO V11 - TELLO DRONE DATASET" + " "*6 + "║")
    print("║" + " "*5 + "Detección: Cooler, Gas_station, Gas_tank, etc." + " "*6 + "║")
    print("╚" + "="*58 + "╝")
    
    # 1. Verificar sistema
    has_gpu = check_system()
    
    if not has_gpu:
        print("⚠️  No se detectó GPU")
        resp = input("¿Continuar con CPU? [s/N]: ")
        if resp.lower() != 's':
            return
    
    # 2. Verificar dataset
    if not os.path.exists(DATASET_PATH_LOCAL):
        print(f"\n✗ Error: Dataset no encontrado en:")
        print(f"   {DATASET_PATH_LOCAL}")
        print(f"\n💡 Solución:")
        print(f"   1. Transfiere la carpeta datasetDron a Windows")
        print(f"   2. Actualiza DATASET_PATH_LOCAL en este script")
        return
    
    valid, yaml_path = verify_dataset_structure(DATASET_PATH_LOCAL)
    
    if not valid:
        print("\n✗ Estructura del dataset incorrecta")
        return
    
    # 3. Confirmar
    print(f"\n{'='*60}")
    print("CONFIGURACIÓN FINAL")
    print(f"{'='*60}")
    print(f"Dataset: {DATASET_PATH_LOCAL}")
    print(f"Clases: {len(EXPECTED_CLASSES)} - {', '.join(EXPECTED_CLASSES)}")
    print(f"Modelo: {MODEL_SIZE}")
    print(f"Épocas: {EPOCHS}")
    print(f"Imagen: {IMG_SIZE}x{IMG_SIZE}")
    print(f"{'='*60}\n")
    
    input("Presiona Enter para iniciar (Ctrl+C para cancelar)...")
    
    # 4. Entrenar
    model, results = train_tello_yolo(yaml_path, epochs=EPOCHS, img_size=IMG_SIZE)
    
    if model is None:
        return
    
    # 5. Evaluar
    evaluate_model(model, yaml_path)
    
    # 6. Resumen
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)
    print(f"📁 Mejores pesos: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"\n💡 Para usar el modelo:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{PROJECT_NAME}/{RUN_NAME}/weights/best.pt')")
    print(f"  results = model.predict('imagen.jpg')")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()