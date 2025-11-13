#!/usr/bin/env python3
"""
Script para dividir dataset del Tello en train y valid
Busca en carpeta específica de Datasets del Tello
"""

import os
import shutil
import random
from pathlib import Path

# ============================================================
# CONFIGURACIÓN - Cambia esta ruta según tu sistema
# ============================================================
DATASETS_FOLDER = r"C:\Users\ejohn\Documents\Concentracion Drones\Tello\Datasets"

def find_datasets_in_folder(folder_path):
    """Buscar datasets en carpeta específica"""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"❌ Carpeta no existe: {folder_path}")
        return []
    
    print(f"🔍 Buscando datasets en:")
    print(f"   {folder_path}\n")
    
    datasets_found = []
    
    # Buscar carpetas con data.yaml
    try:
        for item in folder_path.iterdir():
            if item.is_dir():
                yaml_file = item / "data.yaml"
                if yaml_file.exists():
                    datasets_found.append(item)
                    print(f"  ✓ {item.name}")
    except PermissionError:
        print(f"❌ Sin permisos para acceder a {folder_path}")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []
    
    return datasets_found


def split_dataset(dataset_path, train_ratio=0.8):
    """Divide el dataset en train y valid"""
    print("\n" + "="*70)
    print("DIVIDIENDO DATASET EN TRAIN Y VALID")
    print("="*70)
    
    dataset_path = Path(dataset_path)
    
    # Verificar train
    train_images_path = dataset_path / "train" / "images"
    train_labels_path = dataset_path / "train" / "labels"
    
    if not train_images_path.exists():
        print(f"❌ No se encontró: {train_images_path}")
        return False
    
    # Obtener imágenes
    image_files = (list(train_images_path.glob("*.jpg")) + 
                   list(train_images_path.glob("*.png")) + 
                   list(train_images_path.glob("*.jpeg")))
    
    total_images = len(image_files)
    print(f"\n📊 Imágenes en train: {total_images}")
    
    if total_images == 0:
        print("❌ No hay imágenes")
        return False
    
    # Verificar labels
    if train_labels_path.exists():
        num_labels = len(list(train_labels_path.glob("*.txt")))
        print(f"📝 Labels encontrados: {num_labels}")
        
        if num_labels == 0:
            print("\n⚠️  ADVERTENCIA: No hay anotaciones (.txt)")
            print("   Debes etiquetar las imágenes primero en Roboflow")
            resp = input("   ¿Continuar? [s/N]: ")
            if resp.lower() != 's':
                return False
        elif num_labels != total_images:
            print(f"⚠️  Imágenes sin label: {total_images - num_labels}")
    else:
        print("❌ No existe train/labels")
        return False
    
    # Verificar si ya existe valid
    valid_images_path = dataset_path / "valid" / "images"
    if valid_images_path.exists() and len(list(valid_images_path.glob("*.*"))) > 0:
        print(f"\n⚠️  Ya existe carpeta 'valid' con {len(list(valid_images_path.glob('*.*')))} archivos")
        resp = input("   ¿Sobrescribir? [s/N]: ")
        if resp.lower() != 's':
            print("❌ Cancelado")
            return False
        
        # Limpiar valid existente
        shutil.rmtree(dataset_path / "valid")
        print("   ✓ Carpeta 'valid' eliminada")
    
    # Calcular división
    train_count = int(total_images * train_ratio)
    valid_count = total_images - train_count
    
    print(f"\n📈 División:")
    print(f"   Train: {train_count} imágenes ({train_ratio*100:.0f}%)")
    print(f"   Valid: {valid_count} imágenes ({(1-train_ratio)*100:.0f}%)")
    
    # Confirmar
    response = input("\n¿Continuar? [S/n]: ").strip().lower()
    if response and response not in ['s', 'si', 'sí', 'y', 'yes', '']:
        print("❌ Cancelado")
        return False
    
    # Mezclar y dividir
    random.seed(42)
    random.shuffle(image_files)
    
    valid_files = image_files[train_count:]
    
    # Crear carpetas valid
    valid_images_path = dataset_path / "valid" / "images"
    valid_labels_path = dataset_path / "valid" / "labels"
    
    valid_images_path.mkdir(parents=True, exist_ok=True)
    valid_labels_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Carpetas creadas:")
    print(f"   {valid_images_path.relative_to(dataset_path)}")
    print(f"   {valid_labels_path.relative_to(dataset_path)}")
    
    # Mover archivos
    print(f"\n📦 Moviendo {valid_count} archivos...")
    
    moved_images = 0
    moved_labels = 0
    missing_labels = []
    
    for img_file in valid_files:
        # Mover imagen
        dest_img = valid_images_path / img_file.name
        shutil.move(str(img_file), str(dest_img))
        moved_images += 1
        
        # Mover label
        label_name = img_file.stem + ".txt"
        label_file = train_labels_path / label_name
        
        if label_file.exists():
            dest_label = valid_labels_path / label_name
            shutil.move(str(label_file), str(dest_label))
            moved_labels += 1
        else:
            missing_labels.append(img_file.name)
        
        # Progreso
        if moved_images % 10 == 0 or moved_images == valid_count:
            percent = (moved_images / valid_count) * 100
            print(f"   [{percent:5.1f}%] {moved_images}/{valid_count}")
    
    print(f"\n✓ Movidos: {moved_images} imgs, {moved_labels} labels")
    
    if missing_labels:
        print(f"⚠️  {len(missing_labels)} imgs sin label")
    
    # Verificar resultado
    train_imgs_final = len(list(train_images_path.glob("*.*")))
    train_lbls_final = len(list(train_labels_path.glob("*.txt")))
    valid_imgs_final = len(list(valid_images_path.glob("*.*")))
    valid_lbls_final = len(list(valid_labels_path.glob("*.txt")))
    
    print(f"\n{'='*70}")
    print(f"✅ DIVISIÓN COMPLETADA")
    print(f"{'='*70}")
    print(f"Train: {train_imgs_final} imgs, {train_lbls_final} labels")
    print(f"Valid: {valid_imgs_final} imgs, {valid_lbls_final} labels")
    print(f"{'='*70}")
    
    print(f"\n💡 Dataset listo para entrenar:")
    print(f"   python train_tello_yolo.py")
    
    return True


def main():
    """Función principal"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*12 + "DIVIDIR DATASET - TELLO DRONE YOLO V11" + " "*18 + "║")
    print("║" + " "*20 + "Crear Train y Valid (80/20)" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    
    # Buscar en carpeta específica
    print(f"\n📂 Carpeta configurada:")
    print(f"   {DATASETS_FOLDER}")
    
    datasets = find_datasets_in_folder(DATASETS_FOLDER)
    
    if len(datasets) == 0:
        print(f"\n❌ No se encontraron datasets en la carpeta")
        print(f"\n💡 Verifica que:")
        print(f"   1. La carpeta existe")
        print(f"   2. Contiene subcarpetas con data.yaml")
        print(f"   3. La ruta en DATASETS_FOLDER es correcta")
        
        # Opción manual
        print(f"\n📂 ¿Ingresar ruta manualmente?")
        manual_path = input("Ruta (Enter para salir): ").strip().strip('"').strip("'")
        
        if manual_path:
            datasets = [Path(manual_path)]
        else:
            return
    
    # Seleccionar dataset
    dataset_path = None
    
    if len(datasets) == 1:
        dataset_path = datasets[0]
        print(f"\n✓ Dataset: {dataset_path.name}")
        resp = input("¿Usar este? [S/n]: ").strip().lower()
        if resp and resp not in ['s', 'si', 'sí', 'y', 'yes', '']:
            return
    
    elif len(datasets) > 1:
        print(f"\n📁 Datasets encontrados:")
        for i, ds in enumerate(datasets, 1):
            print(f"   {i}. {ds.name}")
        
        try:
            choice = int(input(f"\nElige (1-{len(datasets)}): "))
            if 1 <= choice <= len(datasets):
                dataset_path = datasets[choice - 1]
            else:
                print("❌ Opción inválida")
                return
        except ValueError:
            print("❌ Entrada inválida")
            return
    
    if dataset_path is None:
        print("\n❌ No se seleccionó dataset")
        return
    
    # Verificar dataset
    if not dataset_path.exists():
        print(f"\n❌ No existe: {dataset_path}")
        return
    
    yaml_file = dataset_path / "data.yaml"
    if not yaml_file.exists():
        print(f"\n❌ No se encontró data.yaml")
        return
    
    # Mostrar info
    try:
        import yaml
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        print(f"\n📄 Dataset info:")
        print(f"   📊 Clases: {data.get('nc', 'N/A')}")
        if 'names' in data:
            names = data['names']
            if isinstance(names, dict):
                names = list(names.values())
            print(f"   🏷️  Nombres: {', '.join(str(n) for n in names[:5])}")
            if len(names) > 5:
                print(f"             ... y {len(names)-5} más")
    except Exception as e:
        print(f"\n⚠️  No se pudo leer data.yaml: {e}")
    
    # Ratio de división
    print(f"\n⚙️  División:")
    print(f"   1. 80/20 (recomendado)")
    print(f"   2. 70/30")
    print(f"   3. 90/10")
    
    choice = input(f"\nElige [1]: ").strip()
    
    train_ratio = {
        '2': 0.7,
        '3': 0.9
    }.get(choice, 0.8)
    
    # Ejecutar
    success = split_dataset(dataset_path, train_ratio)
    
    if success:
        print(f"\n✅ Proceso completado")
        print(f"\n🚀 SIGUIENTE PASO:")
        print(f"   python train_tello_yolo.py")
        print(f"\n💡 Actualiza en train_tello_yolo.py:")
        print(f'   DATASET_PATH_LOCAL = r"{dataset_path}"')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelado")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()