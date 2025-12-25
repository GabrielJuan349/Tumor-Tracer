"""
🧠 Tumor Tracer AI - Pipeline de Detección de Tumores Cerebrales

## 📌 Información del Proyecto
- **Proyecto:** Tumor-Tracer (Machine Learning Clásico)
- **Autores:** Aimé Moral & Gabriel Juan
- **Fecha:** Diciembre 2025
- **Versión:** 2.0
- **GitHub:** [GabrielJuan349/Tumor-Tracer](https://github.com/GabrielJuan349/Tumor-Tracer)

## 🎯 Objetivo
Segmentación automática de tumores cerebrales (gliomas) en imágenes de resonancia magnética (MRI) 
usando **Random Forest** con ingeniería de características avanzada.

## 📊 Dataset
- **Fuente:** [LGG MRI Segmentation (Kaggle)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- **Imágenes:** 3,929 MRI FLAIR + máscaras binarias
- **Resolución:** 256x256 píxeles
- **Formato:** TIF (8-bit grayscale/RGB)

## 🔧 Stack Tecnológico
- `scikit-learn` - Random Forest Classifier
- `OpenCV` - Procesamiento de imágenes
- `NumPy/Pandas` - Manipulación de datos
- `Matplotlib` - Visualización

## 📈 Métricas Objetivo
- **Sensibilidad (Recall):** >85% (detectar todos los tumores)
- **Precisión:** >75% (minimizar falsas alarmas)
- **Dice Score:** >70% (calidad de segmentación)

## 🧬 Metodología

### Pipeline de 5 Etapas
1. **Preprocesamiento Avanzado**: CLAHE, Denoise, Alineación PCA.
2. **Ingeniería de Características**: 21 features (Color, Textura, Espacial, Simetría, Interacción).
3. **Estrategia de Muestreo**: Ratio 1:3 (Tumor:Fondo) para balancear clases.
4. **Modelo**: Random Forest (100 árboles, max_depth=30).
5. **Post-Procesamiento**: Morfología, Filtro de Área, ROI Brain Mask.
"""

# !pip install opencv-python pandas numpy scikit-learn matplotlib tqdm

# ==========================================
# IMPORTS Y CONFIGURACIÓN DEL ENTORNO
# ==========================================

# --- Utilidades Core ---
import os                    # Gestión de rutas y archivos
import sys                   # Detección de entorno (Kaggle/Local)
import glob                  # Búsqueda recursiva de archivos
import random                # Selección aleatoria de imágenes
import time                  # Medición de tiempos
import gc                    # Gestión de memoria (liberación manual)
import shutil                # Operaciones de carpetas
import datetime              # Timestamp para logs
import joblib                # Persistencia de modelos (Save/Load)
from collections import defaultdict # Para agrupar metricas por paciente

# --- Análisis de Datos ---
import pandas as pd          # DataFrame para features (21 columnas × N píxeles)
import numpy as np           # Operaciones vectorizadas (10x más rápido que Python puro)

# --- Procesamiento de Imágenes ---
import cv2                   # OpenCV: CLAHE, morfología, PCA
from scipy import ndimage as nd  # Filtros gaussianos y operaciones ND

# --- Machine Learning ---
from sklearn.ensemble import RandomForestClassifier  # Modelo principal
from sklearn.model_selection import (
    train_test_split,        # Split 80/20 train/test
    cross_validate,          # K-Fold con múltiples métricas
    KFold,                   # Generador de folds
    GroupShuffleSplit        # Split respetando grupos (pacientes)
)
from sklearn.metrics import (
    accuracy_score,          # Métrica general (no ideal para segmentación)
    f1_score,                # Balance Precision/Recall
    precision_score,         # Confianza de las predicciones
    recall_score             # Sensibilidad (crítico en medicina)
)

# --- Visualización ---
import matplotlib            # Control de backend (Agg para headless)
matplotlib.use('Agg')        # Sin ventanas emergentes (estabilidad en Windows/Kaggle)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # Ya no se usa para 3D interactivo
import plotly.graph_objects as go # Para plots 3D interactivos

# --- Progress Bars ---
from tqdm import tqdm        # Barras de progreso para loops largos

# --- Advertencias ---
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suprimir warnings de sklearn

print("[INFO] Imports cargados correctamente")
print(f"   - OpenCV: {cv2.__version__}")
print(f"   - NumPy:  {np.__version__}")
print(f"   - scikit-learn: {__import__('sklearn').__version__}")


# ⚙️ Configuración del Experimento
#
# Para abordar el desbalanceo de clases y la variabilidad de las resonancias magnéticas, 
# hemos definido los siguientes hiperparámetros estratégicos:
#
# ### 🌲 Configuración del Random Forest
# *   **`n_estimators = 100`**: Cantidad de árboles de decisión.
# *   **`max_depth = 30`**: Profundidad máxima para evitar overfitting.
# *   **`class_weight = {0: 1.0, 1: 1.5}`**: Penaliza más los falsos negativos (crítico en medicina).
#
# ### ⚖️ Estrategia de Muestreo (Subsampling)
# *   **Ratio 1:3**: Por cada píxel de tumor, seleccionamos aleatoriamente solo 3 píxeles de fondo.

# ==========================================
# 1. CONFIGURACIÓN Y CONSTANTES
# ==========================================
RANDOM_STATE = 42
RF_ESTIMATORS = 100
RF_MAX_DEPTH = 30
RF_CLASS_WEIGHT = {0: 1, 1: 1.5} # Peso 1.5 a Tumor para priorizar sensibilidad sin disparar FPs
SUBSAMPLE_RATIO = 3  # Ratio 1 pixel tumor : 3 pixeles fondo
CV_FOLDS = 7
NUM_IMAGES = 4000
MODEL_FILENAME = "rf_model.joblib"

PROJECT_ROOT = os.getcwd()

print(f"[INFO] Proyecto raíz establecido en: {PROJECT_ROOT}")


# 📂 Preparación y Migración del Dataset
#
# El dataset original de Kaggle (LGG MRI Segmentation) viene organizado en una estructura anidada.
# Realizamos una **migración a una estructura plana** (`dataset_plano`) para simplificar.

# Redefinicion de las carpetas
# Rutas de configuración
ORIGINAL_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "kaggle_3m")  # Donde están las carpetas ahora
NEW_FLAT_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_plano")      # Donde las quieres poner

# 1. Crear la carpeta nueva si no existe
if not os.path.exists(NEW_FLAT_PATH):
    os.makedirs(NEW_FLAT_PATH)
    print(f"Carpeta creada: {NEW_FLAT_PATH}")

print("--- Iniciando migración de archivos ---")

# 2. Buscar todos los archivos .tif en subcarpetas
# En Windows usamos '**/*.tif' con recursive=True
files = glob.glob(os.path.join(ORIGINAL_DATASET_PATH, '**', '*.tif'), recursive=True)

count = 0
for file_path in files:
    # Obtener solo el nombre del archivo (ej: TCGA_CS_4941_19960909_1.tif)
    filename = os.path.basename(file_path)
    
    # Definir la nueva ruta de destino
    dst_path = os.path.join(NEW_FLAT_PATH, filename)
    
    # Evitar sobrescribir si ya existe
    if not os.path.exists(dst_path):
        # Copiar el archivo (usamos copy2 para preservar metadatos)
        shutil.copy2(file_path, dst_path)
        count += 1

print(f"--- Proceso terminado ---")
print(f"Se han copiado {count} imágenes a '{NEW_FLAT_PATH}'")


# 🛠️ Funciones de Utilidad y Gestión de Archivos
#
# ### 1. Lectura Robusta (`cv2_imread_unicode`)
# Lectura de archivos con caracteres especiales en Windows.
#
# ### 2. Organización de Resultados (`limpiar_directorio_resultados`)
# Reinicia la carpeta `results/` en cada ejecución.
#
# ### 3. Bitácora de Experimentos (`log_experiment_to_md`)
# Anexa métricas al archivo `experiment_history.md`.

# ==========================================
# 2. FUNCIONES DE LECTURA E I/O
# ==========================================
def cv2_imread_unicode(path, flag=cv2.IMREAD_COLOR):
    """
    Lee imágenes con rutas que contienen caracteres Unicode (ñ, á, etc.).
    
    PROBLEMA: cv2.imread() falla en Windows con rutas como "C:/Año2024/María.tif"
    SOLUCIÓN: Leer archivo como bytes → decodificar con OpenCV
    
    Args:
        path (str): Ruta completa de la imagen
        flag (int): cv2.IMREAD_COLOR (RGB) o cv2.IMREAD_GRAYSCALE
        
    Returns:
        np.ndarray: Imagen cargada o None si hay error
    """
    try:
        # Leer archivo como array de bytes
        stream = np.fromfile(path, dtype=np.uint8)
        # Decodificar bytes → imagen OpenCV
        img = cv2.imdecode(stream, flag)
        return img
    except Exception as e:
        print(f"❌ Error leyendo {path}: {e}")
        return None


def limpiar_directorio_resultados(path):
    """
    Elimina y recrea la estructura de carpetas para guardar resultados.
    
    Estructura creada:
    results/
    ├── TP/          # True Positives (tumores correctamente detectados)
    │   ├── Dice_00_10/  # Calidad baja (0-10% overlap)
    │   ├── Dice_10_20/
    │   └── ...
    ├── TN/          # True Negatives (sanos correctos)
    ├── FP/          # False Positives (falsas alarmas)
    └── FN/          # False Negatives (tumores perdidos)
    """
    # Si existe, borrar TODO (limpieza de ejecuciones previas)
    if os.path.exists(path):
        shutil.rmtree(path)
    
    # Crear categorías principales
    categorias = ["TP", "TN", "FP", "FN"]
    for cat in categorias:
        os.makedirs(os.path.join(path, cat), exist_ok=True)
    
    # Subcarpetas TP por calidad se crean dinámicamente durante inferencia
    print(f"[INFO] Directorio de resultados preparado: {path}")

def generar_plot_3d_paciente(pid, slices_data, save_path):
    """
    Genera una visualización 3D interactiva (Malla de Alambre/Contornos) de los bordes del tumor usando Plotly.
    
    Args:
        pid (str): ID del Paciente.
        slices_data (list): Lista de tuplas (slice_idx, mask_gt, mask_pred).
        save_path (str): Ruta donde guardar el archivo .html.
    """
    # Listas para coordenadas, insertaremos None para separar trazos (slices/contornos distintos)
    gt_x, gt_y, gt_z = [], [], []
    pred_x, pred_y, pred_z = [], [], []
    
    has_data = False

    # Procesar cada slice
    for slice_idx, m_gt, m_pred in slices_data:
        # --- Ground Truth ---
        contours_gt, _ = cv2.findContours(m_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_gt:
            has_data = True
            # Cerrar el contorno añadiendo el primer punto al final
            cnt = np.concatenate((cnt, [cnt[0]]), axis=0)
            
            for point in cnt:
                gt_x.append(point[0][0])
                gt_y.append(-point[0][1]) # Invertir Y
                gt_z.append(slice_idx)
            
            # Separador para que no una con el siguiente contorno
            gt_x.append(None)
            gt_y.append(None)
            gt_z.append(None)
                
        # --- Predicción ---
        contours_pred, _ = cv2.findContours(m_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_pred:
            has_data = True
            cnt = np.concatenate((cnt, [cnt[0]]), axis=0)
            
            for point in cnt:
                pred_x.append(point[0][0])
                pred_y.append(-point[0][1])
                pred_z.append(slice_idx)
                
            pred_x.append(None)
            pred_y.append(None)
            pred_z.append(None)
                
    if not has_data:
        return

    fig = go.Figure()

    # Trace GT (Verde)
    if gt_x:
        fig.add_trace(go.Scatter3d(
            x=gt_x, y=gt_y, z=gt_z,
            mode='lines',
            line=dict(color='green', width=4),
            name='Ground Truth',
            opacity=0.7
        ))
        
    # Trace Pred (Rojo)
    if pred_x:
        fig.add_trace(go.Scatter3d(
            x=pred_x, y=pred_y, z=pred_z,
            mode='lines',
            line=dict(color='red', width=4),
            name='Predicción AI',
            opacity=0.7
        ))
        
    fig.update_layout(
        title=f"Reconstrucción 3D Tumor (Contornos) - {pid}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y (Invertido)',
            zaxis_title='Slice (Z)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Guardar como HTML interactivo
    fig.write_html(save_path)

def preparar_dataset_pacientes_temp(origen, destino, n_pacientes):
    """
    Crea un dataset temporal con un subconjunto de pacientes.
    1. Borra 'destino' si existe.
    2. Selecciona 'n_pacientes' al azar de 'origen'.
    3. Copia todas las imágenes de esos pacientes a 'destino' (estructura plana).
    
    Returns:
        list[str]: Lista de IDs de pacientes seleccionados.
    """
    if os.path.exists(destino):
        shutil.rmtree(destino)
    os.makedirs(destino)
    
    # Listar carpetas de pacientes (cada carpeta es un paciente)
    # Estructura: kaggle_3m/TCGA_CS_4941_19960909/
    patient_folders = glob.glob(os.path.join(origen, "*"))
    patient_folders = [p for p in patient_folders if os.path.isdir(p)]
    
    if len(patient_folders) == 0:
        raise ValueError(f"No se encontraron carpetas de pacientes en {origen}")
        
    # Seleccionar N pacientes
    if n_pacientes == 'all' or int(n_pacientes) >= len(patient_folders):
        selected_folders = patient_folders
        print(f"   -> Usando TODOS los pacientes ({len(selected_folders)}).")
    else:
        selected_folders = random.sample(patient_folders, int(n_pacientes))
        print(f"   -> Seleccionados {len(selected_folders)} pacientes al azar.")
        
    selected_pids = []
    count_imgs = 0
    
    print("   -> Copiando archivos al dataset temporal...")
    for folder in tqdm(selected_folders, desc="Preparando Dataset Temp"):
        # Extraer ID paciente (basename de la carpeta)
        # Ojo: A veces el nombre de carpeta es TCGA_CS_4941_19960909, pero el ID base es TCGA_CS_4941.
        # Asumiremos la carpeta como unidad "paciente/estudio".
        folder_name = os.path.basename(folder)
        selected_pids.append(folder_name)
        
        # Buscar .tif dentro
        imgs = glob.glob(os.path.join(folder, "*.tif"))
        for img_path in imgs:
            dst = os.path.join(destino, os.path.basename(img_path))
            shutil.copy2(img_path, dst)
            count_imgs += 1
            
    print(f"   -> Dataset temporal listo en: {destino}")
    print(f"   -> Total imágenes: {count_imgs}")
    return selected_pids

def log_experiment_to_md(params, metrics, timings, cv_full, feat_imps, patient_metrics=None, filename="experiment_history.md"):
    """Guarda los resultados del experimento en un archivo Markdown persistente."""
    path = os.path.join(PROJECT_ROOT, filename)
    mode = 'a' if os.path.exists(path) else 'w'
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(path, mode, encoding='utf-8') as f:
        if mode == 'w':
            f.write("# Historial de Experimentos - Tumor Tracer AI\n\n")
        
        f.write(f"## 🧪 Prueba: {timestamp}\n")
        f.write(f"### 1. Configuración del Experimento\n")
        f.write(f"- **Dataset:** {params['n_images']} imágenes (Train: {params['n_train']}, Test: {params['n_test']})\n")
        f.write(f"- **Random Forest:** `Estimators={params['rf_est']}`, `Depth={params['rf_depth']}`, `ClassWeight={params['rf_weight']}`\n")
        f.write(f"- **Tiempos:** Extrac={timings['extraction']:.1f}s | CV={timings['cv']:.1f}s | Train={timings['train']:.1f}s | Inf={timings['inference']:.1f}s | **Total={timings['total']:.1f}s**\n")
        
        f.write(f"\n### 2. Validación Cruzada (K={CV_FOLDS}) - Estabilidad\n")
        f.write(f"| Fold | F1-Score | Precision | Recall |\n")
        f.write(f"|------|----------|-----------|--------|\n")
        for i in range(CV_FOLDS):
            f.write(f"| {i+1} | {cv_full['test_f1'][i]:.4f} | {cv_full['test_precision'][i]:.4f} | {cv_full['test_recall'][i]:.4f} |\n")
        
        f.write(f"| **Promedio** | **{metrics['cv_f1_mean']:.4f}** ± {metrics['cv_f1_std']*2:.4f} | {metrics['cv_prec_mean']:.4f} | {metrics['cv_rec_mean']:.4f} |\n")
        
        f.write(f"\n### 3. Importancia de Características (Top Influencias)\n")
        f.write(f"| Ranking | Característica | Importancia | Descripción |\n")
        f.write(f"|:-------:|----------------|-------------|-------------|\n")
        
        # Diccionario de descripciones breves
        desc_map = {
            "Green_Excess": "Índice de 'Verdosidad' (G - (R+B)/2)",
            "Green_Texture": "Interacción Verde * Textura",
            "Spatial_Radial": "Distancia al centro del cerebro",
            "A": "Canal A (LAB) - Rojo/Verde",
            "B_lab": "Canal B (LAB) - Azul/Amarillo",
            "Texture_LocalStd": "Complejidad/Rugosidad local",
            "Symmetry": "Diferencia entre hemisferios"
        }
        
        for i, (name, imp) in enumerate(feat_imps):
            desc = desc_map.get(name, "-")
            bold = "**" if i < 3 else ""
            f.write(f"| {i+1} | {bold}{name}{bold} | {imp:.4f} | {desc} |\n")
            
        f.write(f"\n### 4. Resultados Finales (Test Set - {params['n_test']} imágenes)\n")
        f.write(f"#### 📊 Clasificación de Imágenes\n")
        f.write(f"- ✅ **TP (Detectados):** {metrics['TP']} imágenes - *El modelo encontró el tumor correctamente.*\n")
        f.write(f"- ✅ **TN (Sanos):** {metrics['TN']} imágenes - *El modelo confirmó que estaba sano.*\n")
        f.write(f"- ❌ **FP (Falsas Alarmas):** {metrics['FP']} imágenes - *El modelo vio tumor donde no había.*\n")
        f.write(f"- ❌ **FN (Perdidos):** {metrics['FN']} imágenes - *El modelo NO vio el tumor existente.*\n")
        
        f.write(f"\n#### 🎯 Precisión Quirúrgica (Píxel a Píxel)\n")
        f.write(f"- **Sensibilidad (Recall):** `{metrics['Recall']:.2%}`\n")
        f.write(f"  > De todo el tejido tumoral real, el modelo detectó este porcentaje.\n")
        f.write(f"- **Confianza (Precision):** `{metrics['Precision']:.2%}`\n")
        f.write(f"  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.\n")
        f.write(f"- **Calidad de Segmentación (Dice):** `{metrics['Dice']:.2%}`\n")
        f.write(f"- **Limpieza de Ruido:** Se eliminaron **{metrics['NoiseReduced']:,}** píxeles de falsas alarmas durante el post-proceso.\n")
        
        if patient_metrics:
            f.write(f"\n### 5. Resultados por Paciente\n")
            f.write(f"| Paciente | TP | FP | FN | TN | Dice |\n")
            f.write(f"|----------|----|----|----|----|------|\n")
            # Ordenar por Dice descendente
            sorted_pats = sorted(patient_metrics.items(), key=lambda x: x[1]['Dice'], reverse=True)
            for pid, m in sorted_pats:
                f.write(f"| {pid} | {m['TP']} | {m['FP']} | {m['FN']} | {m['TN']} | {m['Dice']:.2%} |\n")

        f.write("\n" + "="*60 + "\n\n")
    
    print(f"\n[HISTORIAL] Resultados detallados guardados en: {filename}")


# 🛠️ Algoritmos de Preprocesamiento
#
# Antes de extraer características, normalizamos las imágenes para reducir la variabilidad.
#
# ### 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
# Mejora el contraste local.
#
# ### 2. Reducción de Ruido (Median Blur)
# Elimina el ruido granulado preservando bordes.
#
# ### 3. Alineación Geométrica con PCA
# Rota el cerebro para que el eje mayor sea vertical.

def apply_clahe(img):
    """Mejora de contraste adaptativa (CLAHE) en canal L (LAB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

def apply_denoise(img):
    """Filtro de Mediana para eliminar ruido 'sal y pimienta'."""
    return cv2.medianBlur(img, 3)

def align_brain(img, mask=None):
    """
    Alineación geométrica basada en PCA.
    Rota el cerebro para que el eje mayor sea vertical.
    Corrige orientación invertida usando heurística de masa.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    puntos = np.column_stack(np.where(thresh > 0)) # (y, x)
    
    if len(puntos) == 0:
        return (img, mask) if mask is not None else img

    # PCA Compute
    mean, eigenvectors, _ = cv2.PCACompute2(puntos.astype(np.float32), mean=None)
    center_img = (img.shape[1] // 2, img.shape[0] // 2)
    center_brain = (mean[0, 1], mean[0, 0])
    
    # Ángulo y Rotación Base
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
    rotation_angle = np.degrees(angle)
    
    # Forzar verticalidad
    if abs(rotation_angle) < 45: 
        rotation_angle += 90
        
    M = cv2.getRotationMatrix2D(center_brain, rotation_angle, 1.0)
    # Ajuste de traslación para centrar
    M[0, 2] += center_img[0] - center_brain[0]
    M[1, 2] += center_img[1] - center_brain[1]
    
    # Verificación de Orientación (Arriba vs Abajo)
    h, w = img.shape[:2]
    img_temp = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    
    gray_aligned = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    _, thresh_a = cv2.threshold(gray_aligned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Heurística: Si la mitad inferior tiene mucha más masa, está invertido
    if np.sum(thresh_a[h//2:, :]) > np.sum(thresh_a[:h//2, :]) * 1.1:
        rotation_angle += 180
        M = cv2.getRotationMatrix2D(center_brain, rotation_angle, 1.0)
        M[0, 2] += center_img[0] - center_brain[0]
        M[1, 2] += center_img[1] - center_brain[1]

    # Transformación Final
    img_aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    if mask is not None:
        mask_aligned = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        return img_aligned, mask_aligned
        
    return img_aligned


# 🧹 Post-Procesamiento y Limpieza
#
# Una vez que el modelo genera una predicción, aplicamos filtros para refinar el resultado.
#
# ### 1. Enmascaramiento del ROI (Region of Interest)
# Descarta predicciones fuera del contorno principal del cerebro.
#
# ### 2. Filtro de Tamaño (Area Threshold)
# Elimina manchas menores a **50 píxeles**.
#
# ### 3. Morfología Matemática
# Opening y Closing para suavizar bordes.

def eliminar_cerebelo_y_ruido(img_input, pred_binaria):
    """
    Limpieza post-predicción (Morphology + Size Filter).
    Nota: Se eliminó el recorte fijo del 40%; el modelo ahora infiere la ubicación.
    """
    # 1. Máscara del cerebro (ROI)
    if len(img_input.shape) == 3:
        gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_input
        
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    brain_mask = np.zeros_like(pred_binaria)
    if len(contours) > 0:
        cv2.drawContours(brain_mask, [max(contours, key=cv2.contourArea)], -1, 1, -1)
        brain_mask = cv2.erode(brain_mask, np.ones((5,5), np.uint8), iterations=2)

    # 2. Aplicar ROI
    cleaned = cv2.bitwise_and(pred_binaria, pred_binaria, mask=brain_mask)

    # 3. Filtrar manchas pequeñas (<50 px)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8))
    output = np.zeros_like(cleaned)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 50:
            output[labels == i] = 1

    # 4. Suavizado Morfológico
    kernel = np.ones((3, 3), np.uint8)
    output = cv2.morphologyEx(output.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)
    return cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel, iterations=1)


# 🔬 Ingeniería de Características (Feature Engineering)
#
# Transformamos cada píxel en un vector de **21 dimensiones**.
# - Color: RGB, Green_Excess, HSV, LAB
# - Textura: Canny, Sobel, LocalStd
# - Interacción: Green_Texture
# - Espacial: Radial, X, Y
# - Simetría

# ==========================================
# 4. INGENIERÍA DE CARACTERÍSTICAS
# ==========================================
def get_symmetry_feature(img):
    """Mapa de diferencia absoluto entre hemisferios (asume alineación vertical)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.absdiff(gray, cv2.flip(gray, 1))

def extract_features(img):
    """
    Genera un vector de características para cada píxel.
    Features: RGB, HSV, LAB, Bordes, Textura Local, Espaciales, Simetría, Interacción Verde.
    """
    df = pd.DataFrame()
    h, w, _ = img.shape

    # --- Color ---
    df['R'] = img[:, :, 2].reshape(-1)
    df['G'] = img[:, :, 1].reshape(-1)
    df['B'] = img[:, :, 0].reshape(-1)
    
    # Feature crítica: Green Excess (G - avg(R,B))
    # Discrimina 'verde artefacto' de 'verde tejido'
    df['Green_Excess'] = df['G'].astype(np.float32) - (df['R'].astype(np.float32) + df['B'].astype(np.float32)) / 2.0

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    df['H'] = img_hsv[:, :, 0].reshape(-1)
    df['S'] = img_hsv[:, :, 1].reshape(-1)
    df['V'] = img_hsv[:, :, 2].reshape(-1)

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    df['L'] = img_lab[:, :, 0].reshape(-1)
    df['A'] = img_lab[:, :, 1].reshape(-1)
    df['B_lab'] = img_lab[:, :, 2].reshape(-1)

    # --- Textura ---
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    df['Canny'] = cv2.Canny(img_gray, 100, 200).reshape(-1)
    df['Gaussian'] = nd.gaussian_filter(img_gray, sigma=3).reshape(-1)
    
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    df['Sobel_Mag'] = np.sqrt(sobel_x**2 + sobel_y**2).reshape(-1)

    # Desviación Estándar Local (Proxy de Entropía/Complejidad)
    img_f = img_gray.astype(np.float32)
    mu = cv2.blur(img_f, (5, 5))
    mu2 = cv2.blur(img_f**2, (5, 5))
    sigma = np.sqrt(np.maximum(mu2 - mu**2, 0))
    df['Texture_LocalStd'] = sigma.reshape(-1)

    # --- Interacción ---
    # Green * Texture: Ayuda a diferenciar ruido verde (liso) de tumor verde (rugoso)
    df['Green_Texture'] = df['Green_Excess'] * df['Texture_LocalStd']

    # --- Espaciales ---
    y, x = np.mgrid[0:h, 0:w]
    df['Spatial_Y'] = (y / h).astype(np.float32).reshape(-1)
    df['Spatial_X'] = (x / w).astype(np.float32).reshape(-1)
    df['Spatial_Radial'] = np.sqrt((df['Spatial_Y'] - 0.5)**2 + (df['Spatial_X'] - 0.5)**2)

    # --- Simetría ---
    df['Symmetry'] = get_symmetry_feature(img).reshape(-1)

    return df, (h, w)


# 📏 Definición de Métricas
#
# Evaluamos el rendimiento píxel a píxel utilizando métricas estándar:
# - Sensibilidad (Recall)
# - Precisión (Precision)
# - Dice Score (F1-Score)

# ==========================================
# 5. METRICAS Y EVALUACIÓN
# ==========================================
def calcular_metricas(y_true, y_pred):
    """Calcula métricas a nivel de píxel."""
    true_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)
    
    tp = np.sum((true_flat == 1) & (pred_flat == 1))
    fp = np.sum((true_flat == 0) & (pred_flat == 1))
    fn = np.sum((true_flat == 1) & (pred_flat == 0))
    tn = np.sum((true_flat == 0) & (pred_flat == 0))
    
    total_pos = tp + fn
    total_det = tp + fp
    
    recall = tp / total_pos if total_pos > 0 else 0.0
    precision = tp / total_det if total_det > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Recall": recall, "Precision": precision, "IoU": iou, "Dice": dice
    }


# 🚀 6. Ejecución del Pipeline
if __name__ == "__main__":
    timings = {}

    print("\n=== TUMOR TRACER AI - MENU PRINCIPAL ===")
    print("1. Entrenar Modelo Nuevo (Dataset Plano)")
    print("2. Analizar Pacientes (Usar Modelo Pre-entrenado)")
    
    try:
        mode_input = input("   > Seleccione opción [1/2]: ").strip()
    except EOFError:
        mode_input = "1"

    # ==========================================
    # MODO 1: ENTRENAMIENTO
    # ==========================================
    if mode_input == "1":
        print("\n--- INICIANDO MODO ENTRENAMIENTO ---")
        t_start_train_mode = time.time()
        
        # 1. Carga de Datos (Dataset Plano)
        search_path = os.path.join(PROJECT_ROOT, "data", "dataset_plano")
        print(f"[1] Buscando imágenes en: {search_path}")
        
        files_found = glob.glob(os.path.join(search_path, '**', '*_mask.tif'), recursive=True)
        if not files_found:
            raise FileNotFoundError("No se encontraron imágenes en dataset_plano.")
            
        valid_pairs = []
        for mask_p in files_found:
            img_p = mask_p.replace('_mask.tif', '.tif')
            if os.path.exists(img_p):
                valid_pairs.append((img_p, mask_p))
                
        # Subsampling
        if len(valid_pairs) > NUM_IMAGES:
            random.seed(RANDOM_STATE)
            train_pairs = random.sample(valid_pairs, NUM_IMAGES)
        else:
            train_pairs = valid_pairs
            
        print(f"    -> Imágenes para entrenamiento: {len(train_pairs)}")
        
        # 2. Extracción de Features
        print(f"\n[2] Extrayendo Características y Entrenando...")
        t_start_extract = time.time()
        X_train_list, Y_train_list = [], []

        for img_p, mask_p in tqdm(train_pairs, desc="Procesando Train"):
            img = cv2_imread_unicode(img_p)
            mask = cv2_imread_unicode(mask_p, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None: continue

            img = apply_clahe(img)
            img = apply_denoise(img)
            img, mask = align_brain(img, mask)

            mask = (mask // 255).reshape(-1)
            features, _ = extract_features(img)
            features = features.astype(np.float32)

            idx_tumor = np.where(mask == 1)[0]
            idx_backg = np.where(mask == 0)[0]
            
            counts_t = len(idx_tumor)
            counts_b = len(idx_backg)
            
            if counts_t > 0:
                needed_b = min(counts_b, counts_t * SUBSAMPLE_RATIO)
                sample_indices = np.concatenate([idx_tumor, np.random.choice(idx_backg, needed_b, replace=False)])
            else:
                sample_indices = np.random.choice(idx_backg, min(counts_b, 2000), replace=False)

            X_train_list.append(features.iloc[sample_indices])
            Y_train_list.append(mask[sample_indices])

        X_train = pd.concat(X_train_list)
        Y_train = np.concatenate(Y_train_list)
        del X_train_list, Y_train_list
        gc.collect()
        
        timings['extraction'] = time.time() - t_start_extract
        print(f"    -> Dataset Construido: {len(X_train):,} píxeles.")

        # 3. Entrenar Modelo
        print(f"\n[3] Entrenando Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            class_weight=RF_CLASS_WEIGHT,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0
        )
        rf_model.fit(X_train, Y_train)
        
        # 4. Guardar Modelo
        print(f"\n[4] Guardando Modelo en '{MODEL_FILENAME}'...")
        joblib.dump(rf_model, os.path.join(PROJECT_ROOT, MODEL_FILENAME))
        print("    -> [OK] Modelo guardado correctamente.")
        print(f"    -> Tiempo Total Entrenamiento: {time.time() - t_start_train_mode:.1f}s")


    # ==========================================
    # MODO 2: INFERENCIA / ANALISIS
    # ==========================================
    elif mode_input == "2":
        model_path = os.path.join(PROJECT_ROOT, MODEL_FILENAME)
        if not os.path.exists(model_path):
            print(f"\n❌ ERROR: No se encontró el modelo '{MODEL_FILENAME}'.")
            print("   Por favor, ejecute la opción 1 primero para entrenar.")
            sys.exit(1)
            
        print("\n--- INICIANDO MODO ANÁLISIS DE PACIENTES ---")
        
        # Cargar Modelo
        print(f"[1] Cargando modelo desde '{MODEL_FILENAME}'...")
        rf_model = joblib.load(model_path)
        print("    -> [OK] Modelo cargado.")
        
        # Seleccionar Pacientes
        print("\n[2] Selección de Pacientes")
        try:
            n_pat = input("   > ¿Cuántos pacientes analizar? (N o 'all'): ").strip()
            if n_pat.lower() != 'all': _ = int(n_pat)
        except:
            n_pat = "5"
            
        orig_kaggle = os.path.join(PROJECT_ROOT, "data", "kaggle_3m")
        temp_kaggle = os.path.join(PROJECT_ROOT, "data", "dataset_temp_pacientes")
        
        # Preparar datos
        preparar_dataset_pacientes_temp(orig_kaggle, temp_kaggle, n_pat)
        
        # Buscar archivos
        search_files = glob.glob(os.path.join(temp_kaggle, '*.tif'))
        # Filtrar solo mascaras para iterar
        mask_files = [f for f in search_files if '_mask.tif' in f]
        test_pairs = []
        for m in mask_files:
            img = m.replace('_mask.tif', '.tif')
            test_pairs.append((img, m))
            
        print(f"    -> Analizando {len(test_pairs)} imágenes...")
        
        # Inferencia
        results_dir = os.path.join(PROJECT_ROOT, "results")
        limpiar_directorio_resultados(results_dir)
        
        patient_3d_data = defaultdict(list)
        patient_metrics = defaultdict(lambda: {"TP":0, "FP":0, "FN":0, "TN":0, "DiceAcc":0.0, "CountTP":0})
        
        print(f"\n[3] Ejecutando Inferencia...")
        for img_p, mask_p in tqdm(test_pairs, desc="Inferencia"):
            img_orig = cv2_imread_unicode(img_p)
            mask_orig = cv2_imread_unicode(mask_p, cv2.IMREAD_GRAYSCALE)
            if img_orig is None: continue
            
            fname = os.path.basename(img_p)
            parts = fname.split('_')
            pid = "_".join(parts[:3])
            try: slice_idx = int(parts[-1].split('.')[0])
            except: slice_idx = 0
            
            # Preproceso
            img = apply_clahe(img_orig)
            img = apply_denoise(img)
            img, mask = align_brain(img, mask_orig)
            mask_bin = (mask // 255).astype(np.uint8)

            # Prediccion
            feat_df, (h, w) = extract_features(img)
            pred_flat = rf_model.predict(feat_df)
            pred_map = pred_flat.reshape(h, w).astype(np.uint8)
            
            # Limpieza
            clean_map = eliminar_cerebelo_y_ruido(img, pred_map)
            
            # Metricas
            m = calcular_metricas(mask_bin, clean_map)
            
            # Acumular
            for k in ["TP", "FP", "FN", "TN"]:
                patient_metrics[pid][k] += m[k]
                
            if np.sum(mask_bin) > 0 or np.sum(clean_map) > 0:
                patient_3d_data[pid].append((slice_idx, mask_bin, clean_map))
                
            # Guardar visual (Solo ejemplos o todos?)
            # Guardaremos todo para mantener consistencia con anterior
            has_tumor = np.sum(mask_bin) > 0
            detected = np.sum(clean_map) > 0
            cat = "TP" if (has_tumor and detected) else "TN"
            if has_tumor and not detected: cat = "FN"
            if not has_tumor and detected: cat = "FP"
            
            save_subdir = os.path.join("ByPatient", pid, cat)
            os.makedirs(os.path.join(results_dir, save_subdir), exist_ok=True)
            
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axs[0].set_title("Input")
            axs[1].imshow(mask_bin, cmap='gray'); axs[1].set_title("GT")
            axs[2].imshow(clean_map, cmap='Reds'); axs[2].set_title("AI Pred")
            plt.savefig(os.path.join(results_dir, save_subdir, f"res_{fname}.png"))
            plt.close()

        # Generar 3D
        print("\n[4] Generando 3D Interactivo...")
        for pid, slices in tqdm(patient_3d_data.items(), desc="Rendering 3D"):
            slices.sort(key=lambda x: x[0])
            save_path = os.path.join(results_dir, "ByPatient", pid, "3d_reconstruction.html")
            generar_plot_3d_paciente(pid, slices, save_path)
            
        print("\n[INFO] Análisis completado. Resultados en 'results/'.")


# 🔮 Conclusiones y Líneas de Futuro
#
# ### 1. Gestión de la Incertidumbre
# Clasificar predicciones en Alta Sospecha, Indecisos, Alta Confianza Negativa.
#
# ### 2. Impacto
# Reducción de carga del 60% y optimización del tiempo del radiólogo.
