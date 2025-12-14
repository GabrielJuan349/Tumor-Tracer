import pandas as pd
import numpy as np
import cv2
import os
import glob
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy import ndimage as nd
import matplotlib
# Configurar backend "Agg" (No interactivo) para evitar errores de hilos/Tcl
# Esto significa que las imágenes se guardarán pero NO se abrirán ventanas emergentes.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import gc

# ==========================================
# 0. FUNCIONES DE AYUDA (Lectura y Limpieza)
# ==========================================
def cv2_imread_unicode(path, flag=cv2.IMREAD_COLOR):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, flag)
        return img
    except Exception as e:
        print(f"Error leyendo {path}: {e}")
        return None

def eliminar_cerebelo_y_ruido(img_rgb, prediccion_binaria):
    """
    Skull Stripping Avanzado + Eliminación de Cerebelo

    PROBLEMA SOLUCIONADO: El modelo confunde el cerebelo (parte inferior del cerebro)
    con tumores, generando falsos positivos masivos.

    ESTRATEGIA:
    1. Detecta el contorno del cerebro usando HSV-V
    2. ELIMINA EL 40% INFERIOR de la imagen (donde está el cerebelo)
    3. Filtra componentes conectados pequeños (ruido)
    4. Aplica limpieza morfológica agresiva
    """
    h = prediccion_binaria.shape[0]

    # ========================================
    # PASO 1: SKULL STRIPPING CON HSV
    # ========================================
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    v_channel = img_hsv[:, :, 2]

    ret, thresh = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros_like(prediccion_binaria)  # Devolver imagen vacía si falla

    largest_contour = max(contours, key=cv2.contourArea)
    brain_mask = np.zeros_like(v_channel)
    cv2.drawContours(brain_mask, [largest_contour], -1, 255, -1)

    # ========================================
    # PASO 2: ELIMINAR CEREBELO (40% INFERIOR)
    # ========================================
    # El cerebelo está en la parte baja de las imágenes MRI axiales
    # Crear máscara que bloquee el 40% inferior de la imagen
    cerebelo_cutoff = int(h * 0.60)  # Mantener solo el 60% superior
    brain_mask[cerebelo_cutoff:, :] = 0  # Borrar todo lo que esté debajo

    # Erosión más agresiva para eliminar bordes del cráneo
    kernel = np.ones((5, 5), np.uint8)
    brain_mask = cv2.erode(brain_mask, kernel, iterations=2)

    # ========================================
    # PASO 3: APLICAR MÁSCARA A LA PREDICCIÓN
    # ========================================
    prediccion_limpia = cv2.bitwise_and(prediccion_binaria, prediccion_binaria, mask=brain_mask)

    # ========================================
    # PASO 4: FILTRAR COMPONENTES CONECTADOS PEQUEÑOS
    # ========================================
    # Eliminar regiones menores a 100 píxeles (probablemente ruido)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        prediccion_limpia.astype(np.uint8), connectivity=8
    )

    # Crear imagen de salida vacía
    output = np.zeros_like(prediccion_limpia)

    # Iterar sobre cada componente (ignorar el fondo, label=0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Solo mantener componentes con área >= 100 píxeles
        if area >= 100:
            output[labels == i] = 1

    # ========================================
    # PASO 5: LIMPIEZA MORFOLÓGICA AGRESIVA
    # ========================================
    kernel_clean = np.ones((3, 3), np.uint8)

    # Opening: Eliminar píxeles aislados
    output = cv2.morphologyEx(output.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean, iterations=2)

    # Closing: Cerrar huecos
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel_clean, iterations=1)

    return output

def calcular_metricas(y_true, y_pred):
    """
    Calcula métricas enfocadas en el TUMOR, ignorando el fondo masivo (Accuracy Global engañoso).
    
    Métricas implementadas:
    - IoU (Intersection over Union): Acierto real sobre la zona de interés.
    - Dice Score (F1): Similar a IoU pero da más peso a los aciertos.
    
    Lógica para CASOS SIN TUMOR (Mask Vacía):
    1. Si Mask está vacía y Predicción está vacía -> ÉXITO (IoU = 1.0)
    2. Si Mask está vacía y Predicción tiene algo -> FALLO (IoU = 0.0, Penalización Falsos Positivos)
    """
    # Aplanar arrays para comparar píxel a píxel
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    # Calcular componenetes de la matriz de confusión manualmente para control total
    # TP: Pixel es 1 en ambos
    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    
    # FP: Pixel es 0 en Realidad pero 1 en Predicción (Alucinación)
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    
    # FN: Pixel es 1 en Realidad pero 0 en Predicción (Tumor perdido)
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    
    # TN: Pixel es 0 en ambos (Fondo correctamente ignorado) - NO LO USAMOS para métricas tumorales
    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    
    # CASO ESPECIAL: IMAGEN SIN TUMOR (Ground Truth vacío)
    if np.sum(y_true_flat) == 0:
        if np.sum(y_pred_flat) == 0:
            return {
                "IoU": 1.0, "Dice": 1.0, "Precision": 1.0, "Recall": 1.0,
                "TP": 0, "FP": 0, "FN": 0, "TN": tn, "Note": "No Tumor (Correcto)"
            }
        else:
            return {
                "IoU": 0.0, "Dice": 0.0, "Precision": 0.0, "Recall": 0.0,
                "TP": 0, "FP": fp, "FN": 0, "TN": tn, "Note": "No Tumor (Falso Positivo!)"
            }

    # CÁLCULO DE MÉTRICAS (Cuando hay tumor)
    # IoU = TP / (TP + FP + FN)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # Dice (F1) = 2*TP / (2*TP + FP + FN)
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Note": "Tumor Presente"
    }

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) 

POSSIBLE_PATHS = [
    os.path.join(PROJECT_ROOT, "data", "dataset_plano"),
    os.path.join(PROJECT_ROOT, "data", "kaggle_3m"),
    "../data/dataset_plano/",
]

files_found = []
print("--- DIAGNÓSTICO DE RUTAS ---")
for path in POSSIBLE_PATHS:
    full_path = os.path.abspath(path)
    if os.path.exists(full_path):
        candidates = glob.glob(os.path.join(full_path, '**', '*_mask.tif'), recursive=True)
        if len(candidates) > 0:
            print(f" -> ¡ÉXITO! Encontradas {len(candidates)} máscaras en: {full_path}")
            files_found = candidates
            break

if not files_found:
    print("[ERROR] No se encontraron archivos.")
    exit()

# ==========================================
# 2. FEATURE EXTRACTION AVANZADO (MULTIESPACIO DE COLOR)
# ==========================================
def extract_features(img):
    """
    Extrae características avanzadas de múltiples espacios de color:
    - RGB: Para capturar información de color básica
    - HSV: Crítico para separar crominancia (matiz) del brillo (valor)
    - LAB: Para imitar la percepción visual humana
    - Texturas: Canny y Gaussian sobre escala de grises
    """
    df = pd.DataFrame()

    # Validar que la imagen sea a color
    if len(img.shape) != 3:
        raise ValueError("La imagen debe ser a color (3 canales)")

    # ========================================
    # 1. ESPACIO RGB (Color Básico)
    # ========================================
    df['R'] = img[:, :, 2].reshape(-1)  # OpenCV usa BGR, no RGB
    df['G'] = img[:, :, 1].reshape(-1)
    df['B'] = img[:, :, 0].reshape(-1)

    # ========================================
    # 2. ESPACIO HSV (Separación Crominancia/Luminancia)
    # ========================================
    # HSV es crítico para distinguir entre "brillo" y "color real"
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    df['H'] = img_hsv[:, :, 0].reshape(-1)  # Hue (Matiz/Tono)
    df['S'] = img_hsv[:, :, 1].reshape(-1)  # Saturation (Saturación)
    df['V'] = img_hsv[:, :, 2].reshape(-1)  # Value (Brillo)

    # ========================================
    # 3. ESPACIO LAB (Percepción Humana)
    # ========================================
    # LAB modela mejor cómo los humanos perciben el color
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    df['L'] = img_lab[:, :, 0].reshape(-1)  # Luminance (Luminosidad)
    df['A'] = img_lab[:, :, 1].reshape(-1)  # A: verde -> rojo
    df['B_lab'] = img_lab[:, :, 2].reshape(-1)  # B: azul -> amarillo

    # ========================================
    # 4. CARACTERÍSTICAS DE TEXTURA (Escala de Grises)
    # ========================================
    # Convertir a gris SOLO para análisis de textura
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detector de bordes Canny
    edges = cv2.Canny(img_gray, 100, 200)
    df['Canny'] = edges.reshape(-1)

    # Filtro Gaussiano (suavizado)
    gaussian = nd.gaussian_filter(img_gray, sigma=3)
    df['Gaussian'] = gaussian.reshape(-1)

    # Detector de bordes Sobel en X e Y
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    df['Sobel_X'] = np.abs(sobel_x).reshape(-1)
    df['Sobel_Y'] = np.abs(sobel_y).reshape(-1)

    return df, img_gray.shape

# ==========================================
# 3. BÚSQUEDA Y SELECCIÓN DE IMÁGENES
# ==========================================
print("\n--- Buscando imágenes ---")

# Obtener todas las máscaras encontradas
mask_paths = files_found
# Derivar las rutas de imágenes desde las máscaras
img_paths = [f.replace('_mask.tif', '.tif') for f in mask_paths]

# Crear pares imagen-máscara válidos
valid_pairs = []
for img_path, mask_path in zip(img_paths, mask_paths):
    if os.path.exists(img_path):
        valid_pairs.append((img_path, mask_path))

print(f"Encontrados {len(valid_pairs)} pares imagen-máscara válidos")

# ==========================================
# SELECCIÓN ALEATORIA DE 500 IMÁGENES
# ==========================================
random.seed(42)  # Para reproducibilidad
sample_size = min(500, len(valid_pairs))  # 100 o menos si no hay suficientes
# sample_size = min(3929, len(valid_pairs))  # 100 o menos si no hay suficientes
selected_pairs = random.sample(valid_pairs, sample_size)

print(f"Seleccionadas {sample_size} imágenes al azar")

# Separar en listas
selected_images = [p[0] for p in selected_pairs]
selected_masks = [p[1] for p in selected_pairs]

# ==========================================
# SPLIT TRAIN/TEST (80/20)
# ==========================================
train_imgs, test_imgs, train_masks, test_masks = train_test_split(
    selected_images, selected_masks, 
    test_size=0.2, 
    random_state=42
)

print(f"Train: {len(train_imgs)} imágenes | Test: {len(test_imgs)} imágenes")


# ==========================================
# 4. EXTRACCIÓN DE CARACTERÍSTICAS (OPTIMIZADO PARA MEMORIA)
# ==========================================
print(f"\n--- Extrayendo características de {len(train_imgs)} imágenes de entrenamiento ---")

X_list = []
Y_list = []

# Ratio de subsampling para No-Tumor.
# Guardamos todos los píxeles de tumor y 3 veces esa cantidad de fondo.
# Esto reduce masivamente el uso de RAM sin perder información crítica.
RATIO_NO_TUMOR = 3 

print(f"Estrategia de Subsampling: 1 Tumor : {RATIO_NO_TUMOR} No-Tumor")

start_time_extraction = time.time() # START TIMER

for img_path, mask_path in tqdm(zip(train_imgs, train_masks), total=len(train_imgs), desc="Extrayendo features"):
    img = cv2_imread_unicode(img_path, cv2.IMREAD_COLOR)
    mask = cv2_imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None:
        continue
    
    # Normalizar máscara a 0 y 1
    mask = mask // 255 
    mask_flat = mask.reshape(-1)

    # Extraer características
    features, _ = extract_features(img)
    
    # OPTIMIZACIÓN DE MEMORIA: Convertir a float32 inmediatamente
    features = features.astype(np.float32)
    
    # --- SUBSAMPLING INTELIGENTE ---
    # Identificar índices
    idx_tumor = np.where(mask_flat == 1)[0]
    idx_backg = np.where(mask_flat == 0)[0]
    
    # Si hay tumor, tomamos todos los píxeles de tumor
    # Y una muestra del fondo proporcional
    if len(idx_tumor) > 0:
        n_tumor = len(idx_tumor)
        n_backg = min(len(idx_backg), n_tumor * RATIO_NO_TUMOR)
        
        # Selección aleatoria del fondo
        if n_backg > 0:
            idx_backg_sample = np.random.choice(idx_backg, n_backg, replace=False)
            indices_finales = np.concatenate([idx_tumor, idx_backg_sample])
        else:
            indices_finales = idx_tumor # Caso raro: imagen es todo tumor
            
    else:
        # Si NO hay tumor, tomamos una muestra pequeña del fondo para que el modelo conozca tejidos sanos
        # (ej. 2000 píxeles por imagen sana ~ 3% de una imagen 256x256)
        n_sample = min(len(idx_backg), 2000)
        indices_finales = np.random.choice(idx_backg, n_sample, replace=False)

    # Filtrar DataFrame y Array de etiquetas
    X_subset = features.iloc[indices_finales]
    Y_subset = mask_flat[indices_finales]
    
    X_list.append(X_subset)
    Y_list.append(Y_subset)

end_time_extraction = time.time() # END TIMER
print(f"Tiempo de extracción (Train): {end_time_extraction - start_time_extraction:.2f} segundos")

# Liberar memoria explícitamente
del img, mask, features, mask_flat, idx_tumor, idx_backg
gc.collect()

if len(X_list) == 0:
    print("Error: No se pudieron cargar imágenes.")
    exit()

# Concatenar todos los datos
print("Concatenando datos en memoria...")
X_prev = pd.concat(X_list)
Y_prev = np.concatenate(Y_list)

print(f"Dataset de entrenamiento FINAL: {X_prev.shape[0]:,} píxeles")
print(f"Distribución: No Tumor={np.sum(Y_prev==0):,}, Tumor={np.sum(Y_prev==1):,}")
print(f"Uso de memoria estimado (X): {X_prev.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ==========================================
# 5. CROSS-VALIDATION CON K-FOLD (k=7)
# ==========================================

print("\n--- Realizando Cross-Validation con K-Fold (k=7) ---")


# Submuestreo para cross-validation (usar todos los píxeles es muy costoso)
# Tomamos una muestra estratificada para CV
sample_cv_size = min(100000, len(Y_prev))  # Máximo 100k píxeles para CV
indices = np.random.choice(len(Y_prev), sample_cv_size, replace=False)
X_cv = X_prev.iloc[indices]
Y_cv = Y_prev[indices]

print(f"Usando {sample_cv_size} píxeles para cross-validation")

# Definir K-Fold
start_time_cv = time.time()
kfold = KFold(n_splits=7, shuffle=True, random_state=42)

# Modelo para CV
cv_model = RandomForestClassifier(
    n_estimators=60,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced',
    verbose=1
)

# Ejecutar cross-validation
cv_scores = cross_val_score(cv_model, X_cv, Y_cv, cv=kfold, scoring='f1', n_jobs=-1)

end_time_cv = time.time()

print("\n" + "="*60)
print("RESULTADOS CROSS-VALIDATION (7-Fold)")
print("="*60)
print(f"Píxeles analizados: {sample_cv_size:,}")
print(f"Modelo: RandomForest (n_estimators=60, max_depth=20)")
print(f"Métrica: F1-Score (Tumor)")
print(f"Tiempo total: {end_time_cv - start_time_cv:.2f} segundos")

print("\n" + "-"*60)
print("F1-Scores por Fold:")
print("-"*60)
for i, score in enumerate(cv_scores, 1):
    bar_length = int(score * 50)  # Barra visual de 50 caracteres máximo
    bar = "█" * bar_length + "░" * (50 - bar_length)
    print(f"  Fold {i:2d}: {score:.4f}  {bar}")

print("\n" + "-"*60)
print("Estadísticas Globales:")
print("-"*60)
print(f"  F1-Score Promedio : {cv_scores.mean():.4f}")
print(f"  Desviación Estándar: {cv_scores.std():.4f}")
print(f"  Intervalo Confianza: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f} (95%)")
print(f"  F1-Score Máximo   : {cv_scores.max():.4f} (Fold {np.argmax(cv_scores)+1})")
print(f"  F1-Score Mínimo   : {cv_scores.min():.4f} (Fold {np.argmin(cv_scores)+1})")
print(f"  Mediana           : {np.median(cv_scores):.4f}")
print(f"  Rango             : {cv_scores.max() - cv_scores.min():.4f}")

print("\n" + "-"*60)
print("Interpretación:")
print("-"*60)
variance = cv_scores.std()
if variance < 0.05:
    stability = "✅ EXCELENTE - Modelo muy estable entre folds"
elif variance < 0.10:
    stability = "✓ BUENA - Varianza aceptable"
else:
    stability = "⚠ ALTA - Revisar distribución de datos o hiperparámetros"
print(f"  Estabilidad: {stability}")

avg_f1 = cv_scores.mean()
if avg_f1 > 0.80:
    performance = "✅ EXCELENTE - Detección de tumor muy precisa"
elif avg_f1 > 0.60:
    performance = "✓ BUENA - Detección aceptable"
else:
    performance = "⚠ MEJORABLE - Considerar más features o ajuste de parámetros"
print(f"  Rendimiento: {performance}")

print("="*60)

# ==========================================
# 6. ENTRENAMIENTO DEL MODELO FINAL
# ==========================================
print("\n--- Entrenando modelo final con todos los datos de train ---")

start_time_train = time.time()

model = RandomForestClassifier(
    n_estimators=60,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced',
    verbose=1  # Mostrar progreso del entrenamiento
)

X = X_prev
Y = Y_prev

print(f"Configuración del modelo:")
print(f"  - Estimadores: 60")
print(f"  - Max Depth: 20")
print(f"  - Características de entrada: {X.shape[1]}")
print(f"  - Total de píxeles: {X.shape[0]:,}")

model.fit(X, Y)

end_time_train = time.time()

print(f"\n--- Modelo Entrenado Exitosamente en {end_time_train - start_time_train:.2f} segundos ---")

# Guardar referencias para compatibilidad con celdas posteriores
subset_size = len(train_imgs)
image_paths = train_imgs + test_imgs
mask_paths = train_masks + test_masks

# ==========================================
# 6. EVALUACIÓN Y VISUALIZACIÓN
# ==========================================
print(f"\n--- Probando predicción en {len(test_imgs)} imágenes de test ---")

# Crear directorios para guardar resultados
results_dir = os.path.join(PROJECT_ROOT, "results")

# LIMPIEZA PREVIA: Borrar carpeta results antigua si existe
if os.path.exists(results_dir):
    import shutil
    shutil.rmtree(results_dir)
    print(f"Carpeta {results_dir} limpiada.")

# Estructura de carpetas: TP se divide en High y Low Accuracy
categories = ["TP", "TN", "FP", "FN"]
for cat in categories:
    os.makedirs(os.path.join(results_dir, cat), exist_ok=True)

# Subcarpetas para TP
tp_high_dir = os.path.join(results_dir, "TP", "High_Accuracy")
tp_low_dir = os.path.join(results_dir, "TP", "Low_Accuracy")
os.makedirs(tp_high_dir, exist_ok=True)
os.makedirs(tp_low_dir, exist_ok=True)

print(f"Los resultados se guardarán en: {results_dir}")

# Acumuladores de métricas globales (Píxel a Píxel)
all_true = []
all_pred_raw = []
all_pred_clean = []

# Listas para métricas PROMEDIO (Por Imagen)
tp_recalls = []   # Para guardar el % de acierto en casos TP
tp_miss_rates = [] # Para guardar el % de fallo en casos TP
fp_brain_ratios = [] # Para guardar el % de cerebro confundido en casos FP

# Contadores para reporte de clasificación
counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

# Límite de visualización en pantalla
DISPLAY_LIMIT = 10
# Lista para guardar las figuras y mostrarlas al final
figures_to_show = []

print(f"Procesando {len(test_imgs)} imágenes (Visualizando primeras {DISPLAY_LIMIT})...\n")

start_time_inference = time.time() # START TIMER

for i in tqdm(range(len(test_imgs)), desc="Procesando Test Set"):
    test_path = test_imgs[i]
    test_mask_path = test_masks[i]
    img_name = os.path.basename(test_path)
    
    img_test = cv2_imread_unicode(test_path, cv2.IMREAD_COLOR)
    mask_real = cv2_imread_unicode(test_mask_path, cv2.IMREAD_GRAYSCALE)

    if img_test is None: continue

    # Ground Truth a binario
    mask_real_bin = (mask_real // 255).astype(np.uint8)

    try:
        # Predicción
        features_test, shape_original = extract_features(img_test)
        prediccion = model.predict(features_test)
        
        h, w = shape_original[0], shape_original[1]
        matriz_raw = prediccion.reshape(h, w).astype(np.uint8)
        
        # ========================================
        # PASO 3: LIMPIEZA DE BORDES (Skull Stripping + Eliminación de Cerebelo)
        # ========================================
        # IMPORTANTE: Necesitamos la máscara del cerebro para calcular métricas FP vs Brain
        # Por eficiencia, la función eliminar_cerebelo_y_ruido ya calcula la máscara internamente,
        # pero aquí la recalculamos rápido para tener el Área del Cerebro.
        gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY) if len(img_test.shape)==3 else img_test
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        brain_area_pixels = 0
        if len(contours) > 0:
            brain_area_pixels = cv2.contourArea(max(contours, key=cv2.contourArea)) # Aprox del área cerebral

        matriz_limpia = eliminar_cerebelo_y_ruido(img_test, matriz_raw)
        
        # Acumular para métricas globales
        all_true.extend(mask_real_bin.flatten())
        all_pred_raw.extend(matriz_raw.flatten())
        all_pred_clean.extend(matriz_limpia.flatten())

        # Métricas de esta imagen (Píxel a Píxel)
        mets = calcular_metricas(mask_real_bin, matriz_limpia)

        # --- CLASIFICACIÓN DE LA IMAGEN (TP, TN, FP, FN) ---
        has_tumor = np.sum(mask_real_bin) > 0
        detected = np.sum(matriz_limpia) > 0
        
        category = ""
        save_folder = ""
        extra_info = "" # Texto extra para el título
        
        if has_tumor and detected:
            category = "TP"
            # Calcular Recall específico de esta imagen (Acierto)
            recall_img = mets["TP"] / (mets["TP"] + mets["FN"])
            tp_recalls.append(recall_img)
            tp_miss_rates.append(1 - recall_img)
            
            # Desglose TP
            if recall_img > 0.70:
                save_folder = tp_high_dir
                cat_display = "TP (High Acc)"
            else:
                save_folder = tp_low_dir
                cat_display = "TP (Low Acc)"
                
            # Info extendida TP: Muestra Acierto, Perdido y EXCESO (FP)
            extra_info = f"OK: {mets['TP']} | Miss: {mets['FN']} | Excess(FP): {mets['FP']}"
            
        elif not has_tumor and not detected:
            category = "TN"
            save_folder = os.path.join(results_dir, "TN")
            cat_display = "TN"
            
        elif has_tumor and not detected:
            category = "FN"
            save_folder = os.path.join(results_dir, "FN")
            cat_display = "FN"
            
        elif not has_tumor and detected:
            category = "FP"
            save_folder = os.path.join(results_dir, "FP")
            cat_display = "FP"
            
            # Calcular ratio de FP vs Cerebro
            if brain_area_pixels > 0:
                fp_ratio = mets["FP"] / brain_area_pixels
                fp_brain_ratios.append(fp_ratio)
                extra_info = f"FP: {mets['FP']} px ({fp_ratio:.2%} del cerebro)"
            else:
                extra_info = f"FP: {mets['FP']} px"
        
        counts[category] += 1

        # Generar Figura
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # Título enriquecido
        title_lines = [
            f'[{cat_display}] {img_name}',
            f'IoU: {mets["IoU"]:.2%} | Dice: {mets["Dice"]:.2%}',
            extra_info if extra_info else mets["Note"]
        ]
        fig.suptitle("\n".join(title_lines), fontsize=14, fontweight='bold', y=0.98)

        # [1] Original
        axes[0, 0].imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'(1) Original')
        axes[0, 0].axis('off')

        # [2] Mask Real
        axes[0, 1].imshow(mask_real, cmap='gray')
        axes[0, 1].set_title('(2) Mask Real')
        if has_tumor:
             axes[0, 1].text(0.5, -0.1, f'Tumor Real: {mets["TP"] + mets["FN"]:,} px', transform=axes[0,1].transAxes, ha='center')
        axes[0, 1].axis('off')

        # [3] Predicción Cruda
        axes[1, 0].imshow(matriz_raw, cmap='Reds')
        axes[1, 0].set_title('(3) Predicción Cruda')
        pixels_raw = np.sum(matriz_raw)
        axes[1, 0].text(0.5, -0.1, f'Raw Detect: {pixels_raw:,}', transform=axes[1,0].transAxes, ha='center')
        axes[1, 0].axis('off')

        # [4] Predicción Limpia
        axes[1, 1].imshow(matriz_limpia, cmap='Reds')
        axes[1, 1].set_title('(4) Predicción Limpia')
        
        # Texto detallado TP/FP en el plot
        res_text = f"Final: {mets['TP']+mets['FP']:,}"
        if category == "TP":
             # Aquí incluimos FP como parte del análisis de calidad (Exceso)
             res_text += f"\n[OK] (TP): {mets['TP']:,}"
             res_text += f"\n[MISS] (FN): {mets['FN']:,}"
             res_text += f"\n[?] Excess (FP): {mets['FP']:,}"
        elif category == "FP":
             res_text += f"\n[!] Falsa Alarma: {mets['FP']:,} px"
             
        axes[1, 1].text(0.5, -0.15, res_text, transform=axes[1,1].transAxes, ha='center', color='red')
        axes[1, 1].axis('off')

        plt.tight_layout()
        
        # GUARDAR IMAGEN
        save_path = os.path.join(save_folder, f"Result_{img_name}.png")
        plt.savefig(save_path)
        
        plt.close(fig) # Liberar memoria inmediatamente

    except Exception as e:
        print(f"Error procesando {test_path}: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# 7. REPORTE FINAL GLOBAL DETALLADO
# ==========================================
print("\n" + "="*60)
print("RESUMEN DE CLASIFICACIÓN (Por Imagen)")
print("="*60)
print(f"Total Imágenes Analizadas: {len(test_imgs)}")
print(f"✅ TN (Sano Correcto)   : {counts['TN']:3d}")
print(f"✅ TP (Tumor Detectado) : {counts['TP']:3d}")
print(f"❌ FP (Falsa Alarma)    : {counts['FP']:3d}")
print(f"❌ FN (Tumor Perdido)   : {counts['FN']:3d}")

# Reporte de Promedios
print("-" * 60)
print("ANÁLISIS DE CALIDAD DE DETECCIÓN (Promedios por Imagen)")
print("-" * 60)
if len(tp_recalls) > 0:
    avg_recall = np.mean(tp_recalls)
    avg_miss = np.mean(tp_miss_rates)
    print(f"PARA CASOS TP (Tumor Detectado):")
    print(f"  - Promedio de Tumor Detectado: {avg_recall:.2%} (Calidad del acierto)")
    print(f"  - Promedio de Tumor Perdido  : {avg_miss:.2%}")
else:
    print("No hubo casos TP para analizar promedios.")

if len(fp_brain_ratios) > 0:
    avg_fp_ratio = np.mean(fp_brain_ratios)
    print(f"\nPARA CASOS FP (Falsos Positivos):")
    print(f"  - Promedio de Cerebro confundido con Tumor: {avg_fp_ratio:.2%} del área cerebral")
else:
    print("\nNo hubo casos FP para analizar promedios.")
    
if len(all_true) > 0:
    print("\n" + "="*60)
    print("MÉTRICAS GLOBALES (A Nivel de Píxel)")
    print("="*60)
    
    y_true_all = np.array(all_true)
    y_raw_all = np.array(all_pred_raw)
    y_clean_all = np.array(all_pred_clean)
    
    def print_comprehensive_metrics(title, true, pred):
        m = calcular_metricas(true, pred)
        total_pixels = len(true)
        
        print(f"\n[{title}]")
        print("-" * 40)
        # Matriz de Confusión
        print(f"True Positives  (TP): {m['TP']:10,} ({m['TP']/total_pixels:.2%} del total)")
        print(f"True Negatives  (TN): {m['TN']:10,} ({m['TN']/total_pixels:.2%} del total)")
        print(f"False Positives (FP): {m['FP']:10,} ({m['FP']/total_pixels:.2%} del total) <--- RUIDO")
        print(f"False Negatives (FN): {m['FN']:10,} ({m['FN']/total_pixels:.2%} del total) <--- TUMOR PERDIDO")
        print("-" * 40)
        # Scores
        print(f"Accuracy : {m.get('Accuracy', (m['TP']+m['TN'])/total_pixels):.2%}")
        print(f"Recall   : {m['Recall']:.2%}")
        print(f"Precision: {m['Precision']:.2%}")
        print(f"F1-Score : {m.get('F1', m['Dice']):.2%}")
        print(f"IoU      : {m['IoU']:.2%}")

        return m

    m_raw = print_comprehensive_metrics("PREDICCIÓN CRUDA (Sin Filtros)", y_true_all, y_raw_all)
    m_clean = print_comprehensive_metrics("PREDICCIÓN FINAL (Skull Stripping)", y_true_all, y_clean_all)
    
    print("\n" + "="*60)
    print("MEJORA DE LIMPIEZA")
    print("="*60)
    fp_reduction = m_raw['FP'] - m_clean['FP']
    print(f"Falsos Positivos ELIMINADOS: {fp_reduction:,}")
    if m_raw['FP'] > 0:
        print(f"Reducción de Ruido: {(fp_reduction/m_raw['FP']):.2%}")

end_time_inference = time.time() # END TIMER
total_inference_time = end_time_inference - start_time_inference
avg_inference_time = total_inference_time / len(test_imgs) if len(test_imgs) > 0 else 0

print("\n" + "="*60)
print("VALIDACIÓN CRUZADA (K-FOLD) - RESUMEN")
print("="*60)
print(f"F1-Score Promedio (7-Fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Rango de Scores: [{cv_scores.min():.4f} - {cv_scores.max():.4f}]")
print(f"Mediana: {np.median(cv_scores):.4f}")
print(f"Tiempo Cross-Validation: {end_time_cv - start_time_cv:.2f} s")
if cv_scores.std() < 0.05:
    print("Estabilidad del Modelo: ✅ EXCELENTE (Baja varianza entre folds)")
elif cv_scores.std() < 0.10:
    print("Estabilidad del Modelo: ✓ BUENA (Varianza aceptable)")
else:
    print("Estabilidad del Modelo: ⚠ REVISAR (Alta varianza entre folds)")

print("\n" + "="*60)
print("TIEMPOS DE EJECUCIÓN")
print("="*60)
print(f"Extracción Features (Train): {end_time_extraction - start_time_extraction:.2f} s")
print(f"Cross-Validation (7-Fold) : {end_time_cv - start_time_cv:.2f} s")
print(f"Entrenamiento Modelo Final : {end_time_train - start_time_train:.2f} s")
print(f"Inferencia Total (Test)    : {total_inference_time:.2f} s")
print(f"Inferencia Promedio/Img    : {avg_inference_time:.4f} s ({1/avg_inference_time:.1f} FPS)")
print(f"TIEMPO TOTAL PIPELINE      : {end_time_extraction - start_time_extraction + end_time_cv - start_time_cv + end_time_train - start_time_train + total_inference_time:.2f} s")

print("\nProceso guardado completado. Resultados en 'results/'")
print("\n--- FIN DEL DIAGNÓSTICO ---")
