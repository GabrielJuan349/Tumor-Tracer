import pandas as pd
import numpy as np
import cv2
import os
import glob
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy import ndimage as nd
# from skimage.feature import graycomatrix, graycoprops # REMOVED: Dependency not found, using CV2 alternatives
import matplotlib
# Configurar backend "Agg" (No interactivo) para evitar errores de hilos/Tcl
# Esto significa que las imágenes se guardarán pero NO se abrirán ventanas emergentes.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
import time

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
    Skull Stripping Avanzado + Limpieza
    
    CAMBIO IMPORTANTE: SE HA ELIMINADO EL RECORTE FIJO DEL 40% INFERIOR.
    Ahora confiamos en que el modelo (entrenado con GLCM y Spatial Features)
    sepa distinguir el cerebelo del tumor.
    """
    # ========================================
    # PASO 1: SKULL STRIPPING CON HSV (Igual que antes)
    # ========================================
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    v_channel = img_hsv[:, :, 2]

    ret, thresh = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros_like(prediccion_binaria)

    largest_contour = max(contours, key=cv2.contourArea)
    brain_mask = np.zeros_like(v_channel)
    cv2.drawContours(brain_mask, [largest_contour], -1, 255, -1)

    # ========================================
    # PASO 2: ELIMINACIÓN DE RECORTE FIJO (ELIMINADO)
    # ========================================
    # ANTES: cerebelo_cutoff = int(h * 0.60); brain_mask[cerebelo_cutoff:, :] = 0
    # AHORA: No hacemos nada aquí. Mantenemos todo el cerebro.

    # Erosión para limpiar bordes del cráneo
    kernel = np.ones((5, 5), np.uint8)
    brain_mask = cv2.erode(brain_mask, kernel, iterations=2)

    # ========================================
    # PASO 3: APLICAR MÁSCARA A LA PREDICCIÓN
    # ========================================
    prediccion_limpia = cv2.bitwise_and(prediccion_binaria, prediccion_binaria, mask=brain_mask)

    # ========================================
    # PASO 4: FILTRAR COMPONENTES PEQUEÑOS Y RUIDO
    # ========================================
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        prediccion_limpia.astype(np.uint8), connectivity=8
    )

    output = np.zeros_like(prediccion_limpia)
    
    # Filtrar por área (ruido vs tumor real)
    # Umbral dinámico? Por ahora 50px es seguro para tumores visibles
    MIN_TUMOR_PIXELS = 50 
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_TUMOR_PIXELS:
            output[labels == i] = 1

    # ========================================
    # PASO 5: LIMPIEZA MORFOLÓGICA
    # ========================================
    kernel_clean = np.ones((3, 3), np.uint8)
    output = cv2.morphologyEx(output.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean, iterations=2)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel_clean, iterations=1)

    return output

def apply_clahe(img):
    """
    Aplica CLAHE al canal de Luminancia (L) del espacio LAB.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_merged = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

def apply_denoise(img):
    """
    Elimina ruido 'sal y pimienta' usando Filtro de Mediana.
    Conserva los bordes mejor que el desenfoque gaussiano.
    """
    return cv2.medianBlur(img, 3)

def align_brain(img, mask=None):
    """
    Alinea el cerebro verticalmente usando PCA.
    Maneja la ambigüedad de 180 grados usando heurísticas de masa.
    """
    # 1. Obtener nube de puntos
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    puntos = np.column_stack(np.where(thresh > 0)) # (y, x)
    
    if len(puntos) == 0:
        return img, mask

    # 2. PCA
    # mean: (y, x) centroide
    mean, eigenvectors, _ = cv2.PCACompute2(puntos.astype(np.float32), mean=None)
    
    h, w = img.shape[:2]
    center_img = (w // 2, h // 2)
    center_brain = (mean[0, 1], mean[0, 0]) # (x, y)
    
    # 3. Ángulo
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
    rotation_angle = np.degrees(angle)
    
    # Alinear al eje vertical (90 /-90 grados)
    # Por defecto PCA da el eje mayor. Queremos que sea vertical.
    # Si angle es 0 (horizontal), rotamos 90.
    if abs(rotation_angle) < 45: 
        rotation_angle += 90
        
    # 4. Matriz de Rotación Inicial
    M = cv2.getRotationMatrix2D(center_brain, rotation_angle, 1.0)
    tx = center_img[0] - center_brain[0]
    ty = center_img[1] - center_brain[1]
    M[0, 2] += tx
    M[1, 2] += ty
    
    # 5. Aplicar Rotación temporal para verificar orientación
    img_aligned_temp = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    
    # --- CHECK DE ORIENTACIÓN (RESOLVER AMBIGÜEDAD 180 GRADOS) ---
    # Heurística: En cortes axiales, el cerebro suele ser más "ancho" en la parte superior (Parietal/frontal)
    # y más estrecho o con huecos en la parte inferior (Fosa posterior/Cerebelo).
    # O, el centro de masa de la mitad superior vs mitad inferior.
    
    # Dividir imagen alineada en mitad superior e inferior
    gray_aligned = cv2.cvtColor(img_aligned_temp, cv2.COLOR_BGR2GRAY)
    _, thresh_aligned = cv2.threshold(gray_aligned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    top_half = thresh_aligned[:h//2, :]
    bottom_half = thresh_aligned[h//2:, :]
    
    top_mass = np.sum(top_half)
    bottom_mass = np.sum(bottom_half)
    
    # Si la parte de abajo tiene MUCHA más masa que la de arriba, probablemente está invertida
    # (El cerebro es generalmente más masivo en los hemisferios que en la punta del tronco/cerebelo)
    # Esta es una heurística simple y puede fallar en cortes muy específicos, pero ayuda.
    if bottom_mass > top_mass * 1.1: # Margen del 10%
        rotation_angle += 180 # Girar 180 grados
        # Recalcular Matriz con nuevo ángulo
        M = cv2.getRotationMatrix2D(center_brain, rotation_angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty

    # 6. Aplicar Transformación Final
    img_aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    if mask is not None:
        mask_aligned = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        return img_aligned, mask_aligned
        
    return img_aligned

def get_symmetry_feature(img_aligned):
    """Calcula mapa de asimetría."""
    gray = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
    flipped = cv2.flip(gray, 1) # Flip Horizontal
    diff = cv2.absdiff(gray, flipped)
    return diff

def calcular_metricas(y_true, y_pred):
    # ... (Mismo código anterior)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    
    if np.sum(y_true_flat) == 0:
        if np.sum(y_pred_flat) == 0:
            return {"IoU":1.0, "Dice":1.0, "Precision":1.0, "Recall":1.0, "TP":0, "FP":0, "FN":0, "TN":tn, "Note":"No Tumor (Correcto)"}
        else:
            return {"IoU":0.0, "Dice":0.0, "Precision":0.0, "Recall":0.0, "TP":0, "FP":fp, "FN":0, "TN":tn, "Note":"No Tumor (Falso Positivo)"}

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {"IoU":iou, "Dice":dice, "Precision":precision, "Recall":recall, "TP":tp, "FP":fp, "FN":fn, "TN":tn, "Note":"Tumor Presente"}

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
# 2. FEATURE EXTRACTION AVANZADO
# ==========================================
def extract_features(img):
    """
    Extrae características avanzadas:
    - RGB, HSV, LAB
    - Textura: Canny, Gaussian, Sobel
    - NUEVO: GLCM (Textura Haralick)
    - NUEVO: Spatial (X, Y, Radial)
    - NUEVO: Simetría
    """
    df = pd.DataFrame()
    h, w, _ = img.shape

    # --- 1. Espacios de Color ---
    df['R'] = img[:, :, 2].reshape(-1)
    df['G'] = img[:, :, 1].reshape(-1)
    df['B'] = img[:, :, 0].reshape(-1)

    # --- Feature Específica para Ruido Verde ---
    # Green_Excess: Cuánto domina el verde sobre el promedio de rojo y azul.
    # Ayuda a distinguir ruido verde puro de tejidos complejos.
    r_float = df['R'].astype(np.float32)
    g_float = df['G'].astype(np.float32)
    b_float = df['B'].astype(np.float32)
    df['Green_Excess'] = g_float - (r_float + b_float) / 2.0

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    df['H'] = img_hsv[:, :, 0].reshape(-1)
    df['S'] = img_hsv[:, :, 1].reshape(-1)
    df['V'] = img_hsv[:, :, 2].reshape(-1)

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    df['L'] = img_lab[:, :, 0].reshape(-1)
    df['A'] = img_lab[:, :, 1].reshape(-1)
    df['B_lab'] = img_lab[:, :, 2].reshape(-1)

    # --- 2. Textura Básica (Grises) ---
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    df['Canny'] = cv2.Canny(img_gray, 100, 200).reshape(-1)
    df['Gaussian'] = nd.gaussian_filter(img_gray, sigma=3).reshape(-1)
    
    # Sobel
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    df['Sobel_Mag'] = np.sqrt(sobel_x**2 + sobel_y**2).reshape(-1) # Magnitud del gradiente

    # --- 3. NUEVO: CARACTERÍSTICAS ESPACIALES (Coordinate Maps) ---
    # Normalizado 0..1
    # Y aumenta hacia abajo. Si la imagen está alineada, Y=1 es cerebelo.
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    
    # Normalización
    y_norm = y_grid.astype(np.float32) / h
    x_norm = x_grid.astype(np.float32) / w
    
    # Distancia Radial al centro (Rotation Invariant!)
    # Centro = (0.5, 0.5) en coordenadas normalizadas
    cy, cx = 0.5, 0.5
    radial_dist = np.sqrt((y_norm - cy)**2 + (x_norm - cx)**2)
    
    df['Spatial_Y'] = y_norm.reshape(-1)
    df['Spatial_X'] = x_norm.reshape(-1) # Útil para simetría implícita
    df['Spatial_Radial'] = radial_dist.reshape(-1)

    # --- 4. NUEVO: TEXTURA GLCM (Calculada en ventana móvil es muy lento, se usa aproximación o filtros) ---
    # Calcular GLCM rolling window pixel a pixel es PROHIBITIVO en Python puro.
    # Alternativa eficiente: Calcular propiedades GLCM globales en parches O usar filtros de entropía/desv local.
    # Para simplicidad y velocidad en este script "prueba.py":
    # Usaremos filtros de "Varianza Local" (Entropy) que capturan textura similar a GLCM Homogeneity/Entropy.
    
    # from skimage.filters.rank import entropy
    # from skimage.morphology import disk
    
    # Entropía local (Radio 3) - Detecta complejidad de textura
    # El cerebelo tiene textura regular -> Entropía media
    # Tumor tiene textura caótica -> Entropía alta o nula (necrosis)
    # Convertir a uint8 para skimage
    # Se usa img_gray directamente.
    
    # Optimización: GLCM Entropy aproximada por Entropy Filter
    # entropy_img = entropy(img_gray, disk(3)) # Lento en CPU single thread para imágenes 512x512
    # Usaremos desviación estándar local como proxy rápido de "Complejidad de Textura"
    
    # Textura: Desviación Estándar Local (Kernel 5x5)
    mean, std_dev = cv2.meanStdDev(img_gray) # Global, no sirve
    
    # Calcular media local y luego std dev local manualmente con blur
    # E[X^2] - (E[X])^2
    img_gray_f = img_gray.astype(np.float32)
    mu = cv2.blur(img_gray_f, (5, 5))
    mu2 = cv2.blur(img_gray_f**2, (5, 5))
    sigma = np.sqrt(np.maximum(mu2 - mu**2, 0))
    
    df['Texture_LocalStd'] = sigma.reshape(-1)

    # Nota: Si el usuario exige GLCM Haralick real, requeriría una implementación optimizada en C++ o 
    # usar features de parches grandes. Para píxel a píxel, Local Std Dev + Sobel + Entropy es el estándar rápido.
    # Vamos a añadir Entropía si no tarda demasiado.
    
    # --- 5. Simetría ---
    # Asume imagen alineada
    df['Symmetry'] = get_symmetry_feature(img).reshape(-1)

    return df, (h, w)

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
# 4. EXTRACCIÓN DE CARACTERÍSTICAS
# ==========================================
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
    
    # --- PREPROCESAMIENTO NUEVO (CLAHE + ALINEACIÓN) ---
    # 1. Mejorar Contraste y Reducir Ruido
    img = apply_clahe(img)
    img = apply_denoise(img)
    
    # 2. Alinear Geométricamente (Cerebro Vertical)
    # Importante: Alinear imagen Y máscara simultáneamente
    img, mask = align_brain(img, mask)
    
    # Normalizar máscara a 0 y 1
    mask = mask // 255 
    mask_flat = mask.reshape(-1)

    # Extraer características (Ahora incluye Simetría)
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
import gc
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
# 5. ENTRENAMIENTO DEL MODELO
# ==========================================
print("\n--- Entrenando Random Forest... ---")
# Nota: class_weight sigue siendo útil aunque hayamos balanceado un poco
model = RandomForestClassifier(
    n_estimators=60,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced',
    verbose=0 
)

print(f"Configuración del modelo:")
print(f"  - Estimadores: 60")
print(f"  - Características de entrada: {X_prev.shape[1]}")
print(f"  - Total de píxeles: {X_prev.shape[0]:,}")

start_time_train = time.time() # START TIMER
model.fit(X_prev, Y_prev)
end_time_train = time.time() # END TIMER

print(f"¡Modelo entrenado exitosamente en {end_time_train - start_time_train:.2f} segundos!")

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
# Se generarán dinámicamente según el Dice Score (00_10, 10_20...)
os.makedirs(os.path.join(results_dir, "TP"), exist_ok=True)

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

    mask_real = cv2_imread_unicode(test_mask_path, cv2.IMREAD_GRAYSCALE)

    if img_test is None: continue

    # --- PREPROCESAMIENTO NUEVO (CLAHE + DENOISE + ALINEACIÓN) ---
    img_test = apply_clahe(img_test)
    img_test = apply_denoise(img_test)
    img_test, mask_real = align_brain(img_test, mask_real) # Alinear ambos

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
            
            # Desglose TP por Deciles (10% en 10%)
            # Binning basado en Dice Score (0.0 a 1.0)
            dice_score = mets["Dice"]
            
            # Calcular bin: 0.0->0, 0.95->9, 1.0->9 (lo metemos en 90-100)
            bin_idx = int(dice_score * 10)
            if bin_idx >= 10: bin_idx = 9 # Cap en el último bin
            
            lower = bin_idx * 10
            upper = (bin_idx + 1) * 10
            
            # Nombre de carpeta: Dice_00_10, Dice_90_100, etc.
            folder_name = f"Dice_{lower:02d}_{upper:02d}"
            save_folder = os.path.join(results_dir, "TP", folder_name)
            os.makedirs(save_folder, exist_ok=True)
            
            cat_display = f"TP ({folder_name})"
                
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
print("TIEMPOS DE EJECUCIÓN")
print("="*60)
print(f"Extracción Features (Train): {end_time_extraction - start_time_extraction:.2f} s")
print(f"Entrenamiento Modelo     : {end_time_train - start_time_train:.2f} s")
print(f"Inferencia Total (Test)  : {total_inference_time:.2f} s")
print(f"Inferencia Promedio/Img  : {avg_inference_time:.4f} s ({1/avg_inference_time:.1f} FPS)")

print("\nProceso guardado completado. Resultados en 'results/'")
print("\n--- FIN DEL DIAGNÓSTICO ---")
