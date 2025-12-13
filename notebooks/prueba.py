import pandas as pd
import numpy as np
import cv2
import os
import glob
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import ndimage as nd
from matplotlib import pyplot as plt
from tqdm import tqdm

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
sample_size = min(100, len(valid_pairs))  # 100 o menos si no hay suficientes
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
print(f"\n--- Extrayendo características de {len(train_imgs)} imágenes de entrenamiento ---")

X_list = []
Y_list = []

for img_path, mask_path in tqdm(zip(train_imgs, train_masks), total=len(train_imgs), desc="Extrayendo features"):
    img = cv2_imread_unicode(img_path, cv2.IMREAD_COLOR)
    mask = cv2_imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None:
        continue
    
    # Normalizar máscara a 0 y 1
    mask = mask // 255 

    # Extraer características de todas las imágenes (con o sin tumor)
    features, _ = extract_features(img)
    X_list.append(features)
    Y_list.append(mask.reshape(-1))

if len(X_list) == 0:
    print("Error: No se pudieron cargar imágenes.")
    exit()

# Concatenar todos los datos
X_prev = pd.concat(X_list)
Y_prev = np.concatenate(Y_list)

print(f"Dataset de entrenamiento: {X_prev.shape[0]:,} píxeles, {X_prev.shape[1]} características")
print(f"Distribución de clases: No Tumor={np.sum(Y_prev==0):,}, Tumor={np.sum(Y_prev==1):,}")

# ==========================================
# 5. ENTRENAMIENTO DEL MODELO
# ==========================================
print("\n--- Entrenando Random Forest... ---")

# Configuración mejorada del modelo
# - n_estimators=60: Más árboles para manejar las nuevas características de color (RGB, HSV, LAB)
# - max_depth=20: Mayor profundidad para capturar relaciones complejas entre características
# - min_samples_split=10: Evita sobreajuste requiriendo más muestras para dividir nodos
# - min_samples_leaf=5: Cada hoja debe tener al menos 5 muestras
# - class_weight='balanced': Compensa desbalance entre píxeles de tumor vs no-tumor
# - n_jobs=-1: Usa todos los núcleos de CPU disponibles
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

print(f"Configuración del modelo:")
print(f"  - Estimadores: 60")
print(f"  - Características de entrada: {X_prev.shape[1]}")
print(f"  - Total de píxeles: {X_prev.shape[0]:,}")

model.fit(X_prev, Y_prev)
print("\n¡Modelo entrenado exitosamente!")

# ==========================================
# 6. PRUEBA CON VISUALIZACIÓN MEJORADA
# ==========================================
print(f"\n--- Probando predicción en {len(test_imgs)} imágenes de test ---")

# Calculamos cuántas vamos a mostrar (10 imágenes o las que haya si son menos)
num_to_show = min(10, len(test_imgs))

# Seleccionar índices aleatorios del conjunto de test
random.seed(42)  # Para reproducibilidad (puedes cambiar o quitar el seed)
random_indices = random.sample(range(len(test_imgs)), num_to_show)

print(f"Visualizando {num_to_show} resultados con comparación detallada...\n")

# for i in range(num_to_show):
for idx, i in enumerate(random_indices):
    test_path = test_imgs[i]
    test_mask_path = test_masks[i]

    print(f"[{i+1}/{num_to_show}] Analizando: {os.path.basename(test_path)}")

    img_test = cv2_imread_unicode(test_path, cv2.IMREAD_COLOR)
    mask_real = cv2_imread_unicode(test_mask_path, cv2.IMREAD_GRAYSCALE)

    if img_test is None:
        print(f"  ⚠️ Error: No se pudo leer la imagen")
        continue

    try:
        # ========================================
        # PASO 1: PREDICCIÓN
        # ========================================
        features_test, shape_original = extract_features(img_test)
        prediccion = model.predict(features_test)

        # ========================================
        # PASO 2: RECONSTRUCCIÓN
        # ========================================
        h, w = shape_original[0], shape_original[1]
        matriz_raw = prediccion.reshape(h, w).astype(np.uint8)

        # ========================================
        # PASO 3: LIMPIEZA DE BORDES (Skull Stripping + Eliminación de Cerebelo)
        # ========================================
        matriz_limpia = eliminar_cerebelo_y_ruido(img_test, matriz_raw)
        imagen_predicha = (matriz_limpia * 255).astype(np.uint8)

        # ========================================
        # PASO 4: VISUALIZACIÓN (4 SUBPLOTS)
        # ========================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Detección de Tumor Cerebral - Imagen {i+1}/{num_to_show}',
                     fontsize=16, fontweight='bold', y=0.98)

        # [1] Imagen Original (Color)
        axes[0, 0].imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('(1) Original MRI - Color', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        axes[0, 0].text(0.5, -0.05, 'Imagen a Color (RGB)',
                        ha='center', transform=axes[0, 0].transAxes, fontsize=9, style='italic')

        # [2] Máscara Real (Ground Truth)
        if mask_real is not None:
            axes[0, 1].imshow(mask_real, cmap='gray')
            axes[0, 1].set_title('(2) Ground Truth - Máscara Real', fontsize=12, fontweight='bold', color='green')
            tumor_pixels = np.sum(mask_real > 0)
            axes[0, 1].text(0.5, -0.05, f'Píxeles de Tumor: {tumor_pixels:,}',
                            ha='center', transform=axes[0, 1].transAxes, fontsize=9, style='italic')
        else:
            axes[0, 1].text(0.5, 0.5, 'Máscara no disponible', ha='center', va='center', fontsize=12)
        axes[0, 1].axis('off')

        # [3] Predicción Cruda (Sin Filtro)
        axes[1, 0].imshow(matriz_raw, cmap='Reds')
        axes[1, 0].set_title('(3) Predicción Cruda (Sin Filtro)', fontsize=12, fontweight='bold', color='orange')
        raw_pixels = np.sum(matriz_raw > 0)
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, -0.05, f'Píxeles Detectados: {raw_pixels:,} | Incluye Ruido',
                        ha='center', transform=axes[1, 0].transAxes, fontsize=9, style='italic')

        # [4] Predicción Final (Limpia)
        axes[1, 1].imshow(imagen_predicha, cmap='Reds')
        axes[1, 1].set_title('(4) Predicción Final (Post-Procesada)', fontsize=12, fontweight='bold', color='red')
        clean_pixels = np.sum(imagen_predicha > 0)
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, -0.05, f'Píxeles Finales: {clean_pixels:,} | Skull Stripping + Limpieza',
                        ha='center', transform=axes[1, 1].transAxes, fontsize=9, style='italic')

        plt.tight_layout()
        plt.show()

        print(f"  ✓ Procesamiento exitoso: {clean_pixels:,} píxeles de tumor detectados\n")

    except Exception as e:
        print(f"  ⚠️ Error al procesar: {str(e)}\n")

if num_to_show == 0:
    print("No hay imágenes de test disponibles.")