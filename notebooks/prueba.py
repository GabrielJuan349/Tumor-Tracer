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

def limpiar_bordes_cerebro(img_rgb, prediccion_binaria):
    """
    1. Detecta el contorno del cerebro.
    2. Lo encoge (erosión) para quitar el cráneo.
    3. Filtra la predicción para borrar lo que esté fuera del cerebro.
    """
    # Convertir a gris y binarizar para encontrar el "blob" del cerebro
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Encontrar el contorno más grande (el cerebro)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return prediccion_binaria # Si falla, devolvemos original
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Crear una máscara vacía y dibujar el cerebro
    brain_mask = np.zeros_like(gray)
    cv2.drawContours(brain_mask, [largest_contour], -1, 255, -1)
    
    # EROSIÓN: "Comerse" los bordes de la máscara para eliminar el cráneo
    kernel = np.ones((5,5), np.uint8)
    brain_mask = cv2.erode(brain_mask, kernel, iterations=2)
    
    # Aplicar máscara a la predicción (AND lógico)
    # Lo que sea 1 en prediccion Y 1 en cerebro se queda. El resto se va.
    prediccion_limpia = cv2.bitwise_and(prediccion_binaria, prediccion_binaria, mask=brain_mask)
    
    # LIMPIEZA EXTRA: Morphological Opening para quitar puntitos sueltos (ruido)
    kernel_clean = np.ones((3,3), np.uint8)
    prediccion_limpia = cv2.morphologyEx(prediccion_limpia, cv2.MORPH_OPEN, kernel_clean)
    
    return prediccion_limpia

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
# 2. FEATURE EXTRACTION
# ==========================================
def extract_features(img):
    df = pd.DataFrame()
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        df['R'] = img[:, :, 2].reshape(-1)
        df['G'] = img[:, :, 1].reshape(-1)
        df['B'] = img[:, :, 0].reshape(-1)
    else:
        img_gray = img
        df['Original Pixel'] = img_gray.reshape(-1)
    
    edges = cv2.Canny(img_gray, 100, 200)
    df['Canny'] = edges.reshape(-1)
    
    gaussian = nd.gaussian_filter(img_gray, sigma=3)
    df['Gaussian'] = gaussian.reshape(-1)
    
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
model = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42, class_weight='balanced')
model.fit(X_prev, Y_prev)
print("¡Modelo entrenado!")

# ==========================================
# 6. PRUEBA CON VISUALIZACIÓN MEJORADA
# ==========================================
print(f"\n--- Probando predicción en {len(test_imgs)} imágenes de test ---")

# Calculamos cuántas vamos a mostrar (Mínimo 10 o las que haya si son menos)
num_to_show = min(10, len(test_imgs))

print(f"Visualizando los primeros {num_to_show} resultados...\n")

for i in range(num_to_show):
    test_path = test_imgs[i]
    test_mask_path = test_masks[i]
    
    print(f"[{i+1}/{num_to_show}] Analizando: {os.path.basename(test_path)}")
    
    img_test = cv2_imread_unicode(test_path, cv2.IMREAD_COLOR)
    mask_real = cv2_imread_unicode(test_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img_test is not None:
        # 1. Predicción
        features_test, shape_original = extract_features(img_test)
        prediccion = model.predict(features_test)
        
        # 2. Reconstrucción
        h, w = shape_original[0], shape_original[1]
        matriz_raw = prediccion.reshape(h, w).astype(np.uint8)
        
        # 3. Limpieza de bordes
        matriz_limpia = limpiar_bordes_cerebro(img_test, matriz_raw)
        imagen_predicha = (matriz_limpia * 255).astype(np.uint8)

        # 4. Visualización (2 Filas x 2 Columnas)
        plt.figure(figsize=(10, 8))
        
        # Fila 1, Columna 1: Original
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
        plt.title(f"Imagen {i+1}: MRI Original")
        plt.axis('off')
        
        # Fila 1, Columna 2: Realidad
        plt.subplot(2, 2, 2)
        if mask_real is not None:
            plt.imshow(mask_real, cmap='gray')
            plt.title("Realidad (Mask)")
        plt.axis('off')

        # Fila 2, Columna 1: Predicción SUCIA
        plt.subplot(2, 2, 3)
        plt.imshow(matriz_raw, cmap='gray')
        plt.title("Predicción (Sin Filtro)")
        plt.axis('off')
        
        # Fila 2, Columna 2: Predicción LIMPIA
        plt.subplot(2, 2, 4)
        plt.imshow(imagen_predicha, cmap='gray')
        plt.title("Predicción Final (Limpia)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Error al leer la imagen {i+1}")
else:
    print("No hay imágenes de test disponibles.")