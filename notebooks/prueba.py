import pandas as pd
import numpy as np
import cv2
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from scipy import ndimage as nd
from matplotlib import pyplot as plt

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
# 3. ENTRENAMIENTO
# ==========================================
print("\n--- Paso 1: Procesando datos para entrenar ---")

mask_paths = files_found
img_paths = [f.replace('_mask.tif', '.tif') for f in mask_paths]

target_samples = 50 
X_list = []
Y_list = []
samples_collected = 0

for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
    if samples_collected >= target_samples: break
    if not os.path.exists(img_path): continue
    
    img = cv2_imread_unicode(img_path, cv2.IMREAD_COLOR)
    mask = cv2_imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None: continue
        
    mask = mask // 255
    if np.max(mask) == 1: 
        features, _ = extract_features(img)
        X_list.append(features)
        Y_list.append(mask.reshape(-1))
        samples_collected += 1
        if samples_collected % 10 == 0:
            print(f"   -> Procesadas {samples_collected} imágenes...")

X = pd.concat(X_list)
Y = np.concatenate(Y_list)

print("--- Paso 2: Entrenando Random Forest... ---")
model = RandomForestClassifier(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42)
model.fit(X, Y)
print("¡Modelo entrenado!")

# ==========================================
# 4. PRUEBA CON VISUALIZACIÓN MEJORADA
# ==========================================
print("\n--- Paso 3: Probando predicción vs Realidad ---")

idx_prueba = i + 1

if idx_prueba < len(img_paths):
    test_path = img_paths[idx_prueba]
    test_mask_path = mask_paths[idx_prueba] 
    
    print(f"Analizando: {os.path.basename(test_path)}")
    
    img_test = cv2_imread_unicode(test_path, cv2.IMREAD_COLOR)
    mask_real = cv2_imread_unicode(test_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img_test is not None:
        # 1. Predicción
        features_test, shape_original = extract_features(img_test)
        prediccion = model.predict(features_test)
        
        # 2. Reconstrucción Matriz Original
        h, w = shape_original[0], shape_original[1]
        matriz_raw = prediccion.reshape(h, w).astype(np.uint8)
        
        # 3. >>> LIMPIEZA DE BORDES (NUEVO PASO) <<<
        # Aquí eliminamos el cráneo y el ruido
        matriz_limpia = limpiar_bordes_cerebro(img_test, matriz_raw)
        
        imagen_predicha = (matriz_limpia * 255).astype(np.uint8)

        # 4. Visualización (4 Columnas)
        plt.figure(figsize=(16, 5))
        
        # Original
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
        plt.title("MRI Original")
        plt.axis('off')
        
        # Realidad
        plt.subplot(1, 4, 2)
        if mask_real is not None:
            plt.imshow(mask_real, cmap='gray')
            plt.title("Realidad (Mask)")
        plt.axis('off')

        # Predicción SUCIA (Con bordes de cráneo)
        plt.subplot(1, 4, 3)
        plt.imshow(matriz_raw, cmap='gray')
        plt.title("Predicción (Sin Filtro)")
        plt.axis('off')
        
        # Predicción LIMPIA (Final)
        plt.subplot(1, 4, 4)
        plt.imshow(imagen_predicha, cmap='gray')
        plt.title("Predicción Final\n(Limpia)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        print("Gráfica generada exitosamente.")
    else:
        print("No se pudo leer la imagen de prueba.")
else:
    print("No quedan imágenes para probar.")