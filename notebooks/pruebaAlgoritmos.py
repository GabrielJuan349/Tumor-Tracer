
"""
Script de Detección de Tumores Cerebrales (Random Forest)
=========================================================
Este script implementa un pipeline completo de Machine Learning para segmentación de tumores.
Incluye:
1.  Preprocesamiento Avanzado (CLAHE, Denoise, Alineación PCA).
2.  Ingeniería de Características (Color, Textura, Espacial, Simetría, Interacción).
3.  Entrenamiento eficiente con Subsampling.
4.  Validación Cruzada y Evaluación detallada.

Autor: [Tu Nombre/Equipo]
Fecha: Diciembre 2025
"""

import os
import glob
import random
import time
import gc
import shutil
import traceback

import pandas as pd
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configurar backend no interactivo para estabilidad en Windows
matplotlib.use('Agg')


# ==========================================
# 1. CONFIGURACIÓN Y CONSTANTES
# ==========================================
RANDOM_STATE = 42
RF_ESTIMATORS = 60
RF_MAX_DEPTH = 20
RF_CLASS_WEIGHT = {0: 1, 1: 1.5} # Peso 1.5 a Tumor para priorizar sensibilidad sin disparar FPs
SUBSAMPLE_RATIO = 3  # Ratio 1 pixel tumor : 3 pixeles fondo
CV_FOLDS = 7

# ==========================================
# 2. FUNCIONES DE LECTURA E I/O
# ==========================================
def cv2_imread_unicode(path, flag=cv2.IMREAD_COLOR):
    """Lee imágenes soportando caracteres unicode en la ruta (Windows)."""
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, flag)
        return img
    except Exception as e:
        print(f"Error leyendo {path}: {e}")
        return None

def limpiar_directorio_resultados(path):
    """Elimina y recrea el directorio de resultados para una ejecución limpia."""
    if os.path.exists(path):
        shutil.rmtree(path)
    
    subdirs = ["TP", "TN", "FP", "FN"]
    for cat in subdirs:
        os.makedirs(os.path.join(path, cat), exist_ok=True)
    
    # Subcarpetas TP por calidad (Deciles) se crean dinámicamente luego
    print(f"Directorio de resultados preparado: {path}")

# ==========================================
# 3. FUNCIONES DE PREPROCESAMIENTO
# ==========================================
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

# ==========================================
# 6. PIPELINE PRINCIPAL (MAIN)
# ==========================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    
    print("\n=== INICIANDO PIPELINE DE DETECCIÓN DE TUMORES ===")
    
    # --- 1. Buscar Datos ---
    print("\n[1] Buscando Dataset...")
    search_paths = [
        os.path.join(PROJECT_ROOT, "data", "dataset_plano"),
        os.path.join(PROJECT_ROOT, "data", "kaggle_3m")
    ]
    
    files_found = []
    for p in search_paths:
        if os.path.exists(p):
            curr = glob.glob(os.path.join(p, '**', '*_mask.tif'), recursive=True)
            if curr:
                files_found = curr
                print(f"    -> Encontrado: {p} ({len(curr)} máscaras)")
                break
    
    if not files_found:
        print("[ERROR] No se encontraron datos. Revise las rutas.")
        exit()

    # Preparar pares validos
    valid_pairs = []
    for mask_p in files_found:
        img_p = mask_p.replace('_mask.tif', '.tif')
        if os.path.exists(img_p):
            valid_pairs.append((img_p, mask_p))

    # Selección Aleatoria
    sample_size = min(500, len(valid_pairs))
    random.seed(RANDOM_STATE)
    selected = random.sample(valid_pairs, sample_size)
    print(f"    -> Seleccionadas {len(selected)} imágenes para el proceso.")

    # Split Train/Test
    train_pairs, test_pairs = train_test_split(selected, test_size=0.2, random_state=RANDOM_STATE)
    print(f"    -> Train: {len(train_pairs)} | Test: {len(test_pairs)}")

    # --- 2. Extracción de Features (Entrenamiento) ---
    print(f"\n[2] Extracción de Características (Train)...")
    X_train_list, Y_train_list = [], []
    
    start_time = time.time()
    for img_p, mask_p in tqdm(train_pairs, desc="Procesando Train"):
        img = cv2_imread_unicode(img_p)
        mask = cv2_imread_unicode(mask_p, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: continue

        # Pipeline Preproceso
        img = apply_clahe(img)
        img = apply_denoise(img)
        img, mask = align_brain(img, mask)

        mask = (mask // 255).reshape(-1)
        features, _ = extract_features(img)
        features = features.astype(np.float32)

        # Subsampling estrategico
        idx_tumor = np.where(mask == 1)[0]
        idx_backg = np.where(mask == 0)[0]
        
        counts_t = len(idx_tumor)
        counts_b = len(idx_backg)
        
        sample_indices = []
        if counts_t > 0:
            # Tomar todo el tumor y 3x de fondo
            needed_b = min(counts_b, counts_t * SUBSAMPLE_RATIO)
            if needed_b > 0:
                sample_indices = np.concatenate([
                    idx_tumor, 
                    np.random.choice(idx_backg, needed_b, replace=False)
                ])
            else:
                sample_indices = idx_tumor
        else:
            # Imagen sana: tomar pequeña muestra representativa
            sample_indices = np.random.choice(idx_backg, min(counts_b, 2000), replace=False)

        X_train_list.append(features.iloc[sample_indices])
        Y_train_list.append(mask[sample_indices])

    print(f"    -> Tiempo Extracción: {time.time() - start_time:.1f}s")
    
    # Consolidar
    X_train = pd.concat(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    del X_train_list, Y_train_list
    gc.collect()
    
    print(f"    -> Dataset Final: {len(X_train):,} píxeles.")
    print(f"    -> Distribución: Tumor={np.sum(Y_train==1):,}, Fondo={np.sum(Y_train==0):,}")

    # --- 3. Validación Cruzada ---
    print(f"\n[3] Validación Cruzada (K-Fold={CV_FOLDS})...")
    # Muestra reducida para CV rapido
    cv_idx = np.random.choice(len(Y_train), min(100000, len(Y_train)), replace=False)
    
    rf_model = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight=RF_CLASS_WEIGHT,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # Usamos cross_validate para obtener múltiples métricas
    scoring_metrics = ['f1', 'precision', 'recall']
    scores = cross_validate(rf_model, X_train.iloc[cv_idx], Y_train[cv_idx], cv=kfold, scoring=scoring_metrics, n_jobs=1)
    
    print(f"    -> Resultados por Fold:")
    print(f"       {'Fold':<5} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print(f"       {'-'*35}")
    
    for i in range(CV_FOLDS):
        f1 = scores['test_f1'][i]
        prec = scores['test_precision'][i]
        rec = scores['test_recall'][i]
        print(f"       {i+1:<5d} {f1:<10.4f} {prec:<10.4f} {rec:<10.4f}")
        
    print(f"       {'-'*35}")
    print(f"    -> PROMEDIOS:")
    print(f"       F1-Score  : {scores['test_f1'].mean():.4f} (+/- {scores['test_f1'].std()*2:.4f})")
    print(f"       Precision : {scores['test_precision'].mean():.4f}")
    print(f"       Recall    : {scores['test_recall'].mean():.4f}")
    
    stability = scores['test_f1'].std() < 0.05
    print(f"    -> Estado: {'✅ ESTABLE' if stability else '⚠️ INESTABLE'}")

    # --- 4. Entrenamiento Final ---
    print(f"\n[4] Entrenando Modelo Final...")
    rf_model.fit(X_train, Y_train)
    
    # Importancias
    imps = rf_model.feature_importances_
    feat_names = X_train.columns
    sorted_idx = np.argsort(imps)[::-1]
    
    print("\n    -> IMPORTANCIA DE CARACTERÍSTICAS (Todas):")
    print(f"       {'Ranking':<8} {'Feature':<20} {'Importancia':<10}")
    print(f"       {'-'*40}")
    for i in range(len(feat_names)):
        idx = sorted_idx[i]
        print(f"       {i+1:<8d} {feat_names[idx]:<20s} : {imps[idx]:.4f}")

    # --- 5. Inferencia y Evaluación (Test) ---
    print(f"\n[5] Evaluando en Test Set ({len(test_pairs)} imágenes)...")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    limpiar_directorio_resultados(results_dir)
    
    global_metrics = {"TP":0, "FP":0, "FN":0, "TN":0}
    img_counts = {"TP":0, "TN":0, "FP":0, "FN":0}
    tp_qualities = [] # Dice scores

    metrics_raw = {"TP":0, "FP":0, "FN":0} # Antes de limpiar
    
    total_cleaned_pixels = 0
    
    for img_p, mask_p in tqdm(test_pairs, desc="Inferencia"):
        img_orig = cv2_imread_unicode(img_p)
        mask_orig = cv2_imread_unicode(mask_p, cv2.IMREAD_GRAYSCALE)
        fname = os.path.basename(img_p)
        
        if img_orig is None: continue

        # Preproceso Test
        img = apply_clahe(img_orig)
        img = apply_denoise(img)
        img, mask = align_brain(img, mask_orig)
        mask_bin = (mask // 255).astype(np.uint8)

        # Prediccion
        feat_df, (h, w) = extract_features(img)
        pred_flat = rf_model.predict(feat_df)
        pred_map = pred_flat.reshape(h, w).astype(np.uint8)
        
        # Guardar metricas RAW
        m_raw = calcular_metricas(mask_bin, pred_map)
        metrics_raw["FP"] += m_raw["FP"]

        # Limpieza
        clean_map = eliminar_cerebelo_y_ruido(img, pred_map)
        
        # Metricas FINALES
        m_final = calcular_metricas(mask_bin, clean_map)
        total_cleaned_pixels += (m_raw["FP"] - m_final["FP"])
        
        # Acumular globales
        for k in global_metrics: global_metrics[k] += m_final[k]
        
        # Clasificar Imagen
        has_tumor = np.sum(mask_bin) > 0
        detected = np.sum(clean_map) > 0
        cat = "TN"
        if has_tumor and detected: cat = "TP"
        elif has_tumor and not detected: cat = "FN"
        elif not has_tumor and detected: cat = "FP"
        
        img_counts[cat] += 1
        
        # Guardar (Solo TP o errores FP/FN interesante guardar)
        # Gestionar carpetas dinámicas para TP
        save_subdir = cat
        extra_txt = ""
        
        if cat == "TP":
            dice = m_final["Dice"]
            tp_qualities.append(dice)
            decile = min(int(dice * 10), 9) * 10
            save_subdir = os.path.join("TP", f"Dice_{decile:02d}_{decile+10:02d}")
            os.makedirs(os.path.join(results_dir, save_subdir), exist_ok=True)
            extra_txt = f"Dice: {dice:.2%}"
        elif cat == "FP":
            extra_txt = f"Ruido: {m_final['FP']} px"
            
        # Generar Plot (Solo guardar)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"[{cat}] {fname} | {extra_txt}")
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axs[0].set_title("Input (Aligned)")
        axs[1].imshow(mask_bin, cmap='gray'); axs[1].set_title("Ground Truth")
        axs[2].imshow(clean_map, cmap='Reds'); axs[2].set_title("Predicción AI")
        for ax in axs: ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, save_subdir, f"res_{fname}.png"))
        plt.close()

    # --- 6. Reporte Final ---
    print("\n" + "="*60)
    print("REPORTE FINAL DE EJECUCIÓN")
    print("="*60)
    
    # 1. Imagenes
    n_test = len(test_pairs)
    print("1. CLASIFICACIÓN DE IMÁGENES")
    print(f"   Total: {n_test}")
    print(f"   ✅ TP: {img_counts['TP']:3d} ({img_counts['TP']/n_test:6.2%}) - Detectados Correctamente")
    print(f"   ✅ TN: {img_counts['TN']:3d} ({img_counts['TN']/n_test:6.2%}) - Sanos Correctos")
    print(f"   ❌ FP: {img_counts['FP']:3d} ({img_counts['FP']/n_test:6.2%}) - Falsas Alarmas")
    print(f"   ❌ FN: {img_counts['FN']:3d} ({img_counts['FN']/n_test:6.2%}) - Tumores Perdidos")
    
    # 2. Calidad
    print("\n2. CALIDAD DE SEGMENTACIÓN (Casos TP)")
    if tp_qualities:
        avg_dice = np.mean(tp_qualities)
        print(f"   Dice Promedio: {avg_dice:.2%}")
        # Histograma simple
        hist, _ = np.histogram(tp_qualities, bins=[0, 0.5, 0.7, 0.9, 1.01])
        print(f"   - Excelentes (>90%): {hist[3]}")
        print(f"   - Buenos (70-90%):   {hist[2]}")
        print(f"   - Regulares (50-70%):{hist[1]}")
        print(f"   - Pobres (<50%):     {hist[0]}")
    else:
        print("   (No hubo casos TP)")

    # 3. Pixeles
    print("\n3. PRECISIÓN QUIRÚRGICA (Píxeles)")
    tot_p = global_metrics["TP"] + global_metrics["FN"]
    tot_det = global_metrics["TP"] + global_metrics["FP"]
    
    sens = global_metrics["TP"] / tot_p if tot_p > 0 else 0
    conf = global_metrics["TP"] / tot_det if tot_det > 0 else 0
    
    print(f"   Sensibilidad (Recall): {sens:6.2%} (Tumor real encontrado)")
    print(f"   Confianza (Precision): {conf:6.2%} (Píxeles rojos que son tumor)")
    print(f"   Ruido Eliminado:       {total_cleaned_pixels:,} píxeles")
    
    print("\n[FIN] Resultados guardados en 'results/'")
