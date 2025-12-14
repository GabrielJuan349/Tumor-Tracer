# Explicación del Código: `prueba.py`

Este documento detalla el funcionamiento del script `prueba.py`, diseñado para la detección de tumores cerebrales en imágenes MRI utilizando un enfoque de Machine Learning clásico (Random Forest) con ingeniería de características avanzada.

## 1. Configuración Inicial y Backend
El script comienza configurando `matplotlib` para usar el backend **"Agg"**.
- **¿Por qué?** Para evitar errores de concurrencia (`thread errors`) en Windows cuando se generan muchas gráficas mientras se usa procesamiento paralelo.
- **Efecto**: Las imágenes se guardan directamente en el disco duro y no se abren en ventanas emergentes, lo que hace el proceso más rápido y estable.

## 2. Ingeniería de Características (`extract_features`)
En lugar de pasar la imagen cruda al modelo, extraemos información rica de cada píxel utilizando múltiples espacios de color y texturas.

### Espacios de Color
1.  **RGB**: Información básica de color.
2.  **HSV**: Crítico para separar crominancia (matiz) del brillo (valor).
3.  **LAB**: Espacio de color perceptualmente uniforme. `L` es usado para CLAHE.

### Features Geométricas y Espaciales (NUEVO)
1.  **Simetría**: Mapa de diferencia entre el hemisferio izquierdo y derecho (tras alineación).
2.  **Spatial Y/X**: Coordenadas normalizadas. Fundamental para enseñar al modelo dónde está el cerebelo (abajo) sin recortar.
3.  **Distancia Radial**: Distancia al centro. Invariante a rotación.

### Texturas (Escala de Grises)
1.  **Canny/Sobel**: Bordes.
2.  **Local Std Dev**: Medida de complejidad de textura (Proxy rápido de GLCM Entropy). Distingue cerebelo (patrón regular) de tumor (caótico).

**Resultado**: Cada píxel se convierte en un vector enriquecido (aprox 17 dimensiones) que describe color, textura, posición y simetría.

## 3. Optimización de Memoria (Subsampling)
Entrenar con 400 imágenes completas (256x256 cada una) generaría ~26 millones de puntos de datos, saturando la memoria RAM.
- **Estrategia**: Muestreo 1:3.
    - Se guardan **TODOS** los píxeles identificados como **Tumor**.
    - Se guarda solo una **muestra aleatoria** de píxeles de **Fondo (No Tumor)** (3 veces la cantidad de tumor).
- **Conversión `float32`**: Los datos se convierten a precisión simple (float32) para reducir el uso de RAM a la mitad.

## 4. Fase de Entrenamiento (`RandomForestClassifier`)
Se utiliza un clasificador Random Forest con los siguientes parámetros ajustados:
- `n_estimators=60`: Número de árboles de decisión (balance entre velocidad y precisión).
- `max_depth=20`: Profundidad máxima para capturar complejidad sin sobreajustar (overfitting).
- `class_weight='balanced'`: Da más importancia a la clase minoritaria (Tumor) para que el modelo no la ignore.
- `n_jobs=-1`: Usa todos los núcleos del procesador para entrenar más rápido.

## 5. Limpieza y Preprocesamiento (`eliminar_cerebelo_y_ruido` + `align_brain`)

### A. Preprocesamiento (Antes de extracción)
1.  **CLAHE**: Mejora de contraste adaptativa en canal L.
2.  **Filtro de Mediana**: (`cv2.medianBlur`, k=3). Elimina ruido "sal y pimienta" preservando bordes. Fundamental para evitar que micro-manchas brillantes se confundan con calcificaciones tumorales.
3.  **Alineación PCA**: Rotación automática para que el cerebro esté vertical.
    - Se usa una heurística de distribución de masa para corregir la inversión (upside-down), garantizando que las coordenadas espaciales sean consistentes.

### B. Post-Procesamiento (Limpieza de Predicciones)
La función de limpieza ha evolucionado para ser **Adaptativa**:
1.  **Skull Stripping (HSV/Otsu)**: Elimina cráneo y fondo.
2.  **Eliminación de Recorte Fijo**: SE ELIMINÓ el corte del 40% inferior. El modelo ahora decide qué es tumor y qué es cerebelo basándose en textura y posición.
3.  **Filtrado de Ruido y Morfología**: Se eliminan detecciones minúsculas (<50px) y se suavizan contornos.

## 6. Evaluación y Categorización Inteligente
El script organiza los resultados automáticamente en carpetas según la **calidad de la segmentación**, medida por el **Dice Score** (que penaliza falsos positivos):

- **TP (True Positive)**: Tumor detectado. Se subdivide en **Deciles** para análisis fino:
    - `Dice_90_100`: Segmentación casi perfecta.
    - `Dice_80_90`, `Dice_70_80`... : Calidad decreciente.
    - `Dice_00_10`: Detección pobre o extremadamente ruidosa.
- **TN (True Negative)**: Sano, detectado como sano (Carpeta `TN`).
- **FP (False Positive)**: Sano, pero el modelo detectó tumor (Carpeta `FP`, "Falsa Alarma").
- **FN (False Negative)**: Tumor presente pero no detectado (Carpeta `FN`, "Tumor Perdido").

Además, calcula métricas avanzadas como **IoU (Intersection over Union)** y **Dice Score**, que son mucho más fiables que el Accuracy simple para problemas médicos.

## 7. Reporte Final
Al finalizar, se imprime un reporte detallado en la terminal con:
- Tiempos de ejecución (Extracción, Entrenamiento, Inferencia).
- Conteos de clasificación de imágenes.
- Métricas globales píxel a píxel (cuántos píxeles exactos se fallaron o acertaron).
- Reducción de ruido conseguida por la fase de limpieza.
