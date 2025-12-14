# 🧠 Tumor-Tracer AI

**Sistema de Segmentación de Tumores Cerebrales mediante Machine Learning Clásico (v2.0)**

Este proyecto implementa un pipeline completo de detección y segmentación de gliomas en imágenes de resonancia magnética (MRI) utilizando **Random Forest** y técnicas avanzadas de procesamiento de imágenes.

## 👥 Autores
- **Aimé Moral**
- **Gabriel Juan**

## 🎯 Objetivo del Proyecto
El objetivo principal es asistir en el diagnóstico médico mediante la segmentación automática de tumores cerebrales de bajo grado (LGG). El sistema busca maximizar la **Sensibilidad (Recall)** para asegurar que no se pierdan casos positivos, manteniendo un equilibrio con la precisión para evitar falsas alarmas.

## 📊 Dataset
Utilizamos el dataset **LGG MRI Segmentation** de Kaggle.
- **Contenido**: 3,929 imágenes MRI (FLAIR) y sus correspondientes máscaras de segmentación.
- **Resolución**: 256x256 píxeles.
- **Formato**: .tif

## 🧬 Metodología: Pipeline de 5 Etapas

El sistema no utiliza Deep Learning (redes neuronales profundas), sino un enfoque de **Machine Learning Clásico** altamente optimizado mediante ingeniería de características.

### 1. Preprocesamiento Avanzado
Antes de analizar las imágenes, normalizamos los datos para reducir la variabilidad:
- **CLAHE**: Mejora adaptativa del contraste para resaltar estructuras sutiles.
- **Denoise**: Filtro de mediana para eliminar ruido "sal y pimienta" preservando los bordes.
- **Alineación PCA**: Rotación automática del cerebro para alinear su eje mayor verticalmente, corrigiendo inclinaciones de la cabeza del paciente.

### 2. Ingeniería de Características (Feature Engineering)
Transformamos cada píxel en un vector de **21 dimensiones** que describe su contexto:
- **Color**: RGB, HSV, LAB y **Green_Excess** (índice para diferenciar tejido patológico).
- **Textura**: Detectores de bordes (Canny, Sobel) y desviación estándar local (rugosidad).
- **Espacial**: Coordenadas X, Y y distancia radial al centro.
- **Simetría**: Comparación con el hemisferio opuesto del cerebro (los tumores rompen la simetría).
- **Interacción**: Combinaciones sintéticas como `Green * Texture`.

### 3. Estrategia de Muestreo
Para manejar el desbalanceo extremo de clases (98% fondo vs 2% tumor):
- **Subsampling 1:3**: Por cada píxel de tumor, seleccionamos solo 3 píxeles de fondo para el entrenamiento.
- Esto permite entrenar con ~500k píxeles equilibrados en lugar de millones de píxeles vacíos.

### 4. Modelo: Random Forest
- **Algoritmo**: RandomForestClassifier de `scikit-learn`.
- **Configuración**: 100 árboles, profundidad máxima de 30.
- **Pesos de Clase**: Se penaliza más el error en la clase "Tumor" (1.5x) para priorizar la sensibilidad médica.

### 5. Post-Procesamiento
Limpieza de las predicciones crudas del modelo:
- **Morfología Matemática**: Operaciones de *Opening* y *Closing* para suavizar bordes y rellenar huecos.
- **Filtro de Área**: Eliminación de detecciones menores a 50 píxeles (ruido).
- **ROI Mask**: Restricción de la búsqueda al área del cerebro, ignorando el fondo negro.

## 📂 Estructura del Proyecto

```
Tumor-Tracer/
├── data/
│   ├── kaggle_3m/       # Dataset original (descargar aquí)
│   └── dataset_plano/   # Dataset procesado (generado automáticamente)
├── results/             # Resultados de la inferencia (imágenes TP, FP, FN, TN)
├── experiment_history.md # Log automático de métricas de cada ejecución
├── TumorDetectionPipeline.ipynb # Notebook principal con todo el código
├── README.md            # Documentación del proyecto
└── requirements.txt     # Dependencias
```

## 🚀 Ejecución

1.  **Instalar dependencias**:
    Asegúrate de tener instaladas las librerías necesarias (ver `requirements.txt` o instalar manualmente):
    ```bash
    pip install opencv-python pandas numpy scikit-learn matplotlib tqdm
    ```

2.  **Preparar Datos**:
    Descarga el dataset LGG MRI Segmentation y colócalo en `data/kaggle_3m/`.

3.  **Ejecutar Notebook**:
    Abre y ejecuta todas las celdas de `TumorDetectionPipeline.ipynb`.
    - El script migrará automáticamente los datos a una estructura plana en `data/dataset_plano/`.
    - Entrenará el modelo Random Forest.
    - Evaluará el conjunto de test.
    - Generará reportes visuales en la carpeta `results/`.

## 📈 Resultados y Métricas
El sistema evalúa su desempeño utilizando:
- **Dice Score**: Calidad de la segmentación (superposición).
- **Sensibilidad (Recall)**: Capacidad de detección de tumores.
- **Precisión**: Fiabilidad de las detecciones positivas.

Los resultados detallados de cada experimento se guardan automáticamente en `experiment_history.md`.

---
*Proyecto desarrollado como parte de la asignatura de Aprendizaje Computacional.*
