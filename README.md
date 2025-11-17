# Tumor-Tracer
Un trazador de IA multi-etapa: Clasifica la imagen (SVM), Dibuja el contorno (U-Net) y Perfecciona el trazo (RL).

## 🎯 Descripción del Proyecto

**Tumor-Tracer** es un sistema asistente de segmentación de imágenes médicas diseñado para abordar un problema crítico en HealthTech: **acelerar el diagnóstico y la planificación del tratamiento mediante el análisis automatizado de imágenes de resonancia magnética (MRI) cerebrales**.

### Valor Empresarial

- **Reducción de Horas de Trabajo**: Disminuye significativamente el tiempo que radiólogos y cirujanos dedican al análisis manual de imágenes
- **Mediciones Objetivas**: Proporciona mediciones cuantitativas y objetivas del tamaño y localización de tumores cerebrales
- **Prototipo SaMD**: Desarrolla un prototipo de Software as a Medical Device (SaMD) para aplicaciones clínicas

### Dataset: LGG MRI Segmentation

Este proyecto utiliza el dataset **LGG MRI Segmentation**, ideal por las siguientes razones:

- **Tamaño Manejable**: 88MB con 2,150 archivos de imágenes
- **Ciclos de Entrenamiento Rápidos**: Permite iteraciones rápidas (horas, no días)
- **Tarea Clara**: Segmentación semántica de tumores cerebrales de bajo grado (Low-Grade Glioma)
- **Anotaciones Completas**: Incluye imágenes MRI y máscaras de segmentación (ground truth)

## 🏗️ Arquitectura Multi-Etapa

El proyecto está diseñado en **tres fases evolutivas**, cada una construyendo sobre la anterior, demostrando la progresión desde ML clásico hasta técnicas avanzadas de RL:

### 📊 Fase 1: Clasificador de Imagen Completa (ML Clásico)

**Objetivo de Negocio**: Crear un modelo baseline rápido para una primera criba: ¿contiene esta imagen MRI algún tumor o no?

**Tarea de ML**: Clasificación de Imágenes (Binaria)

**Metodología**:
- **Ingeniería de Características**:
  - Características de textura GLCM (Gray-Level Co-occurrence Matrix)
  - Estadísticas de histograma de intensidad
  - Descriptores de forma
- **Modelo**: Support Vector Machine (SVM)
- **Output**: Predicción binaria (0 = sin tumor, 1 = con tumor)

**Resultado**: Un modelo que puede marcar rápidamente imágenes para revisión, pero no puede indicar dónde está el tumor.

**Limitación**: No proporciona localización espacial del tumor.

---

### 🎨 Fase 2: Segmentador de Precisión (Deep Learning)

**Objetivo de Negocio**: Proporcionar una herramienta de diagnóstico precisa que delinee el contorno exacto del tumor para la planificación quirúrgica.

**Tarea de DL**: Segmentación Semántica

**Metodología**:
- **Arquitectura**: U-Net, el estándar de oro en segmentación de imágenes biomédicas
  - Arquitectura encoder-decoder con skip connections
  - Captura contexto global y detalles locales simultáneamente
- **Entrenamiento**: Dada una imagen MRI, el modelo genera la máscara de segmentación correspondiente
- **Métricas**: Dice Score, Intersection over Union (IoU)

**Resultado**: Un modelo de DL que produce mapas de segmentación precisos, delineando píxel por píxel el contorno del tumor.

**Avance**: Transición de clasificación binaria a localización espacial precisa.

---

### 🤖 Fase 3: Agente de Anotación Interactiva (Reinforcement Learning)

**Objetivo de Negocio**: Reducir drásticamente el tiempo de anotación humana creando un "copiloto" de IA que aprende a corregir errores con la mínima intervención.

**Tarea de RL**: Optimización de Políticas (Active Learning / Interactive Segmentation)

**Metodología (Simulación)**:

**Componentes del MDP (Markov Decision Process)**:

1. **Estado (State)**:
   - Imagen MRI original
   - Máscara de predicción (imperfecta) de la Fase 2
   - Historial de correcciones previas

2. **Acción (Action)**:
   - Acciones de edición: "expandir máscara en el píxel (x,y)"
   - Acciones de consulta: "preguntar al humano por la etiqueta en el píxel (x,y)"
   - Acciones de refinamiento: ajustes locales de la segmentación

3. **Recompensa (Reward)**:
   - Mejora en Dice Score o IoU después de cada acción
   - Penalización por consultas innecesarias al humano
   - Recompensa por convergencia rápida a máscara correcta

4. **Algoritmo**: Deep Q-Network (DQN)
   - Red neuronal que aprende la función Q(s,a)
   - Explora estrategias óptimas de refinamiento
   - Aprende cuándo pedir ayuda humana vs. corregir automáticamente

**Resultado**: Un prototipo de sistema de anotación asistida por IA que aprende activamente, demostrando cómo humanos y IA colaboran en tareas de alto riesgo.

**Innovación**: El agente aprende la política óptima de interacción humano-IA para maximizar calidad minimizando esfuerzo humano.

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.8+
- CUDA (opcional, para entrenamiento acelerado con GPU)

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/GabrielJuan349/Tumor-Tracer.git
cd Tumor-Tracer

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales

**Fase 1 (ML Clásico)**:
- scikit-learn
- scikit-image
- numpy
- pandas
- opencv-python

**Fase 2 (Deep Learning)**:
- tensorflow / pytorch
- keras / torch
- albumentations (data augmentation)
- segmentation-models-pytorch

**Fase 3 (Reinforcement Learning)**:
- gym / gymnasium
- stable-baselines3
- torch

**Utilidades**:
- matplotlib
- seaborn
- tqdm
- pillow

## 📁 Estructura del Proyecto

```
Tumor-Tracer/
├── data/
│   ├── raw/                 # Dataset LGG MRI original
│   ├── processed/           # Datos preprocesados
│   └── augmented/           # Datos aumentados
├── src/
│   ├── phase1_svm/
│   │   ├── feature_extraction.py
│   │   ├── train_svm.py
│   │   └── predict_svm.py
│   ├── phase2_unet/
│   │   ├── unet_model.py
│   │   ├── train_unet.py
│   │   └── predict_unet.py
│   ├── phase3_rl/
│   │   ├── environment.py
│   │   ├── dqn_agent.py
│   │   ├── train_dqn.py
│   │   └── interactive_refine.py
│   └── utils/
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── metrics.py
│       └── visualization.py
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_phase1_svm_experiments.ipynb
│   ├── 03_phase2_unet_training.ipynb
│   └── 04_phase3_rl_training.ipynb
├── models/
│   ├── svm_classifier.pkl
│   ├── unet_segmentation.pth
│   └── dqn_agent.pth
├── results/
│   ├── phase1_metrics/
│   ├── phase2_segmentations/
│   └── phase3_refinements/
├── tests/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   └── test_phase3.py
├── requirements.txt
├── README.md
└── LICENSE
```

## 💻 Uso

### Fase 1: Clasificación con SVM

```bash
# Entrenar clasificador SVM
python src/phase1_svm/train_svm.py --data data/processed --output models/

# Predicción
python src/phase1_svm/predict_svm.py --model models/svm_classifier.pkl --image path/to/mri.png
```

### Fase 2: Segmentación con U-Net

```bash
# Entrenar U-Net
python src/phase2_unet/train_unet.py --data data/processed --epochs 100 --batch-size 8

# Segmentación
python src/phase2_unet/predict_unet.py --model models/unet_segmentation.pth --image path/to/mri.png
```

### Fase 3: Refinamiento con RL

```bash
# Entrenar agente DQN
python src/phase3_rl/train_dqn.py --episodes 1000 --unet-model models/unet_segmentation.pth

# Refinamiento interactivo
python src/phase3_rl/interactive_refine.py --agent models/dqn_agent.pth --image path/to/mri.png
```

## 📊 Métricas de Evaluación

### Fase 1 (Clasificación)
- **Accuracy**: Precisión general del clasificador
- **Precision/Recall**: Para clase positiva (tumor presente)
- **F1-Score**: Balance entre precisión y recall
- **ROC-AUC**: Área bajo la curva ROC

### Fase 2 (Segmentación)
- **Dice Score**: Coeficiente de similitud (principal métrica)
- **IoU (Jaccard Index)**: Intersection over Union
- **Hausdorff Distance**: Distancia máxima entre contornos
- **Pixel Accuracy**: Precisión a nivel de píxel

### Fase 3 (RL)
- **Dice Improvement**: Mejora en Dice Score tras refinamiento
- **Human Queries**: Número de consultas al humano
- **Convergence Steps**: Pasos hasta segmentación óptima
- **Reward per Episode**: Recompensa acumulada

## 🔬 Resultados Esperados

### Fase 1
- **Baseline rápido**: Clasificación en < 100ms por imagen
- **Accuracy objetivo**: > 90% en detección de presencia de tumor

### Fase 2
- **Segmentación precisa**: Dice Score > 0.85
- **Tiempo de inferencia**: < 2 segundos por imagen
- **Calidad clínica**: Contornos utilizables para planificación quirúrgica

### Fase 3
- **Eficiencia de anotación**: Reducción del 70% en tiempo de anotación humana
- **Mejora de segmentación**: +5-10% en Dice Score sobre predicción inicial
- **Interacción inteligente**: < 10 clics humanos para corrección completa

## 🛠️ Desarrollo y Contribución

### Flujo de Trabajo de Desarrollo

1. **Fork** del repositorio
2. Crear una **rama de feature** (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** de cambios (`git commit -m 'Añade nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abrir un **Pull Request**

### Estándares de Código

- Seguir PEP 8 para código Python
- Documentar funciones con docstrings
- Incluir tests unitarios para nuevas funcionalidades
- Mantener cobertura de tests > 80%

## 📚 Referencias y Recursos

### Papers Fundamentales

**Fase 2 - U-Net**:
- Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"

**Fase 3 - RL para Segmentación**:
- Luo, X., et al. (2021). "Deep Reinforcement Learning for Interactive Medical Image Segmentation"

### Dataset
- **LGG MRI Segmentation Dataset**: Disponible en Kaggle
- Link: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

### Herramientas y Frameworks
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Autores y Reconocimientos

- **Equipo Tumor-Tracer**: Desarrollo del sistema multi-etapa
- **Kaggle Community**: Por el dataset LGG MRI Segmentation
- **Comunidad Open Source**: Por las herramientas y frameworks utilizados

## 📧 Contacto

Para preguntas, sugerencias o colaboraciones, por favor:
- Abrir un **Issue** en GitHub
- Contactar al equipo de desarrollo

---

**⚠️ Aviso Legal**: Este es un proyecto de investigación y educativo. No debe utilizarse para diagnóstico clínico real sin la debida validación, certificación médica y aprobación regulatoria.
