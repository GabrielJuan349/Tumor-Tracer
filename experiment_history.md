# Historial de Experimentos - Tumor Tracer AI

## 🧪 Prueba: 2025-12-14 15:47:48
### 1. Configuración del Experimento
- **Dataset:** 500 imágenes (Train: 400, Test: 100)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=19.3s | CV=42.5s | Train=121.0s | Inf=58.0s | **Total=241.8s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.8945 | 0.9113 | 0.8782 |
| 2 | 0.8997 | 0.9260 | 0.8748 |
| 3 | 0.8900 | 0.9025 | 0.8779 |
| 4 | 0.9027 | 0.9192 | 0.8867 |
| 5 | 0.8945 | 0.9073 | 0.8821 |
| 6 | 0.8947 | 0.9087 | 0.8812 |
| 7 | 0.8977 | 0.9108 | 0.8850 |
| **Promedio** | **0.8963** ± 0.0077 | 0.9123 | 0.8808 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **Green_Excess** | 0.1444 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 2 | **A** | 0.1416 | Canal A (LAB) - Rojo/Verde |
| 3 | **Spatial_Radial** | 0.1005 | Distancia al centro del cerebro |
| 4 | Green_Texture | 0.0974 | Interacción Verde * Textura |
| 5 | B_lab | 0.0656 | Canal B (LAB) - Azul/Amarillo |
| 6 | Gaussian | 0.0543 | - |
| 7 | G | 0.0524 | - |
| 8 | Spatial_X | 0.0464 | - |
| 9 | L | 0.0413 | - |
| 10 | Spatial_Y | 0.0361 | - |
| 11 | B | 0.0333 | - |
| 12 | H | 0.0322 | - |
| 13 | R | 0.0317 | - |
| 14 | S | 0.0306 | - |
| 15 | Texture_LocalStd | 0.0303 | Complejidad/Rugosidad local |
| 16 | V | 0.0236 | - |
| 17 | Symmetry | 0.0188 | Diferencia entre hemisferios |
| 18 | Sobel_Mag | 0.0183 | - |
| 19 | Canny | 0.0011 | - |

### 4. Resultados Finales (Test Set - 100 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 32 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 37 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 30 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 1 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `74.14%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `58.21%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `69.71%`
- **Limpieza de Ruido:** Se eliminaron **58,620** píxeles de falsas alarmas durante el post-proceso.

============================================================

