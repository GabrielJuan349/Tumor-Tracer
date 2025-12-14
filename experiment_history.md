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

## 🧪 Prueba: 2025-12-14 16:00:46
### 1. Configuración del Experimento
- **Dataset:** 500 imágenes (Train: 400, Test: 100)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=14.3s | CV=22.6s | Train=111.9s | Inf=54.8s | **Total=204.7s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.8937 | 0.9087 | 0.8792 |
| 2 | 0.8967 | 0.9147 | 0.8794 |
| 3 | 0.8864 | 0.9099 | 0.8640 |
| 4 | 0.8908 | 0.9014 | 0.8805 |
| 5 | 0.8986 | 0.9048 | 0.8926 |
| 6 | 0.8948 | 0.9109 | 0.8792 |
| 7 | 0.8953 | 0.9091 | 0.8820 |
| **Promedio** | **0.8938** ± 0.0075 | 0.9085 | 0.8796 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **A** | 0.1431 | Canal A (LAB) - Rojo/Verde |
| 2 | **Green_Excess** | 0.1391 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 3 | **Spatial_Radial** | 0.1015 | Distancia al centro del cerebro |
| 4 | Green_Texture | 0.0981 | Interacción Verde * Textura |
| 5 | B_lab | 0.0655 | Canal B (LAB) - Azul/Amarillo |
| 6 | Gaussian | 0.0543 | - |
| 7 | G | 0.0528 | - |
| 8 | Spatial_X | 0.0457 | - |
| 9 | L | 0.0381 | - |
| 10 | Spatial_Y | 0.0364 | - |
| 11 | H | 0.0344 | - |
| 12 | R | 0.0341 | - |
| 13 | B | 0.0335 | - |
| 14 | S | 0.0333 | - |
| 15 | Texture_LocalStd | 0.0302 | Complejidad/Rugosidad local |
| 16 | V | 0.0240 | - |
| 17 | Symmetry | 0.0182 | Diferencia entre hemisferios |
| 18 | Sobel_Mag | 0.0165 | - |
| 19 | Canny | 0.0011 | - |

### 4. Resultados Finales (Test Set - 100 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 32 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 38 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 29 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 1 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `74.25%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `58.51%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `69.33%`
- **Limpieza de Ruido:** Se eliminaron **58,627** píxeles de falsas alarmas durante el post-proceso.

============================================================

## 🧪 Prueba: 2025-12-14 19:18:25
### 1. Configuración del Experimento
- **Dataset:** 300 imágenes (Train: 240, Test: 60)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=15.6s | CV=30.8s | Train=71.9s | Inf=110.1s | **Total=231.0s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.9125 | 0.9285 | 0.8970 |
| 2 | 0.9037 | 0.9221 | 0.8861 |
| 3 | 0.9104 | 0.9248 | 0.8964 |
| 4 | 0.9041 | 0.9213 | 0.8875 |
| 5 | 0.9108 | 0.9276 | 0.8945 |
| 6 | 0.9139 | 0.9303 | 0.8981 |
| 7 | 0.9163 | 0.9342 | 0.8992 |
| **Promedio** | **0.9102** ± 0.0088 | 0.9270 | 0.8941 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **A** | 0.1563 | Canal A (LAB) - Rojo/Verde |
| 2 | **Green_Excess** | 0.1503 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 3 | **Green_Texture** | 0.1114 | Interacción Verde * Textura |
| 4 | B_lab | 0.0803 | Canal B (LAB) - Azul/Amarillo |
| 5 | Spatial_Radial | 0.0732 | Distancia al centro del cerebro |
| 6 | Gaussian | 0.0520 | - |
| 7 | G | 0.0514 | - |
| 8 | L | 0.0445 | - |
| 9 | Spatial_X | 0.0416 | - |
| 10 | Spatial_Y | 0.0349 | - |
| 11 | B | 0.0330 | - |
| 12 | S | 0.0305 | - |
| 13 | R | 0.0295 | - |
| 14 | Texture_LocalStd | 0.0284 | Complejidad/Rugosidad local |
| 15 | H | 0.0275 | - |
| 16 | V | 0.0212 | - |
| 17 | Symmetry | 0.0175 | Diferencia entre hemisferios |
| 18 | Sobel_Mag | 0.0155 | - |
| 19 | Canny | 0.0010 | - |

### 4. Resultados Finales (Test Set - 60 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 20 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 20 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 17 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 3 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `60.30%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `45.28%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `65.24%`
- **Limpieza de Ruido:** Se eliminaron **34,751** píxeles de falsas alarmas durante el post-proceso.

============================================================

