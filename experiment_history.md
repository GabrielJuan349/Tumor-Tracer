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

## 🧪 Prueba: 2025-12-14 20:27:16
### 1. Configuración del Experimento
- **Dataset:** 500 imágenes (Train: 400, Test: 100)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=15.8s | CV=30.5s | Train=117.3s | Inf=54.9s | **Total=219.2s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.8997 | 0.9235 | 0.8771 |
| 2 | 0.8846 | 0.9077 | 0.8626 |
| 3 | 0.8946 | 0.9077 | 0.8819 |
| 4 | 0.9002 | 0.9188 | 0.8823 |
| 5 | 0.8881 | 0.9094 | 0.8677 |
| 6 | 0.8949 | 0.9140 | 0.8767 |
| 7 | 0.8935 | 0.9064 | 0.8811 |
| **Promedio** | **0.8937** ± 0.0106 | 0.9125 | 0.8756 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **A** | 0.1432 | Canal A (LAB) - Rojo/Verde |
| 2 | **Green_Excess** | 0.1400 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 3 | **Green_Texture** | 0.0996 | Interacción Verde * Textura |
| 4 | Spatial_Radial | 0.0986 | Distancia al centro del cerebro |
| 5 | B_lab | 0.0626 | Canal B (LAB) - Azul/Amarillo |
| 6 | Gaussian | 0.0550 | - |
| 7 | G | 0.0540 | - |
| 8 | Spatial_X | 0.0461 | - |
| 9 | L | 0.0404 | - |
| 10 | Spatial_Y | 0.0357 | - |
| 11 | B | 0.0348 | - |
| 12 | S | 0.0331 | - |
| 13 | H | 0.0325 | - |
| 14 | R | 0.0318 | - |
| 15 | Texture_LocalStd | 0.0302 | Complejidad/Rugosidad local |
| 16 | V | 0.0241 | - |
| 17 | Symmetry | 0.0187 | Diferencia entre hemisferios |
| 18 | Sobel_Mag | 0.0184 | - |
| 19 | Canny | 0.0010 | - |

### 4. Resultados Finales (Test Set - 100 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 32 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 35 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 32 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 1 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `74.30%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `58.22%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `69.64%`
- **Limpieza de Ruido:** Se eliminaron **58,477** píxeles de falsas alarmas durante el post-proceso.

============================================================

## 🧪 Prueba: 2025-12-14 21:00:00
### 1. Configuración del Experimento
- **Dataset:** 3929 imágenes (Train: 3143, Test: 786)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=110.9s | CV=26.0s | Train=1269.0s | Inf=493.7s | **Total=1902.2s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.8576 | 0.8884 | 0.8288 |
| 2 | 0.8595 | 0.8929 | 0.8285 |
| 3 | 0.8442 | 0.8947 | 0.7991 |
| 4 | 0.8570 | 0.8886 | 0.8276 |
| 5 | 0.8588 | 0.8970 | 0.8239 |
| 6 | 0.8644 | 0.8878 | 0.8422 |
| 7 | 0.8586 | 0.8945 | 0.8254 |
| **Promedio** | **0.8572** ± 0.0115 | 0.8920 | 0.8251 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **Green_Excess** | 0.1472 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 2 | **A** | 0.1458 | Canal A (LAB) - Rojo/Verde |
| 3 | **Green_Texture** | 0.1060 | Interacción Verde * Textura |
| 4 | Spatial_Radial | 0.0891 | Distancia al centro del cerebro |
| 5 | B_lab | 0.0852 | Canal B (LAB) - Azul/Amarillo |
| 6 | Gaussian | 0.0528 | - |
| 7 | G | 0.0507 | - |
| 8 | Spatial_X | 0.0446 | - |
| 9 | L | 0.0382 | - |
| 10 | Spatial_Y | 0.0343 | - |
| 11 | B | 0.0340 | - |
| 12 | Texture_LocalStd | 0.0315 | Complejidad/Rugosidad local |
| 13 | H | 0.0280 | - |
| 14 | S | 0.0272 | - |
| 15 | R | 0.0267 | - |
| 16 | V | 0.0213 | - |
| 17 | Symmetry | 0.0197 | Diferencia entre hemisferios |
| 18 | Sobel_Mag | 0.0168 | - |
| 19 | Canny | 0.0010 | - |

### 4. Resultados Finales (Test Set - 786 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 265 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 271 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 239 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 11 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `82.01%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `59.04%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `70.11%`
- **Limpieza de Ruido:** Se eliminaron **493,565** píxeles de falsas alarmas durante el post-proceso.

============================================================

## 🧪 Prueba: 2025-12-25 16:28:42
### 1. Configuración del Experimento
- **Dataset:** 300 imágenes (Train: 240, Test: 60)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=8.7s | CV=20.1s | Train=55.0s | Inf=31.4s | **Total=115.7s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.9074 | 0.9297 | 0.8863 |
| 2 | 0.9148 | 0.9281 | 0.9018 |
| 3 | 0.9114 | 0.9268 | 0.8965 |
| 4 | 0.9063 | 0.9202 | 0.8928 |
| 5 | 0.9109 | 0.9296 | 0.8930 |
| 6 | 0.9117 | 0.9269 | 0.8971 |
| 7 | 0.9070 | 0.9246 | 0.8900 |
| **Promedio** | **0.9099** ± 0.0058 | 0.9266 | 0.8939 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **Green_Excess** | 0.1575 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 2 | **A** | 0.1539 | Canal A (LAB) - Rojo/Verde |
| 3 | **Green_Texture** | 0.1029 | Interacción Verde * Textura |
| 4 | B_lab | 0.0808 | Canal B (LAB) - Azul/Amarillo |
| 5 | Spatial_Radial | 0.0712 | Distancia al centro del cerebro |
| 6 | G | 0.0523 | - |
| 7 | Gaussian | 0.0518 | - |
| 8 | L | 0.0433 | - |
| 9 | Spatial_X | 0.0428 | - |
| 10 | Spatial_Y | 0.0349 | - |
| 11 | B | 0.0325 | - |
| 12 | H | 0.0314 | - |
| 13 | S | 0.0311 | - |
| 14 | R | 0.0291 | - |
| 15 | Texture_LocalStd | 0.0280 | Complejidad/Rugosidad local |
| 16 | V | 0.0231 | - |
| 17 | Symmetry | 0.0170 | Diferencia entre hemisferios |
| 18 | Sobel_Mag | 0.0154 | - |
| 19 | Canny | 0.0010 | - |

### 4. Resultados Finales (Test Set - 60 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 20 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 21 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 16 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 3 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `61.06%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `45.05%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `66.10%`
- **Limpieza de Ruido:** Se eliminaron **34,505** píxeles de falsas alarmas durante el post-proceso.

============================================================

## 🧪 Prueba: 2025-12-25 16:44:09
### 1. Configuración del Experimento
- **Dataset:** 300 imágenes (Train: 240, Test: 60)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=8.1s | CV=20.6s | Train=53.7s | Inf=30.0s | **Total=124.8s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.9173 | 0.9310 | 0.9039 |
| 2 | 0.9119 | 0.9351 | 0.8898 |
| 3 | 0.9143 | 0.9281 | 0.9009 |
| 4 | 0.9164 | 0.9359 | 0.8978 |
| 5 | 0.9125 | 0.9325 | 0.8934 |
| 6 | 0.9128 | 0.9295 | 0.8968 |
| 7 | 0.9113 | 0.9239 | 0.8991 |
| **Promedio** | **0.9138** ± 0.0042 | 0.9309 | 0.8974 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **A** | 0.1541 | Canal A (LAB) - Rojo/Verde |
| 2 | **Green_Excess** | 0.1525 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 3 | **Green_Texture** | 0.1084 | Interacción Verde * Textura |
| 4 | B_lab | 0.0788 | Canal B (LAB) - Azul/Amarillo |
| 5 | Spatial_Radial | 0.0724 | Distancia al centro del cerebro |
| 6 | G | 0.0554 | - |
| 7 | Gaussian | 0.0521 | - |
| 8 | L | 0.0472 | - |
| 9 | Spatial_X | 0.0425 | - |
| 10 | Spatial_Y | 0.0349 | - |
| 11 | B | 0.0328 | - |
| 12 | S | 0.0293 | - |
| 13 | R | 0.0291 | - |
| 14 | Texture_LocalStd | 0.0282 | Complejidad/Rugosidad local |
| 15 | H | 0.0273 | - |
| 16 | V | 0.0211 | - |
| 17 | Symmetry | 0.0175 | Diferencia entre hemisferios |
| 18 | Sobel_Mag | 0.0150 | - |
| 19 | Canny | 0.0010 | - |

### 4. Resultados Finales (Test Set - 60 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 21 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 21 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 16 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 2 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `60.87%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `44.79%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `63.10%`
- **Limpieza de Ruido:** Se eliminaron **35,267** píxeles de falsas alarmas durante el post-proceso.

============================================================

## 🧪 Prueba: 2025-12-25 16:46:55
### 1. Configuración del Experimento
- **Dataset:** 300 imágenes (Train: 224, Test: 76)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=8.7s | CV=21.8s | Train=45.1s | Inf=40.6s | **Total=117.8s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.9213 | 0.9332 | 0.9096 |
| 2 | 0.9132 | 0.9206 | 0.9060 |
| 3 | 0.9181 | 0.9301 | 0.9065 |
| 4 | 0.9102 | 0.9281 | 0.8929 |
| 5 | 0.9131 | 0.9217 | 0.9047 |
| 6 | 0.9157 | 0.9315 | 0.9005 |
| 7 | 0.9208 | 0.9258 | 0.9159 |
| **Promedio** | **0.9161** ± 0.0078 | 0.9273 | 0.9051 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **Green_Excess** | 0.1462 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 2 | **A** | 0.1450 | Canal A (LAB) - Rojo/Verde |
| 3 | **Green_Texture** | 0.1021 | Interacción Verde * Textura |
| 4 | B_lab | 0.0804 | Canal B (LAB) - Azul/Amarillo |
| 5 | Spatial_Radial | 0.0633 | Distancia al centro del cerebro |
| 6 | Gaussian | 0.0575 | - |
| 7 | G | 0.0570 | - |
| 8 | Spatial_X | 0.0471 | - |
| 9 | L | 0.0446 | - |
| 10 | Spatial_Y | 0.0362 | - |
| 11 | R | 0.0339 | - |
| 12 | S | 0.0329 | - |
| 13 | B | 0.0318 | - |
| 14 | V | 0.0299 | - |
| 15 | Texture_LocalStd | 0.0292 | Complejidad/Rugosidad local |
| 16 | H | 0.0282 | - |
| 17 | Symmetry | 0.0173 | Diferencia entre hemisferios |
| 18 | Sobel_Mag | 0.0164 | - |
| 19 | Canny | 0.0010 | - |

### 4. Resultados Finales (Test Set - 76 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 23 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 36 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 11 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 6 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `49.13%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `79.14%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `62.07%`
- **Limpieza de Ruido:** Se eliminaron **16,755** píxeles de falsas alarmas durante el post-proceso.

============================================================

## 🧪 Prueba: 2025-12-25 17:03:55
### 1. Configuración del Experimento
- **Dataset:** 82 imágenes (Train: 56, Test: 26)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=2.3s | CV=15.2s | Train=3.6s | Inf=13.5s | **Total=35.2s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.9772 | 0.9722 | 0.9822 |
| 2 | 0.9772 | 0.9798 | 0.9747 |
| 3 | 0.9817 | 0.9792 | 0.9841 |
| 4 | 0.9809 | 0.9806 | 0.9813 |
| 5 | 0.9780 | 0.9746 | 0.9815 |
| 6 | 0.9777 | 0.9766 | 0.9787 |
| 7 | 0.9780 | 0.9735 | 0.9826 |
| **Promedio** | **0.9787** ± 0.0034 | 0.9766 | 0.9808 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **A** | 0.2329 | Canal A (LAB) - Rojo/Verde |
| 2 | **Green_Excess** | 0.1447 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 3 | **B_lab** | 0.1344 | Canal B (LAB) - Azul/Amarillo |
| 4 | Spatial_Radial | 0.1125 | Distancia al centro del cerebro |
| 5 | Spatial_Y | 0.0557 | - |
| 6 | S | 0.0462 | - |
| 7 | Green_Texture | 0.0395 | Interacción Verde * Textura |
| 8 | L | 0.0371 | - |
| 9 | G | 0.0331 | - |
| 10 | H | 0.0255 | - |
| 11 | Gaussian | 0.0237 | - |
| 12 | Spatial_X | 0.0219 | - |
| 13 | R | 0.0216 | - |
| 14 | Texture_LocalStd | 0.0208 | Complejidad/Rugosidad local |
| 15 | B | 0.0189 | - |
| 16 | V | 0.0150 | - |
| 17 | Symmetry | 0.0092 | Diferencia entre hemisferios |
| 18 | Sobel_Mag | 0.0070 | - |
| 19 | Canny | 0.0002 | - |

### 4. Resultados Finales (Test Set - 26 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 0 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 17 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 3 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 6 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `0.00%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `0.00%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `0.00%`
- **Limpieza de Ruido:** Se eliminaron **1,148** píxeles de falsas alarmas durante el post-proceso.

### 5. Resultados por Paciente
| Paciente | TP | FP | FN | TN | Dice |
|----------|----|----|----|----|------|
| TCGA_FG_5964 | 0 | 178 | 5592 | 1698166 | 0.00% |

============================================================

## 🧪 Prueba: 2025-12-25 17:10:36
### 1. Configuración del Experimento
- **Dataset:** 340 imágenes (Train: 261, Test: 79)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=8.8s | CV=17.6s | Train=42.2s | Inf=37.7s | **Total=296.1s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.9425 | 0.9437 | 0.9414 |
| 2 | 0.9373 | 0.9505 | 0.9245 |
| 3 | 0.9376 | 0.9449 | 0.9305 |
| 4 | 0.9389 | 0.9391 | 0.9387 |
| 5 | 0.9438 | 0.9532 | 0.9345 |
| 6 | 0.9472 | 0.9522 | 0.9422 |
| 7 | 0.9302 | 0.9323 | 0.9282 |
| **Promedio** | **0.9397** ± 0.0102 | 0.9451 | 0.9343 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **A** | 0.1787 | Canal A (LAB) - Rojo/Verde |
| 2 | **Green_Excess** | 0.1478 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 3 | **B_lab** | 0.1243 | Canal B (LAB) - Azul/Amarillo |
| 4 | Green_Texture | 0.0832 | Interacción Verde * Textura |
| 5 | Spatial_Radial | 0.0782 | Distancia al centro del cerebro |
| 6 | Spatial_X | 0.0630 | - |
| 7 | H | 0.0434 | - |
| 8 | Spatial_Y | 0.0433 | - |
| 9 | R | 0.0359 | - |
| 10 | L | 0.0358 | - |
| 11 | B | 0.0294 | - |
| 12 | S | 0.0285 | - |
| 13 | G | 0.0245 | - |
| 14 | Gaussian | 0.0241 | - |
| 15 | V | 0.0209 | - |
| 16 | Texture_LocalStd | 0.0171 | Complejidad/Rugosidad local |
| 17 | Sobel_Mag | 0.0114 | - |
| 18 | Symmetry | 0.0095 | Diferencia entre hemisferios |
| 19 | Canny | 0.0010 | - |

### 4. Resultados Finales (Test Set - 79 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 1 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 56 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 2 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 20 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `0.00%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `0.00%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `0.00%`
- **Limpieza de Ruido:** Se eliminaron **6,577** píxeles de falsas alarmas durante el post-proceso.

### 5. Resultados por Paciente
| Paciente | TP | FP | FN | TN | Dice |
|----------|----|----|----|----|------|
| TCGA_DU_7013 | 0 | 30 | 12627 | 3198607 | 0.00% |
| TCGA_HT_7877 | 0 | 113 | 8176 | 1957791 | 0.00% |

============================================================

## 🧪 Prueba: 2025-12-25 17:16:58
### 1. Configuración del Experimento
- **Dataset:** 56 imágenes (Train: 20, Test: 36)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=1.0s | CV=20.3s | Train=5.3s | Inf=18.7s | **Total=45.9s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.9767 | 0.9727 | 0.9807 |
| 2 | 0.9791 | 0.9735 | 0.9848 |
| 3 | 0.9785 | 0.9726 | 0.9844 |
| 4 | 0.9816 | 0.9776 | 0.9856 |
| 5 | 0.9810 | 0.9776 | 0.9844 |
| 6 | 0.9815 | 0.9741 | 0.9891 |
| 7 | 0.9807 | 0.9757 | 0.9858 |
| **Promedio** | **0.9799** ± 0.0034 | 0.9748 | 0.9850 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **B_lab** | 0.2424 | Canal B (LAB) - Azul/Amarillo |
| 2 | **Green_Excess** | 0.1827 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 3 | **A** | 0.1675 | Canal A (LAB) - Rojo/Verde |
| 4 | Green_Texture | 0.1022 | Interacción Verde * Textura |
| 5 | V | 0.0591 | - |
| 6 | G | 0.0479 | - |
| 7 | S | 0.0412 | - |
| 8 | Gaussian | 0.0397 | - |
| 9 | L | 0.0296 | - |
| 10 | Spatial_Y | 0.0208 | - |
| 11 | B | 0.0135 | - |
| 12 | R | 0.0127 | - |
| 13 | Spatial_Radial | 0.0118 | Distancia al centro del cerebro |
| 14 | Spatial_X | 0.0107 | - |
| 15 | H | 0.0065 | - |
| 16 | Texture_LocalStd | 0.0045 | Complejidad/Rugosidad local |
| 17 | Sobel_Mag | 0.0038 | - |
| 18 | Symmetry | 0.0031 | Diferencia entre hemisferios |
| 19 | Canny | 0.0003 | - |

### 4. Resultados Finales (Test Set - 36 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 0 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 27 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 1 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 8 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `0.00%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `0.00%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `0.00%`
- **Limpieza de Ruido:** Se eliminaron **654** píxeles de falsas alarmas durante el post-proceso.

### 5. Resultados por Paciente
| Paciente | TP | FP | FN | TN | Dice |
|----------|----|----|----|----|------|
| TCGA_DU_7299 | 0 | 36 | 4934 | 2354326 | 0.00% |

============================================================

## 🧪 Prueba: 2025-12-25 17:22:16
### 1. Configuración del Experimento
- **Dataset:** 362 imágenes (Train: 285, Test: 77)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=9.4s | CV=19.0s | Train=47.4s | Inf=37.3s | **Total=127.8s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.9338 | 0.9470 | 0.9210 |
| 2 | 0.9273 | 0.9354 | 0.9194 |
| 3 | 0.9290 | 0.9460 | 0.9126 |
| 4 | 0.9262 | 0.9313 | 0.9212 |
| 5 | 0.9310 | 0.9359 | 0.9261 |
| 6 | 0.9333 | 0.9480 | 0.9190 |
| 7 | 0.9280 | 0.9365 | 0.9197 |
| **Promedio** | **0.9298** ± 0.0055 | 0.9400 | 0.9198 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **Green_Excess** | 0.1174 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 2 | **A** | 0.1173 | Canal A (LAB) - Rojo/Verde |
| 3 | **Spatial_Radial** | 0.1085 | Distancia al centro del cerebro |
| 4 | B_lab | 0.0923 | Canal B (LAB) - Azul/Amarillo |
| 5 | Green_Texture | 0.0836 | Interacción Verde * Textura |
| 6 | Spatial_Y | 0.0834 | - |
| 7 | Spatial_X | 0.0711 | - |
| 8 | Gaussian | 0.0563 | - |
| 9 | G | 0.0492 | - |
| 10 | L | 0.0379 | - |
| 11 | Texture_LocalStd | 0.0322 | Complejidad/Rugosidad local |
| 12 | R | 0.0271 | - |
| 13 | S | 0.0248 | - |
| 14 | B | 0.0242 | - |
| 15 | H | 0.0211 | - |
| 16 | V | 0.0198 | - |
| 17 | Sobel_Mag | 0.0187 | - |
| 18 | Symmetry | 0.0142 | Diferencia entre hemisferios |
| 19 | Canny | 0.0009 | - |

### 4. Resultados Finales (Test Set - 77 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 26 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 32 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 15 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 4 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `29.76%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `55.71%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `39.95%`
- **Limpieza de Ruido:** Se eliminaron **21,735** píxeles de falsas alarmas durante el post-proceso.

### 5. Resultados por Paciente
| Paciente | TP | FP | FN | TN | Dice |
|----------|----|----|----|----|------|
| TCGA_DU_6408 | 25151 | 19959 | 57319 | 3567587 | 39.43% |
| TCGA_HT_7680 | 0 | 35 | 2056 | 1374165 | 0.00% |

============================================================

## 🧪 Prueba: 2025-12-25 20:11:18
### 1. Configuración del Experimento
- **Dataset:** 775 imágenes (Train: 673, Test: 102)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=22.2s | CV=19.0s | Train=158.9s | Inf=52.6s | **Total=269.8s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.9173 | 0.9250 | 0.9098 |
| 2 | 0.9223 | 0.9357 | 0.9092 |
| 3 | 0.9212 | 0.9440 | 0.8995 |
| 4 | 0.9266 | 0.9385 | 0.9151 |
| 5 | 0.9226 | 0.9386 | 0.9071 |
| 6 | 0.9177 | 0.9401 | 0.8963 |
| 7 | 0.9191 | 0.9413 | 0.8979 |
| **Promedio** | **0.9210** ± 0.0061 | 0.9376 | 0.9050 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **Green_Excess** | 0.1464 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 2 | **A** | 0.1358 | Canal A (LAB) - Rojo/Verde |
| 3 | **B_lab** | 0.1248 | Canal B (LAB) - Azul/Amarillo |
| 4 | Spatial_Radial | 0.1022 | Distancia al centro del cerebro |
| 5 | Green_Texture | 0.0826 | Interacción Verde * Textura |
| 6 | Spatial_Y | 0.0465 | - |
| 7 | R | 0.0440 | - |
| 8 | L | 0.0397 | - |
| 9 | B | 0.0363 | - |
| 10 | Spatial_X | 0.0356 | - |
| 11 | G | 0.0336 | - |
| 12 | Texture_LocalStd | 0.0328 | Complejidad/Rugosidad local |
| 13 | Gaussian | 0.0298 | - |
| 14 | S | 0.0293 | - |
| 15 | H | 0.0290 | - |
| 16 | V | 0.0217 | - |
| 17 | Sobel_Mag | 0.0176 | - |
| 18 | Symmetry | 0.0115 | Diferencia entre hemisferios |
| 19 | Canny | 0.0007 | - |

### 4. Resultados Finales (Test Set - 102 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 34 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 42 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 25 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 1 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `66.46%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `80.50%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `74.14%`
- **Limpieza de Ruido:** Se eliminaron **33,303** píxeles de falsas alarmas durante el post-proceso.

### 5. Resultados por Paciente
| Paciente | TP | FP | FN | TN | Dice |
|----------|----|----|----|----|------|
| TCGA_HT_7693 | 14439 | 2188 | 1923 | 1357706 | 87.54% |
| TCGA_HT_7684 | 20100 | 1215 | 15510 | 1601575 | 70.62% |
| TCGA_CS_4942 | 6793 | 6207 | 320 | 1297400 | 67.55% |
| TCGA_DU_5854 | 3052 | 1140 | 4651 | 2350453 | 51.32% |

============================================================

## 🧪 Prueba: 2025-12-25 20:26:29
### 1. Configuración del Experimento
- **Dataset:** 761 imágenes (Train: 663, Test: 98)
- **Random Forest:** `Estimators=100`, `Depth=30`, `ClassWeight={0: 1, 1: 1.5}`
- **Tiempos:** Extrac=24.0s | CV=22.9s | Train=178.5s | Inf=62.2s | **Total=300.4s**

### 2. Validación Cruzada (K=7) - Estabilidad
| Fold | F1-Score | Precision | Recall |
|------|----------|-----------|--------|
| 1 | 0.9211 | 0.9381 | 0.9047 |
| 2 | 0.9141 | 0.9345 | 0.8945 |
| 3 | 0.9173 | 0.9289 | 0.9060 |
| 4 | 0.9211 | 0.9426 | 0.9006 |
| 5 | 0.9180 | 0.9360 | 0.9007 |
| 6 | 0.9114 | 0.9305 | 0.8930 |
| 7 | 0.9164 | 0.9351 | 0.8985 |
| **Promedio** | **0.9171** ± 0.0065 | 0.9351 | 0.8997 |

### 3. Importancia de Características (Top Influencias)
| Ranking | Característica | Importancia | Descripción |
|:-------:|----------------|-------------|-------------|
| 1 | **A** | 0.1598 | Canal A (LAB) - Rojo/Verde |
| 2 | **Green_Excess** | 0.1411 | Índice de 'Verdosidad' (G - (R+B)/2) |
| 3 | **Spatial_Radial** | 0.1180 | Distancia al centro del cerebro |
| 4 | Green_Texture | 0.0993 | Interacción Verde * Textura |
| 5 | B | 0.0595 | - |
| 6 | Gaussian | 0.0490 | - |
| 7 | Spatial_Y | 0.0461 | - |
| 8 | L | 0.0452 | - |
| 9 | Spatial_X | 0.0400 | - |
| 10 | H | 0.0400 | - |
| 11 | G | 0.0389 | - |
| 12 | B_lab | 0.0355 | Canal B (LAB) - Azul/Amarillo |
| 13 | Texture_LocalStd | 0.0327 | Complejidad/Rugosidad local |
| 14 | S | 0.0248 | - |
| 15 | R | 0.0243 | - |
| 16 | V | 0.0184 | - |
| 17 | Symmetry | 0.0135 | Diferencia entre hemisferios |
| 18 | Sobel_Mag | 0.0134 | - |
| 19 | Canny | 0.0006 | - |

### 4. Resultados Finales (Test Set - 98 imágenes)
#### 📊 Clasificación de Imágenes
- ✅ **TP (Detectados):** 25 imágenes - *El modelo encontró el tumor correctamente.*
- ✅ **TN (Sanos):** 28 imágenes - *El modelo confirmó que estaba sano.*
- ❌ **FP (Falsas Alarmas):** 36 imágenes - *El modelo vio tumor donde no había.*
- ❌ **FN (Perdidos):** 9 imágenes - *El modelo NO vio el tumor existente.*

#### 🎯 Precisión Quirúrgica (Píxel a Píxel)
- **Sensibilidad (Recall):** `65.33%`
  > De todo el tejido tumoral real, el modelo detectó este porcentaje.
- **Confianza (Precision):** `26.79%`
  > De todo lo que el modelo marcó en rojo, este porcentaje era realmente tumor.
- **Calidad de Segmentación (Dice):** `54.99%`
- **Limpieza de Ruido:** Se eliminaron **83,429** píxeles de falsas alarmas durante el post-proceso.

### 5. Resultados por Paciente
| Paciente | TP | FP | FN | TN | Dice |
|----------|----|----|----|----|------|
| TCGA_HT_7608 | 30694 | 30127 | 2870 | 1771317 | 65.04% |
| TCGA_CS_5393 | 15522 | 15892 | 14464 | 1264842 | 50.56% |
| TCGA_CS_5395 | 2180 | 86251 | 176 | 1222113 | 4.80% |
| TCGA_HT_7877 | 0 | 0 | 8176 | 1957904 | 0.00% |

============================================================

