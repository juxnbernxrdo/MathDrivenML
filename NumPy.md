# NumPy: La Base Fundamental para Machine Learning en Python

## Tabla de Contenidos

1. [¿Qué es NumPy?](#qué-es-numpy)
2. [Instalación y Configuración](#instalación-y-configuración)
3. [Primeros Pasos: Arrays de NumPy](#primeros-pasos-arrays-de-numpy)
4. [Creación de Arrays](#creación-de-arrays)
5. [Propiedades y Atributos de Arrays](#propiedades-y-atributos-de-arrays)
6. [Indexación y Slicing](#indexación-y-slicing)
7. [Manipulación de Formas (Reshape)](#manipulación-de-formas-reshape)
8. [Operaciones Matemáticas](#operaciones-matemáticas)
9. [Broadcasting: El Poder de NumPy](#broadcasting-el-poder-de-numpy)
10. [Funciones Estadísticas](#funciones-estadísticas)
11. [Álgebra Lineal](#álgebra-lineal)
12. [NumPy vs Listas de Python](#numpy-vs-listas-de-python)
13. [Casos Prácticos para Machine Learning](#casos-prácticos-para-machine-learning)
14. [Optimización y Mejores Prácticas](#optimización-y-mejores-prácticas)
15. [Errores Comunes y Cómo Evitarlos](#errores-comunes-y-cómo-evitarlos)
16. [Recursos Adicionales](#recursos-adicionales)

## ¿Qué es NumPy?

NumPy (Numerical Python) es la biblioteca fundamental para computación científica en Python. Proporciona:

- **Arrays multidimensionales eficientes**: Estructuras de datos optimizadas en C
- **Operaciones vectorizadas**: Cálculos rápidos sin bucles explícitos
- **Broadcasting**: Operaciones entre arrays de diferentes formas
- **Funciones matemáticas**: Amplia colección de funciones numéricas
- **Interoperabilidad**: Base para otras bibliotecas como Pandas, Scikit-learn, TensorFlow

### ¿Por qué es esencial para Machine Learning?

```python
# Ejemplo: Diferencia de rendimiento
import time
import numpy as np

# Con listas de Python
lista = list(range(1000000))
start = time.time()
resultado_lista = [x * 2 for x in lista]
tiempo_lista = time.time() - start

# Con NumPy
array = np.arange(1000000)
start = time.time()
resultado_numpy = array * 2
tiempo_numpy = time.time() - start

print(f"Tiempo con listas: {tiempo_lista:.4f}s")
print(f"Tiempo con NumPy: {tiempo_numpy:.4f}s")
print(f"NumPy es {tiempo_lista/tiempo_numpy:.1f}x más rápido")
```

```
Tiempo con listas: 0.0847s
Tiempo con NumPy: 0.0021s
NumPy es 40.3x más rápido
```

## Instalación y Configuración

### Instalación

```bash
# Instalación básica
pip install numpy

# Con Anaconda (recomendado para ML)
conda install numpy

# Verificar instalación
python -c "import numpy as np; print(np.__version__)"
```

### Importación convencional

```python
import numpy as np  # Convención estándar

# Verificar versión
print(f"NumPy versión: {np.__version__}")
```

## Primeros Pasos: Arrays de NumPy

### Array vs Lista

```python
# Lista de Python
lista_python = [1, 2, 3, 4, 5]

# Array de NumPy
array_numpy = np.array([1, 2, 3, 4, 5])

print(f"Lista Python: {lista_python}")
print(f"Array NumPy: {array_numpy}")
print(f"Tipo de array: {type(array_numpy)}")
print(f"Tipo de datos: {array_numpy.dtype}")
```

```
Lista Python: [1, 2, 3, 4, 5]
Array NumPy: [1 2 3 4 5]
Tipo de array: <class 'numpy.ndarray'>
Tipo de datos: int64
```

## Creación de Arrays

### Métodos Básicos

```python
# Desde listas
arr_1d = np.array([1, 2, 3, 4])
arr_2d = np.array([[1, 2], [3, 4]])
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"1D: {arr_1d}")
print(f"2D:\n{arr_2d}")
print(f"3D:\n{arr_3d}")
```

### Funciones de Creación

```python
# Arrays de ceros y unos
zeros = np.zeros((3, 4))          # 3x4 matriz de ceros
ones = np.ones((2, 3, 4))         # 2x3x4 tensor de unos
full = np.full((2, 2), 7)         # 2x2 matriz llena de 7s

print("Matriz de ceros (3x4):")
print(zeros)

print("\nMatriz llena de 7s:")
print(full)
```

```python
# Secuencias numéricas
arange_arr = np.arange(0, 10, 2)           # [0, 2, 4, 6, 8]
linspace_arr = np.linspace(0, 1, 5)        # 5 valores entre 0 y 1
logspace_arr = np.logspace(0, 2, 3)        # 3 valores logarítmicos

print(f"arange: {arange_arr}")
print(f"linspace: {linspace_arr}")
print(f"logspace: {logspace_arr}")
```

### Arrays Aleatorios (Importantes para ML)

```python
# Configurar semilla para reproducibilidad
np.random.seed(42)

# Diferentes distribuciones
uniform = np.random.uniform(0, 1, (2, 3))      # Distribución uniforme
normal = np.random.normal(0, 1, (2, 3))        # Distribución normal
randint = np.random.randint(0, 10, (2, 3))     # Enteros aleatorios

print("Distribución uniforme:")
print(uniform)

print("\nDistribución normal (media=0, std=1):")
print(normal)
```

### Matrices Especiales

```python
# Matriz identidad (crucial para álgebra lineal)
identity = np.eye(3)

# Matriz diagonal
diagonal = np.diag([1, 2, 3, 4])

print("Matriz identidad 3x3:")
print(identity)

print("\nMatriz diagonal:")
print(diagonal)
```

## Propiedades y Atributos de Arrays

```python
# Crear array de ejemplo
data = np.random.randn(3, 4, 2)

# Propiedades importantes
print(f"Forma (shape): {data.shape}")           # Dimensiones
print(f"Número de dimensiones: {data.ndim}")    # Cantidad de ejes
print(f"Tamaño total: {data.size}")             # Total de elementos
print(f"Tipo de datos: {data.dtype}")           # Tipo de datos
print(f"Tamaño en bytes: {data.nbytes}")        # Memoria utilizada
print(f"Itemsize: {data.itemsize}")             # Bytes por elemento
```

```
Forma (shape): (3, 4, 2)
Número de dimensiones: 3
Tamaño total: 24
Tipo de datos: float64
Tamaño en bytes: 192
Itemsize: 8
```

### Tipos de Datos Específicos

```python
# Especificar tipo de datos (importante para optimización)
int_array = np.array([1, 2, 3], dtype=np.int32)
float_array = np.array([1, 2, 3], dtype=np.float32)
bool_array = np.array([True, False, True], dtype=np.bool_)

print(f"Enteros 32-bit: {int_array.dtype}")
print(f"Flotantes 32-bit: {float_array.dtype}")
print(f"Booleanos: {bool_array.dtype}")

# Conversión de tipos
converted = int_array.astype(np.float64)
print(f"Después de conversión: {converted.dtype}")
```

## Indexación y Slicing

### Indexación Básica (1D)

```python
arr = np.array([10, 20, 30, 40, 50])

print(f"Elemento en índice 0: {arr[0]}")        # 10
print(f"Elemento en índice -1: {arr[-1]}")      # 50
print(f"Slice [1:4]: {arr[1:4]}")               # [20 30 40]
print(f"Cada segundo elemento: {arr[::2]}")      # [10 30 50]
```

### Indexación Multidimensional

```python
# Array 2D
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(f"Elemento [1, 2]: {matrix[1, 2]}")       # 6
print(f"Fila 1: {matrix[1, :]}")                # [4 5 6]
print(f"Columna 2: {matrix[:, 2]}")             # [3 6 9]
print(f"Submatriz [0:2, 1:3]:")
print(matrix[0:2, 1:3])
```

```
Elemento [1, 2]: 6
Fila 1: [4 5 6]
Columna 2: [3 6 9]
Submatriz [0:2, 1:3]:
[[2 3]
 [5 6]]
```

### Indexación Booleana (Muy útil en ML)

```python
# Array de ejemplo
data = np.array([1, -2, 3, -4, 5, -6])

# Crear máscara booleana
positive_mask = data > 0

print(f"Máscara booleana: {positive_mask}")
print(f"Elementos positivos: {data[positive_mask]}")

# Operación en una línea
negative_values = data[data < 0]
print(f"Valores negativos: {negative_values}")

# Modificación condicional
data_copy = data.copy()
data_copy[data_copy < 0] = 0  # Reemplazar negativos con 0
print(f"Después de reemplazar: {data_copy}")
```

### Indexación Avanzada

```python
# Indexación con arrays de enteros
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])

print(f"Elementos en índices [0, 2, 4]: {arr[indices]}")

# Indexación 2D con arrays
matrix = np.arange(12).reshape(3, 4)
rows = np.array([0, 2])
cols = np.array([1, 3])

print(f"Matriz original:")
print(matrix)
print(f"Elementos [0,1] y [2,3]: {matrix[rows, cols]}")
```

## Manipulación de Formas (Reshape)

### Reshape Básico

```python
# Array original
original = np.arange(12)
print(f"Original (shape {original.shape}): {original}")

# Diferentes formas
reshaped_2d = original.reshape(3, 4)
reshaped_3d = original.reshape(2, 2, 3)

print(f"\nReshape a (3, 4):")
print(reshaped_2d)

print(f"\nReshape a (2, 2, 3):")
print(reshaped_3d)
```

### Funciones de Manipulación

```python
# Datos de ejemplo
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# Aplanar (flatten)
flattened = matrix.flatten()  # Copia
raveled = matrix.ravel()      # Vista (más eficiente)

print(f"Original shape: {matrix.shape}")
print(f"Flattened: {flattened}")

# Transponer
transposed = matrix.T
print(f"\nTranspuesta shape: {transposed.shape}")
print(f"Transpuesta:")
print(transposed)

# Agregar/eliminar dimensiones
expanded = np.expand_dims(matrix, axis=0)  # Agregar dimensión
squeezed = np.squeeze(expanded)            # Eliminar dimensiones de tamaño 1

print(f"\nExpanded shape: {expanded.shape}")
print(f"Squeezed shape: {squeezed.shape}")
```

## Operaciones Matemáticas

### Operaciones Elemento por Elemento

```python
# Arrays de ejemplo
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Operaciones aritméticas básicas
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")    # Multiplicación elemento por elemento
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")  # Potencia

# Comparaciones
print(f"a > 2: {a > 2}")
print(f"a == b: {a == b}")
```

### Funciones Matemáticas Universales

```python
# Array de ejemplo
x = np.array([0, np.pi/2, np.pi])

# Funciones trigonométricas
print(f"sin(x): {np.sin(x)}")
print(f"cos(x): {np.cos(x)}")

# Funciones exponenciales y logarítmicas
y = np.array([1, 2, 3])
print(f"exp(y): {np.exp(y)}")
print(f"log(y): {np.log(y)}")
print(f"sqrt(y): {np.sqrt(y)}")

# Funciones de redondeo
z = np.array([1.2, 2.7, 3.1])
print(f"round(z): {np.round(z)}")
print(f"ceil(z): {np.ceil(z)}")
print(f"floor(z): {np.floor(z)}")
```

### Operaciones de Agregación

```python
# Array 2D de ejemplo
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Agregaciones sobre todo el array
print(f"Suma total: {np.sum(data)}")
print(f"Media: {np.mean(data)}")
print(f"Máximo: {np.max(data)}")
print(f"Mínimo: {np.min(data)}")

# Agregaciones por eje
print(f"Suma por filas (axis=1): {np.sum(data, axis=1)}")
print(f"Suma por columnas (axis=0): {np.sum(data, axis=0)}")
print(f"Media por columnas: {np.mean(data, axis=0)}")
```

## Broadcasting: El Poder de NumPy

Broadcasting permite realizar operaciones entre arrays de diferentes formas sin copiar datos explícitamente.

### Reglas de Broadcasting

```python
# Ejemplos de broadcasting válido
a = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)

b = np.array([10, 20, 30])  # Shape: (3,)

# NumPy automáticamente "expande" b a (2, 3)
result = a + b
print(f"a + b:")
print(result)

# Scalar broadcasting
scalar = 5
result_scalar = a * scalar
print(f"\na * 5:")
print(result_scalar)
```

### Broadcasting con Diferentes Dimensiones

```python
# Broadcasting con arrays de diferentes formas
matrix = np.arange(12).reshape(3, 4)  # (3, 4)
column = np.array([[10], [20], [30]]) # (3, 1)
row = np.array([1, 2, 3, 4])          # (4,)

print("Matriz original (3, 4):")
print(matrix)

print("\nSumar columna (3, 1):")
print(matrix + column)

print("\nSumar fila (4,):")
print(matrix + row)
```

### Broadcasting en ML: Normalización

```python
# Ejemplo práctico: normalización de características
# Simular dataset con 100 muestras y 3 características
np.random.seed(42)
dataset = np.random.randn(100, 3) * [10, 5, 2] + [50, 25, 10]

# Calcular media y desviación estándar por característica
mean = np.mean(dataset, axis=0)  # Shape: (3,)
std = np.std(dataset, axis=0)    # Shape: (3,)

# Normalización Z-score usando broadcasting
normalized = (dataset - mean) / std  # Broadcasting automático

print(f"Forma original: {dataset.shape}")
print(f"Media por característica: {mean}")
print(f"Std por característica: {std}")
print(f"Media después de normalizar: {np.mean(normalized, axis=0)}")
print(f"Std después de normalizar: {np.std(normalized, axis=0)}")
```

## Funciones Estadísticas

### Estadísticas Descriptivas

```python
# Dataset de ejemplo
np.random.seed(42)
data = np.random.normal(10, 2, (5, 4))  # Media=10, std=2

print("Dataset de ejemplo:")
print(data)

# Estadísticas básicas
print(f"\nMedia: {np.mean(data):.2f}")
print(f"Mediana: {np.median(data):.2f}")
print(f"Desviación estándar: {np.std(data):.2f}")
print(f"Varianza: {np.var(data):.2f}")

# Percentiles
print(f"Percentil 25: {np.percentile(data, 25):.2f}")
print(f"Percentil 75: {np.percentile(data, 75):.2f}")

# Valores extremos
print(f"Mínimo: {np.min(data):.2f}")
print(f"Máximo: {np.max(data):.2f}")
print(f"Rango: {np.ptp(data):.2f}")  # Peak to peak
```

### Estadísticas por Eje

```python
# Estadísticas por filas y columnas
print("Estadísticas por columna (axis=0):")
print(f"Media: {np.mean(data, axis=0)}")
print(f"Std: {np.std(data, axis=0)}")

print("\nEstadísticas por fila (axis=1):")
print(f"Media: {np.mean(data, axis=1)}")
print(f"Std: {np.std(data, axis=1)}")
```

### Correlación y Covarianza

```python
# Matriz de correlación (importante para análisis de características)
features = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 100)

correlation_matrix = np.corrcoef(features.T)
covariance_matrix = np.cov(features.T)

print("Matriz de correlación:")
print(correlation_matrix)

print("\nMatriz de covarianza:")
print(covariance_matrix)
```

## Álgebra Lineal

### Operaciones Básicas

```python
# Matrices de ejemplo
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Multiplicación matricial
matrix_mult = np.dot(A, B)  # o A @ B
print("Multiplicación matricial A @ B:")
print(matrix_mult)

# Producto elemento por elemento (Hadamard)
element_mult = A * B
print("\nProducto elemento por elemento A * B:")
print(element_mult)
```

### Operaciones Avanzadas

```python
# Matriz cuadrada para operaciones avanzadas
matrix = np.array([[4, 2],
                   [1, 3]])

# Determinante
det = np.linalg.det(matrix)
print(f"Determinante: {det}")

# Inversa
inverse = np.linalg.inv(matrix)
print(f"Matriz inversa:")
print(inverse)

# Verificar: A @ A^-1 = I
identity_check = matrix @ inverse
print(f"Verificación A @ A^-1:")
print(identity_check)

# Valores y vectores propios
eigenvals, eigenvecs = np.linalg.eig(matrix)
print(f"Valores propios: {eigenvals}")
print(f"Vectores propios:")
print(eigenvecs)
```

### Descomposiciones Útiles en ML

```python
# SVD (Singular Value Decomposition) - usado en PCA
matrix = np.random.randn(4, 3)

U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

print(f"Forma original: {matrix.shape}")
print(f"U shape: {U.shape}")
print(f"S shape: {S.shape}")
print(f"Vt shape: {Vt.shape}")

# Reconstrucción
reconstructed = U @ np.diag(S) @ Vt
print(f"Error de reconstrucción: {np.allclose(matrix, reconstructed)}")
```

## NumPy vs Listas de Python

### Comparación de Rendimiento

```python
import time

# Configuración
size = 1000000
python_list = list(range(size))
numpy_array = np.arange(size)

# Operación: elevar al cuadrado
# Con listas de Python
start = time.time()
squared_list = [x**2 for x in python_list]
python_time = time.time() - start

# Con NumPy
start = time.time()
squared_array = numpy_array**2
numpy_time = time.time() - start

print(f"Tiempo Python: {python_time:.4f}s")
print(f"Tiempo NumPy: {numpy_time:.4f}s")
print(f"Aceleración: {python_time/numpy_time:.1f}x")
```

### Comparación de Memoria

```python
import sys

# Comparar uso de memoria
python_list = [1.0] * 1000
numpy_array = np.ones(1000, dtype=np.float64)

list_size = sys.getsizeof(python_list) + sum(sys.getsizeof(x) for x in python_list)
array_size = numpy_array.nbytes

print(f"Lista Python: {list_size} bytes")
print(f"Array NumPy: {array_size} bytes")
print(f"Eficiencia de memoria: {list_size/array_size:.1f}x menos memoria con NumPy")
```

### Cuándo Usar Cada Uno

| Aspecto | Listas Python | Arrays NumPy |
|---------|---------------|--------------|
| **Velocidad** | Lenta para operaciones numéricas | Muy rápida (optimizado en C) |
| **Memoria** | Mayor consumo | Menor consumo |
| **Flexibilidad** | Tipos mixtos, tamaño variable | Tipo homogéneo, tamaño fijo |
| **Operaciones** | Requiere bucles explícitos | Operaciones vectorizadas |
| **Uso ideal** | Datos heterogéneos, estructuras simples | Computación numérica, ML |

## Casos Prácticos para Machine Learning

### 1. Preparación de Datos

```python
# Simular dataset de precios de casas
np.random.seed(42)

# Características: área, habitaciones, edad, distancia_centro
n_samples = 1000
area = np.random.normal(150, 50, n_samples)
rooms = np.random.randint(1, 6, n_samples)
age = np.random.exponential(10, n_samples)
distance = np.random.gamma(2, 2, n_samples)

# Combinar características
features = np.column_stack([area, rooms, age, distance])
print(f"Shape del dataset: {features.shape}")

# Generar precio (variable objetivo)
# Precio = 1000 + 5*área + 10000*habitaciones - 500*edad - 1000*distancia + ruido
price = (1000 + 5*area + 10000*rooms - 500*age - 1000*distance + 
         np.random.normal(0, 5000, n_samples))

print(f"Dataset shape: {features.shape}")
print(f"Target shape: {price.shape}")
print("Primeras 5 filas:")
print(features[:5])
```

### 2. Normalización y Estandarización

```python
# Diferentes métodos de normalización

# Min-Max Scaling (0-1)
def min_max_scale(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Z-score Standardization
def z_score_normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

# Robust Scaling (usa mediana y percentiles)
def robust_scale(X):
    median = np.median(X, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    iqr = q75 - q25
    return (X - median) / iqr

# Aplicar normalizaciones
features_minmax = min_max_scale(features)
features_zscore = z_score_normalize(features)
features_robust = robust_scale(features)

print("Estadísticas originales:")
print(f"Media: {features.mean(axis=0)}")
print(f"Std: {features.std(axis=0)}")

print("\nDespués de Z-score:")
print(f"Media: {features_zscore.mean(axis=0)}")
print(f"Std: {features_zscore.std(axis=0)}")
```

### 3. División de Datos (Train/Validation/Test)

```python
def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """División en conjuntos de entrenamiento, validación y prueba"""
    n_samples = X.shape[0]
    
    # Índices aleatorios
    indices = np.random.permutation(n_samples)
    
    # Calcular puntos de corte
    train_end = int(train_ratio * n_samples)
    val_end = int((train_ratio + val_ratio) * n_samples)
    
    # Dividir índices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])

# Aplicar división
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
    features_zscore, price)

print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Validación: {X_val.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")
```

### 4. Métricas de Evaluación

```python
def calculate_metrics(y_true, y_pred):
    """Calcular métricas de regresión usando NumPy"""
    
    # Error absoluto medio
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Error cuadrático medio
    mse = np.mean((y_true - y_pred)**2)
    
    # Raíz del error cuadrático medio
    rmse = np.sqrt(mse)
    
    # R² (coeficiente de determinación)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# Simular predicciones (modelo lineal simple)
# y = X @ weights + bias
weights = np.array([5, 10000, -500, -1000])
bias = 1000
y_pred = X_val @ weights + bias + np.random.normal(0, 1000, X_val.shape[0])

metrics = calculate_metrics(y_val, y_pred)
print("Métricas del modelo:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}")
```

### 5. Implementación de Gradiente Descendente

```python
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """Implementación básica de gradiente descendente"""
    
    # Agregar columna de bias (intercept)
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    
    # Inicializar parámetros aleatoriamente
    theta = np.random.randn(X_with_bias.shape[1])
    
    # Almacenar historial de costos
    costs = []
    
    m = X.shape[0]  # número de muestras
    
    for epoch in range(epochs):
        # Predicciones
        predictions = X_with_bias @ theta
        
        # Costo (MSE)
        cost = np.mean((predictions - y)**2)
        costs.append(cost)
        
        # Gradientes
        gradients = (2/m) *