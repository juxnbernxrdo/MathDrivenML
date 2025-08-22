# 🎵 Ejercicio Práctico: Sistema de Recomendación Musical con NumPy

## 📋 Contexto del Problema

Trabajas como Data Scientist en **SoundWave**, una plataforma de streaming musical similar a Spotify. Tu equipo necesita desarrollar un sistema de recomendación básico que analice los patrones de escucha de los usuarios para sugerir nuevas canciones.

El sistema debe procesar datos de reproducción musical, calcular similitudes entre usuarios, normalizar las puntuaciones de las canciones, y generar recomendaciones personalizadas usando únicamente **NumPy** (sin bibliotecas de machine learning externas).

---

## 🎯 Objetivos de Aprendizaje

Al completar este ejercicio practicarás:

- ✅ **Creación y manipulación de arrays multidimensionales**
- ✅ **Indexación avanzada y slicing**
- ✅ **Operaciones vectorizadas y broadcasting**
- ✅ **Funciones de agregación y estadísticas**
- ✅ **Álgebra lineal básica (productos matriciales, normas)**
- ✅ **Normalización y estandarización de datos**
- ✅ **Cálculo de similitudes y distancias**
- ✅ **Filtrado y ranking de resultados**

---

## 📊 Datos Iniciales

Comienza ejecutando el siguiente código para generar el dataset simulado:

```python
import numpy as np

# Configurar semilla para reproducibilidad
np.random.seed(42)

# Configuración del dataset
n_usuarios = 150
n_canciones = 50

# Nombres simulados de canciones (géneros musicales)
generos = ['Pop', 'Rock', 'Jazz', 'Electronic', 'Hip-Hop', 'Classical', 'Reggae', 'Folk', 'Blues', 'Country']
nombres_canciones = [f"{genre}_{i+1}" for genre in generos for i in range(5)]

# Matriz de reproducciones: usuarios x canciones
# Valores: número de veces que cada usuario reprodujo cada canción (0-100)
# Aproximadamente 70% de los valores serán 0 (canciones no escuchadas)
reproducciones = np.random.poisson(2, size=(n_usuarios, n_canciones))
reproducciones = np.where(np.random.random((n_usuarios, n_canciones)) > 0.3, 
                         reproducciones, 0)

# Información adicional de las canciones
duraciones = np.random.normal(180, 30, n_canciones)  # Duración en segundos
popularidad = np.random.beta(2, 5, n_canciones) * 100  # Popularidad 0-100

# Información de usuarios (edad)
edades_usuarios = np.random.randint(16, 65, n_usuarios)

print(f"📊 Dataset generado:")
print(f"   - {n_usuarios} usuarios")
print(f"   - {n_canciones} canciones")
print(f"   - Forma de matriz de reproducciones: {reproducciones.shape}")
print(f"   - Reproducciones totales: {np.sum(reproducciones):,}")
print(f"   - Sparsity (% de ceros): {(reproducciones == 0).mean()*100:.1f}%")
```

---

## 🛠️ Tareas a Resolver

### **Tarea 1: Análisis Exploratorio de Datos** 🔍

Implementa las siguientes funciones para entender mejor los datos:

```python
def analizar_dataset(reproducciones, nombres_canciones, edades_usuarios):
    """
    Analiza las características principales del dataset.
    
    Debes calcular y mostrar:
    1. Usuario más activo (más reproducciones totales)
    2. Canción más popular (más reproducciones entre todos los usuarios)
    3. Estadísticas básicas de reproducciones por usuario
    4. Distribución de edades de usuarios activos (que tienen >10 reproducciones totales)
    
    Returns:
        dict con las estadísticas calculadas
    """
    # TU CÓDIGO AQUÍ
    pass

# Ejecutar análisis
estadisticas = analizar_dataset(reproducciones, nombres_canciones, edades_usuarios)
```

**Pistas:**
- Usa `np.sum()` con diferentes ejes para agregaciones
- `np.argmax()` te ayudará a encontrar índices de valores máximos
- Combina indexación booleana para filtrar usuarios activos

---

### **Tarea 2: Normalización y Preprocesamiento** ⚖️

```python
def normalizar_datos(reproducciones):
    """
    Normaliza la matriz de reproducciones usando tres métodos diferentes.
    
    1. Normalización Min-Max por usuario (cada fila entre 0-1)
    2. Normalización Z-score por canción (cada columna)
    3. Normalización L2 por usuario (norma euclidiana = 1)
    
    Args:
        reproducciones: matriz (usuarios, canciones)
        
    Returns:
        tuple con las tres matrices normalizadas
    """
    # TU CÓDIGO AQUÍ
    pass

# Aplicar normalizaciones
repr_minmax, repr_zscore, repr_l2 = normalizar_datos(reproducciones)

# Verificar que las normalizaciones son correctas
print("🔍 Verificaciones:")
print(f"Min-Max - Rango por usuario: [{np.min(repr_minmax, axis=1).min():.3f}, {np.max(repr_minmax, axis=1).max():.3f}]")
print(f"Z-score - Media por canción: {np.mean(repr_zscore, axis=0).mean():.6f}")
print(f"L2 - Norma por usuario (primeros 5): {np.linalg.norm(repr_l2[:5], axis=1)}")
```

**Conceptos clave:**
- Broadcasting para operaciones usuario/canción
- `np.linalg.norm()` para normalización L2
- Manejo de divisiones por cero

---

### **Tarea 3: Cálculo de Similitudes** 🤝

```python
def calcular_similitudes(matriz_normalizada):
    """
    Calcula la similitud coseno entre todos los pares de usuarios.
    
    Similitud coseno = (A · B) / (||A|| * ||B||)
    
    Args:
        matriz_normalizada: matriz (usuarios, canciones) normalizada
        
    Returns:
        matriz de similitudes (usuarios, usuarios)
    """
    # TU CÓDIGO AQUÍ
    pass

def encontrar_usuarios_similares(similitudes, usuario_id, top_k=5):
    """
    Encuentra los K usuarios más similares a un usuario dado.
    
    Args:
        similitudes: matriz de similitudes
        usuario_id: índice del usuario objetivo
        top_k: número de usuarios similares a retornar
        
    Returns:
        array con índices de usuarios similares (excluyendo al usuario mismo)
    """
    # TU CÓDIGO AQUÍ
    pass

# Calcular similitudes
similitudes = calcular_similitudes(repr_l2)

# Encontrar usuarios similares para el usuario 0
usuarios_similares = encontrar_usuarios_similares(similitudes, usuario_id=0, top_k=5)
print(f"👥 Usuarios más similares al usuario 0: {usuarios_similares}")
print(f"📊 Similitudes: {similitudes[0, usuarios_similares]}")
```

---

### **Tarea 4: Sistema de Recomendación** 🎯

```python
def generar_recomendaciones(reproducciones, similitudes, usuario_id, n_recomendaciones=10):
    """
    Genera recomendaciones de canciones para un usuario específico.
    
    Algoritmo:
    1. Encuentra usuarios similares
    2. Para cada canción no escuchada por el usuario objetivo:
       - Calcula puntuación ponderada basada en usuarios similares
    3. Rankea y retorna las top N canciones
    
    Args:
        reproducciones: matriz original de reproducciones
        similitudes: matriz de similitudes entre usuarios
        usuario_id: usuario para quien generar recomendaciones
        n_recomendaciones: número de canciones a recomendar
        
    Returns:
        array con índices de canciones recomendadas (ordenadas por puntuación)
    """
    # TU CÓDIGO AQUÍ
    pass

def evaluar_recomendaciones(reproducciones, recomendaciones_indices, usuario_id, 
                          popularidad, nombres_canciones):
    """
    Evalúa la calidad de las recomendaciones generadas.
    
    Métricas a calcular:
    1. Diversidad de géneros en las recomendaciones
    2. Popularidad promedio de las canciones recomendadas
    3. Coverage: % del catálogo que incluyen las recomendaciones
    
    Args:
        reproducciones: matriz original
        recomendaciones_indices: índices de canciones recomendadas
        usuario_id: usuario objetivo
        popularidad: array con popularidad de cada canción
        nombres_canciones: lista con nombres de canciones
    """
    # TU CÓDIGO AQUÍ
    pass

# Generar recomendaciones para el usuario 0
recomendaciones = generar_recomendaciones(reproducciones, similitudes, usuario_id=0)

print(f"🎵 Top 10 recomendaciones para Usuario 0:")
for i, cancion_idx in enumerate(recomendaciones[:10]):
    print(f"   {i+1}. {nombres_canciones[cancion_idx]} (Popularidad: {popularidad[cancion_idx]:.1f})")

# Evaluar calidad
evaluar_recomendaciones(reproducciones, recomendaciones, 0, popularidad, nombres_canciones)
```

---

### **Tarea 5: Análisis Demográfico** 👥

```python
def analizar_preferencias_por_edad(reproducciones, edades_usuarios, nombres_canciones):
    """
    Analiza las preferencias musicales por grupos de edad.
    
    1. Divide usuarios en 3 grupos de edad: Jóvenes (16-25), Adultos (26-45), Mayores (46+)
    2. Calcula el promedio de reproducciones por canción para cada grupo
    3. Identifica las canciones favoritas de cada grupo etario
    4. Encuentra canciones que son populares en todos los grupos
    
    Returns:
        dict con análisis por grupo etario
    """
    # TU CÓDIGO AQUÍ
    pass

# Ejecutar análisis demográfico
analisis_edades = analizar_preferencias_por_edad(reproducciones, edades_usuarios, nombres_canciones)
```

---

## 🚀 Desafíos Extra (Nivel Avanzado)

Una vez completadas las tareas básicas, intenta resolver estos desafíos más complejos:

### **Desafío 1: Matriz de Factorización (Simplified NMF)** 🧮

```python
def factorizacion_matriz_simplificada(reproducciones, n_factores=10, n_iteraciones=100):
    """
    Implementa una versión simplificada de Non-Negative Matrix Factorization.
    
    Objetivo: Descomponer la matriz de reproducciones R ≈ U × V
    donde U es (usuarios × factores) y V es (factores × canciones)
    
    Esto permite:
    - Reducir dimensionalidad
    - Capturar patrones latentes
    - Generar mejores recomendaciones
    
    Algoritmo básico usando gradient descent
    """
    # DESAFÍO: Implementa el algoritmo de factorización
    pass
```

### **Desafío 2: Cold Start Problem** ❄️

```python
def recomendar_usuario_nuevo(reproducciones_historicas, perfil_nuevo_usuario, 
                            nombres_canciones, popularidad):
    """
    Maneja el problema de cold start: recomendar a usuarios completamente nuevos.
    
    Estrategias a implementar:
    1. Recomendaciones basadas en popularidad
    2. Recomendaciones basadas en géneros preferidos (inferir del perfil)
    3. Híbrido: combinar ambas estrategias
    
    Args:
        perfil_nuevo_usuario: array con pocas reproducciones del usuario nuevo
    """
    # DESAFÍO: Implementa estrategias para usuarios nuevos
    pass
```

### **Desafío 3: Optimización de Rendimiento** ⚡

```python
def optimizar_similitudes_sparse(reproducciones, threshold=0.1):
    """
    Optimiza el cálculo de similitudes para matrices sparse.
    
    Desafíos:
    1. Evita calcular similitudes para usuarios sin canciones en común
    2. Usa operaciones vectorizadas eficientes
    3. Implementa early stopping para similitudes muy bajas
    4. Compara tiempo de ejecución vs. método original
    """
    # DESAFÍO: Implementa versión optimizada
    pass

def benchmark_algoritmos(reproducciones, n_iteraciones=5):
    """
    Compara el rendimiento de diferentes implementaciones.
    """
    # DESAFÍO: Benchmarka diferentes enfoques
    pass
```

### **Desafío 4: Métricas de Evaluación Avanzadas** 📊

```python
def calcular_metricas_recomendacion(reproducciones_test, recomendaciones, usuario_id):
    """
    Calcula métricas avanzadas de sistemas de recomendación:
    
    1. Precision@K: % de canciones recomendadas que son relevantes
    2. Recall@K: % de canciones relevantes que fueron recomendadas  
    3. NDCG (Normalized Discounted Cumulative Gain)
    4. Serendipity: qué tan "sorprendentes" son las recomendaciones
    
    Requiere dividir datos en train/test
    """
    # DESAFÍO: Implementa métricas de evaluación rigurosas
    pass
```

---

## ✅ Criterios de Evaluación

Tu solución será evaluada en base a:

1. **Correctitud funcional** (40%): Los algoritmos producen resultados correctos
2. **Eficiencia de NumPy** (25%): Uso apropiado de vectorización y broadcasting
3. **Calidad del código** (20%): Código limpio, bien comentado y estructurado  
4. **Análisis de resultados** (15%): Interpretación correcta de los outputs

### **Puntos Bonus:**
- Implementar todos los desafíos extra (+20%)
- Visualizaciones creativas de los resultados (+10%)
- Optimizaciones de rendimiento no triviales (+10%)

---

## 🎯 Entregables

1. **Notebook de Jupyter** con todas las funciones implementadas
2. **Análisis de resultados** con interpretaciones de los outputs
3. **Documentación** de decisiones de diseño tomadas
4. **(Opcional)** Implementación de desafíos extra

---

**¡Buena suerte construyendo tu sistema de recomendación musical! 🎵**

*Recuerda: El objetivo es practicar NumPy, no crear el algoritmo más sofisticado. Enfócate en usar las capacidades de NumPy de manera efectiva y elegante.*
