# 📐 Documentación Completa sobre Vectores

## 🎯 Tabla de Contenidos
- [1. Definición de Vector](#1-definición-de-vector)
- [2. Operaciones Básicas con Vectores](#2-operaciones-básicas-con-vectores)
- [3. Propiedades Importantes](#3-propiedades-importantes)
- [4. Magnitud (Norma) de un Vector](#4-magnitud-norma-de-un-vector)
- [5. Dirección Vectorial](#5-dirección-vectorial)
- [6. Producto Punto (Dot Product)](#6-producto-punto-dot-product)
- [7. Proyección Vectorial](#7-proyección-vectorial)
- [8. Distancia entre dos vectores](#8-distancia-entre-dos-vectores-puntos)
- [9. Vectores Unitarios](#9-vectores-unitarios)
- [10. Implementación en Python](#10-implementación-en-python)
- [11. Aplicaciones en Machine Learning](#11-aplicaciones-en-machine-learning)
- [12. Ejercicios Prácticos](#12-ejercicios-prácticos)

---

## 1. Definición de Vector

Un **vector** en 2D es una entidad matemática que tiene **magnitud** (longitud) y **dirección**.

Se representa como un par ordenado:

**v = (x, y)**

Donde:
- `x` es la componente en el eje X
- `y` es la componente en el eje Y

### 🔍 Interpretaciones del Vector
- **Geométrica**: Flecha desde el origen hasta el punto (x, y)
- **Física**: Representa magnitudes como velocidad, fuerza, aceleración
- **Algebraica**: Elemento de un espacio vectorial

---

## 2. Operaciones Básicas con Vectores

### 2.1. Suma de Vectores

Dados dos vectores **A = (ax, ay)** y **B = (bx, by)**:

**A + B = (ax + bx, ay + by)**

#### 📊 Interpretación Geométrica
La suma se realiza colocando el vector **B** en la punta del vector **A** (método del paralelogramo).

### 2.2. Resta de Vectores

**A - B = (ax - bx, ay - by)**

### 2.3. Multiplicación por Escalar

Dado un vector **v = (x, y)** y un escalar **k**:

**k × v = (k × x, k × y)**

#### 📈 Efectos del Escalamiento
- Si `k > 1`: el vector se **alarga**
- Si `0 < k < 1`: el vector se **acorta**
- Si `k = 0`: el vector se convierte en el **vector cero** (0,0)
- Si `k < 0`: el vector **cambia de dirección** y su magnitud se escala por |k|

---

## 3. Propiedades Importantes

### 3.1. Propiedades de la Suma
- **Conmutatividad**: **A + B = B + A**
- **Asociatividad**: **(A + B) + C = A + (B + C)**
- **Elemento neutro**: **v + 0 = v**
- **Elemento inverso**: **v + (-v) = 0**

### 3.2. Propiedades del Producto por Escalar
- **Distributividad respecto a vectores**: **k(A + B) = kA + kB**
- **Distributividad respecto a escalares**: **(k + m)v = kv + mv**
- **Asociatividad**: **(km)v = k(mv)**
- **Elemento neutro**: **1 · v = v**

---

## 4. Magnitud (Norma) de un Vector

La magnitud o norma de un vector **v = (x, y)** es:

**||v|| = √(x² + y²)**

### 🧮 Propiedades de la Norma
- `||v|| ≥ 0` (no negatividad)
- `||v|| = 0` si y solo si `v = 0`
- `||kv|| = |k|||v||`
- `||u + v|| ≤ ||u|| + ||v||` (desigualdad triangular)

---

## 5. Dirección Vectorial

El ángulo `θ` que un vector **v = (x, y)** forma con el eje X positivo:

**θ = arctan(y/x)**

### ⚠️ Función Recomendada
Para evitar errores en cuadrantes, usar:

**θ = atan2(y, x)**

**Conversión a grados:**

**θ_grados = θ_radianes × (180/π)**

---

## 6. Producto Punto (Dot Product)

Dados **A = (ax, ay)** y **B = (bx, by)**:

**A · B = ax × bx + ay × by**

### 🎨 Interpretación Geométrica

**A · B = ||A|| ||B|| cos(θ)**

Donde `θ` es el ángulo entre **A** y **B**.

### 📋 Propiedades del Producto Punto
- **Conmutatividad**: **A · B = B · A**
- **Distributividad**: **A · (B + C) = A · B + A · C**
- **Asociatividad con escalares**: **(kA) · B = k(A · B)**
- **Ortogonalidad**: Si **A · B = 0**, los vectores son perpendiculares

---

## 7. Proyección Vectorial

La proyección de un vector **A** sobre otro vector **B** es:

**proj_B(A) = ((A · B) / ||B||²) × B**

### 📐 Interpretación
Representa la **componente** de **A** en la dirección de **B**.

### 🔢 Componente Escalar
La magnitud de la proyección es:

**comp_B(A) = (A · B) / ||B||**

---

## 8. Distancia entre dos vectores (puntos)

Dados **P = (px, py)** y **Q = (qx, qy)**:

**d(P, Q) = √((px - qx)² + (py - qy)²)**

Esta es la **distancia euclidiana** entre dos puntos.

---

## 9. Vectores Unitarios

Un **vector unitario** tiene magnitud igual a 1:

**||û|| = 1**

### 🎯 Normalización
Para convertir cualquier vector **v** en unitario:

**v̂ = v / ||v||**

### 📊 Vectores Unitarios Básicos
- **î = (1, 0)** - dirección del eje X
- **ĵ = (0, 1)** - dirección del eje Y

---

## 10. Implementación en Python
```python
import numpy as np
import matplotlib.pyplot as plt

def plot_vectors(vectors, colors=None, labels=None, show_info=False):
    """
    Grafica vectores 2D o 3D desde el origen.
    
    Parámetros:
    - vectors: lista de arrays o tuplas con componentes de vectores [(x1, y1), (x2, y2), ...] o [(x1, y1, z1), ...]
    - colors: lista opcional de colores para cada vector ['r', 'g', 'b', ...]
    - labels: lista opcional de etiquetas para cada vector ['v1', 'v2', ...]
    - show_info: bool, si True muestra magnitud y dirección en la etiqueta
    """
    
    # Validación básica
    if not vectors:
        raise ValueError("La lista de vectores está vacía.")
    
    # Convertir a np.array y validar dimensiones
    vectors = [np.array(v) for v in vectors]
    dim = vectors[0].shape[0]
    
    if any(v.shape[0] != dim for v in vectors):
        raise ValueError("Todos los vectores deben tener la misma dimensión.")
    if dim not in (2, 3):
        raise ValueError("Solo se soportan vectores 2D o 3D.")
    
    # Preparar figura
    fig = plt.figure(figsize=(8, 8))
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    origin = np.zeros(dim)
    
    # Colores y etiquetas por defecto
    if colors is None:
        colors = plt.cm.get_cmap('tab10').colors
    if labels is None:
        labels = [f'Vector {i+1}' for i in range(len(vectors))]
    
    # Ajustar límites y graficar
    all_coords = np.array(vectors)
    max_val = np.max(np.abs(all_coords)) * 1.2
    
    if dim == 2:
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    else:
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    # Graficar vectores
    for i, v in enumerate(vectors):
        c = colors[i % len(colors)]
        label = labels[i]
        mag = np.linalg.norm(v)
        
        if dim == 2:
            ax.quiver(*origin, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=c)
            if show_info:
                angle_deg = np.degrees(np.arctan2(v[1], v[0]))
                label += f"\n|v|={mag:.2f}, θ={angle_deg:.1f}°"
            ax.text(v[0]*1.05, v[1]*1.05, label, color=c, fontsize=10)
        else:
            ax.quiver(*origin, v[0], v[1], v[2], length=mag, color=c, normalize=True)
            if show_info:
                label += f"\n|v|={mag:.2f}"
            ax.text(v[0]*1.05, v[1]*1.05, v[2]*1.05, label, color=c, fontsize=10)
    
    ax.set_title(f'Visualización de Vectores {dim}D')
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
v2d_1 = (3, 4)
v2d_2 = (-2, 5)
v2d_3 = (5, -3)

v3d_1 = (1, 2, 3)
v3d_2 = (-1, 0, 4)
v3d_3 = (3, -2, 1)

# Visualizar 2D
plot_vectors([v2d_1, v2d_2, v2d_3], show_info=True)

# Visualizar 3D
plot_vectors([v3d_1, v3d_2, v3d_3], show_info=True)
```

---

## 11. Aplicaciones en Machine Learning

### 11.1. Representación de Datos
```python
# Los datos se representan como vectores de características
data_point = Vector2D(height=175, weight=70)  # Ejemplo: altura y peso
```

### 11.2. Distancia Euclidiana para Clustering
```python
def euclidean_distance(point1, point2):
    """Calcula distancia euclidiana entre dos puntos"""
    return point1.distance_to(point2)

# K-Means utiliza esta distancia para agrupar datos
```

### 11.3. Producto Punto en Redes Neuronales
```python
def linear_layer(input_vector, weight_vector, bias):
    """Capa lineal básica de una red neuronal"""
    return input_vector.dot(weight_vector) + bias
```

### 11.4. Gradiente Descent
```python
def update_weights(weights, gradient, learning_rate):
    """Actualización de pesos usando gradiente descendente"""
    return weights - gradient * learning_rate
```

### 11.5. Cosine Similarity (Similaridad del Coseno)
```python
def cosine_similarity(v1, v2):
    """
    Calcula la similaridad del coseno entre dos vectores
    Formula: cos(θ) = (A · B) / (||A|| ||B||)
    Resultado: 1 = iguales, 0 = perpendiculares, -1 = opuestos
    """
    dot_product = v1.dot(v2)
    magnitude_product = v1.magnitude() * v2.magnitude()
    
    if magnitude_product == 0:
        return 0
    
    return dot_product / magnitude_product

# Ejemplo de uso en recomendaciones
user_preferences = Vector2D(0.8, 0.3)  # Le gustan películas de acción, poco romance
movie_features = Vector2D(0.9, 0.1)    # Película de acción con poco romance

similarity = cosine_similarity(user_preferences, movie_features)
print(f"Similaridad: {similarity:.3f}")  # Valor alto = buena recomendación
```

---

## 12. Ejercicios Prácticos

### 🎯 Ejercicio 1: Operaciones Básicas
Dados los vectores **a = (2, -3)** y **b = (-1, 4)**:
1. Calcula **a + b**
2. Calcula **a - b**
3. Calcula **3a**
4. Encuentra **||a||** y **||b||**

### 🎯 Ejercicio 2: Producto Punto
1. Calcula **a · b**
2. Determina si los vectores son perpendiculares
3. Calcula el ángulo entre los vectores

### 🎯 Ejercicio 3: Proyección
1. Proyecta **a** sobre **b**
2. Proyecta **b** sobre **a**
3. Visualiza ambas proyecciones

### 🎯 Ejercicio 4: Aplicación Práctica
Implementa un algoritmo de K-Means simple que:
1. Use vectores para representar puntos de datos
2. Calcule distancias euclidianas
3. Actualice centroides como promedios vectoriales

### 🎯 Ejercicio 5: Sistema de Recomendación Simple
Crea un sistema que:
1. Represente usuarios y elementos como vectores de características
2. Use similaridad del coseno para hacer recomendaciones
3. Encuentre los elementos más similares a las preferencias del usuario

### 💡 Soluciones

```python
# Clase Vector2D para operaciones vectoriales
import math

class Vector2D:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    
    def __add__(self, other):
        """Suma de vectores"""
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Resta de vectores"""
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Multiplicación por escalar"""
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        """Multiplicación por escalar (orden inverso)"""
        return self * scalar
    
    def __str__(self):
        """Representación en string"""
        return f"({self.x:.2f}, {self.y:.2f})"
    
    def __repr__(self):
        """Representación para debugging"""
        return f"Vector2D({self.x}, {self.y})"
    
    def magnitude(self):
        """Magnitud del vector"""
        return math.sqrt(self.x**2 + self.y**2)
    
    def dot(self, other):
        """Producto punto con otro vector"""
        return self.x * other.x + self.y * other.y
    
    def project_onto(self, other):
        """Proyección de este vector sobre otro"""
        if other.magnitude() == 0:
            return Vector2D(0, 0)
        scalar = self.dot(other) / (other.magnitude() ** 2)
        return other * scalar
    
    def distance_to(self, other):
        """Distancia euclidiana a otro vector"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def normalize(self):
        """Vector unitario en la misma dirección"""
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

def plot_vectors(*vectors, labels=None, title='Vectores'):
    """
    Función para visualizar vectores usando matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Configurar la figura
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # Colores para los vectores
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Graficar cada vector desde el origen
        for i, vector in enumerate(vectors):
            color = colors[i % len(colors)]
            label = labels[i] if labels and i < len(labels) else f'Vector {i+1}'
            
            # Graficar el vector como una flecha
            ax.quiver(0, 0, vector.x, vector.y, 
                     angles='xy', scale_units='xy', scale=1, 
                     color=color, label=label, alpha=0.7)
            
            # Agregar etiqueta al final del vector
            ax.text(vector.x * 1.05, vector.y * 1.05, label, 
                   fontsize=10, ha='center', va='center')
        
        # Configurar ejes
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib no está disponible. No se puede mostrar la gráfica.")
        print("Instala matplotlib con: pip install matplotlib")

# Solución Ejercicio 1
print("=== EJERCICIO 1: OPERACIONES BÁSICAS ===")
a = Vector2D(2, -3)
b = Vector2D(-1, 4)

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"3a = {a * 3}")
print(f"||a|| = {a.magnitude():.2f}")
print(f"||b|| = {b.magnitude():.2f}")

# Solución Ejercicio 2
print("\n=== EJERCICIO 2: PRODUCTO PUNTO ===")
dot_ab = a.dot(b)
print(f"a · b = {dot_ab}")
print(f"Perpendiculares: {'Sí' if dot_ab == 0 else 'No'}")

import math
cos_theta = dot_ab / (a.magnitude() * b.magnitude())
angle_rad = math.acos(abs(cos_theta))  # abs() para evitar errores de redondeo
angle_deg = math.degrees(angle_rad)
print(f"Ángulo: {angle_deg:.2f}°")

# Solución Ejercicio 3
print("\n=== EJERCICIO 3: PROYECCIÓN ===")
proj_a_on_b = a.project_onto(b)
proj_b_on_a = b.project_onto(a)
print(f"Proyección de a sobre b: {proj_a_on_b}")
print(f"Proyección de b sobre a: {proj_b_on_a}")

# Visualización de proyecciones
plot_vectors(a, b, proj_a_on_b, proj_b_on_a,
            labels=['Vector a', 'Vector b', 'proj_b(a)', 'proj_a(b)'],
            title='Proyecciones Vectoriales')

# Solución Ejercicio 4: K-Means Simple
print("\n=== EJERCICIO 4: K-MEANS SIMPLE ===")

class KMeansSimple:
    def __init__(self, k=2):
        self.k = k
        self.centroids = []
    
    def fit(self, points, max_iterations=100):
        # Inicializar centroides aleatoriamente
        import random
        self.centroids = [Vector2D(random.uniform(-5, 5), random.uniform(-5, 5)) 
                         for _ in range(self.k)]
        
        for iteration in range(max_iterations):
            # Asignar puntos a centroides más cercanos
            clusters = [[] for _ in range(self.k)]
            
            for point in points:
                distances = [point.distance_to(centroid) for centroid in self.centroids]
                closest_centroid = distances.index(min(distances))
                clusters[closest_centroid].append(point)
            
            # Actualizar centroides
            new_centroids = []
            for i, cluster in enumerate(clusters):
                if cluster:  # Si el cluster no está vacío
                    avg_x = sum(p.x for p in cluster) / len(cluster)
                    avg_y = sum(p.y for p in cluster) / len(cluster)
                    new_centroids.append(Vector2D(avg_x, avg_y))
                else:
                    new_centroids.append(self.centroids[i])  # Mantener centroide anterior
            
            # Verificar convergencia
            converged = all(
                old.distance_to(new) < 0.01 
                for old, new in zip(self.centroids, new_centroids)
            )
            
            self.centroids = new_centroids
            
            if converged:
                print(f"Convergencia alcanzada en iteración {iteration + 1}")
                break
        
        return self.centroids

# Datos de prueba
data_points = [
    Vector2D(1, 2), Vector2D(2, 1), Vector2D(2, 3),  # Cluster 1
    Vector2D(8, 8), Vector2D(9, 9), Vector2D(7, 9)   # Cluster 2
]

kmeans = KMeansSimple(k=2)
final_centroids = kmeans.fit(data_points)

print("Centroides finales:")
for i, centroid in enumerate(final_centroids):
    print(f"Centroide {i+1}: {centroid}")

# Solución Ejercicio 5: Sistema de Recomendación
print("\n=== EJERCICIO 5: SISTEMA DE RECOMENDACIÓN ===")

def cosine_similarity(v1, v2):
    dot_product = v1.dot(v2)
    magnitude_product = v1.magnitude() * v2.magnitude()
    return dot_product / magnitude_product if magnitude_product != 0 else 0

# Perfiles de usuario (acción, romance)
user_profile = Vector2D(0.9, 0.2)  # Le gusta mucho acción, poco romance

# Catálogo de películas
movies = {
    "Mad Max": Vector2D(0.95, 0.1),      # Mucha acción, poco romance
    "The Notebook": Vector2D(0.1, 0.9),  # Poca acción, mucho romance
    "Mr. & Mrs. Smith": Vector2D(0.7, 0.6),  # Acción y romance balanceados
    "John Wick": Vector2D(0.98, 0.05),   # Casi pura acción
    "Titanic": Vector2D(0.2, 0.95)       # Poco acción, mucho romance
}

# Calcular similaridades
similarities = {}
for movie_name, movie_vector in movies.items():
    similarity = cosine_similarity(user_profile, movie_vector)
    similarities[movie_name] = similarity

# Ordenar por similaridad
recommended_movies = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

print(f"Perfil de usuario: Acción={user_profile.x:.1f}, Romance={user_profile.y:.1f}")
print("\nRecomendaciones ordenadas por similaridad:")
for movie, similarity in recommended_movies:
    print(f"{movie}: {similarity:.3f}")
```
---

## 🔗 Referencias y Lectura Adicional

### 📖 **Libros Recomendados**
1. **"Linear Algebra and Its Applications"** - Gilbert Strang
   - Capítulos 1-3: Vectores y espacios vectoriales
2. **"Introduction to Linear Algebra"** - Gilbert Strang
   - Enfoque más aplicado y visual
3. **"Mathematics for Machine Learning"** - Deisenroth, Faisal, Ong
   - Capítulo 2: Álgebra lineal aplicada a ML

### 🌐 **Recursos Online**
- **Khan Academy**: [Linear Algebra Course](https://www.khanacademy.org/math/linear-algebra)
- **3Blue1Brown**: [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- **MIT OpenCourseWare**: [18.06 Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)

### 🔧 **Bibliotecas Python Útiles**
- **NumPy**: Para operaciones vectoriales eficientes
- **SciPy**: Álgebra lineal avanzada
- **Matplotlib**: Visualización de vectores
- **Plotly**: Visualizaciones interactivas

### 📊 **Herramientas de Visualización**
- **GeoGebra**: Para visualizaciones geométricas interactivas
- **Desmos**: Calculadora gráfica online
- **Wolfram Alpha**: Para cálculos simbólicos
