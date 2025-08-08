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

### 10.1. Clase Vector Básica

```python
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, atan2, degrees

class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def magnitude(self):
        return sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x/mag, self.y/mag)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def angle(self):
        return degrees(atan2(self.y, self.x))
    
    def distance_to(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def project_onto(self, other):
        dot_product = self.dot(other)
        other_mag_squared = other.x**2 + other.y**2
        if other_mag_squared == 0:
            return Vector2D(0, 0)
        scalar = dot_product / other_mag_squared
        return other * scalar
    
    def __repr__(self):
        return f"Vector2D({self.x:.2f}, {self.y:.2f})"
```

### 10.2. Visualización Avanzada

```python
def plot_vectors(*vectors, labels=None, colors=None, title="Operaciones Vectoriales"):
    """
    Visualiza múltiples vectores con etiquetas y colores personalizados
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    if labels is None:
        labels = [f'Vector {i+1}' for i in range(len(vectors))]
    
    origin = np.array([0, 0])
    
    for i, vector in enumerate(vectors):
        if hasattr(vector, 'x'):  # Si es nuestra clase Vector2D
            x, y = vector.x, vector.y
        else:  # Si es un array de NumPy
            x, y = vector[0], vector[1]
            
        ax.quiver(*origin, x, y, 
                 angles='xy', scale_units='xy', scale=1, 
                 color=colors[i % len(colors)], 
                 label=f'{labels[i]}: ({x:.2f}, {y:.2f})',
                 width=0.005, headwidth=3)
    
    # Configuración del gráfico
    max_val = max([max(abs(v.x), abs(v.y)) if hasattr(v, 'x') 
                   else max(abs(v[0]), abs(v[1])) for v in vectors])
    limit = max_val * 1.2
    
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
v1 = Vector2D(3, 4)
v2 = Vector2D(-2, 3)
suma = v1 + v2
diferencia = v1 - v2
escalado = v1 * 0.5

plot_vectors(v1, v2, suma, diferencia, escalado,
            labels=['Vector A', 'Vector B', 'A + B', 'A - B', '0.5 × A'],
            title='Operaciones Básicas con Vectores')
```

### 10.3. Ejemplos Prácticos

```python
# Ejemplo 1: Operaciones básicas
print("=== OPERACIONES BÁSICAS ===")
v1 = Vector2D(3, 4)
v2 = Vector2D(-1, 2)

print(f"Vector 1: {v1}")
print(f"Vector 2: {v2}")
print(f"Suma: {v1 + v2}")
print(f"Resta: {v1 - v2}")
print(f"Escalar × 2: {v1 * 2}")
print(f"Magnitud v1: {v1.magnitude():.2f}")
print(f"Ángulo v1: {v1.angle():.2f}°")

# Ejemplo 2: Producto punto y proyección
print("\n=== PRODUCTO PUNTO Y PROYECCIÓN ===")
dot_product = v1.dot(v2)
print(f"Producto punto: {dot_product}")

if dot_product == 0:
    print("Los vectores son perpendiculares")
elif dot_product > 0:
    print("El ángulo entre vectores es agudo")
else:
    print("El ángulo entre vectores es obtuso")

proyeccion = v1.project_onto(v2)
print(f"Proyección de v1 sobre v2: {proyeccion}")

# Ejemplo 3: Vector unitario
print("\n=== VECTORES UNITARIOS ===")
v1_unit = v1.normalize()
print(f"Vector unitario de v1: {v1_unit}")
print(f"Magnitud del vector unitario: {v1_unit.magnitude():.6f}")
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

## 🔧 Herramientas de Análisis Vectorial

### Calculadora de Vectores Interactiva

```python
class VectorCalculator:
    """Calculadora interactiva para operaciones vectoriales"""
    
    @staticmethod
    def menu():
        print("\n=== CALCULADORA DE VECTORES ===")
        print("1. Suma de vectores")
        print("2. Resta de vectores")
        print("3. Producto punto")
        print("4. Ángulo entre vectores")
        print("5. Proyección vectorial")
        print("6. Normalizar vector")
        print("7. Distancia entre puntos")
        print("8. Visualizar vectores")
        print("0. Salir")
    
    @staticmethod
    def get_vector_input(name):
        x = float(input(f"Ingrese componente x de {name}: "))
        y = float(input(f"Ingrese componente y de {name}: "))
        return Vector2D(x, y)
    
    def run(self):
        while True:
            self.menu()
            choice = input("\nSeleccione una opción: ")
            
            if choice == '0':
                print("¡Hasta luego!")
                break
            elif choice == '1':
                v1 = self.get_vector_input("vector 1")
                v2 = self.get_vector_input("vector 2")
                result = v1 + v2
                print(f"Resultado: {v1} + {v2} = {result}")
            elif choice == '2':
                v1 = self.get_vector_input("vector 1")
                v2 = self.get_vector_input("vector 2")
                result = v1 - v2
                print(f"Resultado: {v1} - {v2} = {result}")
            elif choice == '3':
                v1 = self.get_vector_input("vector 1")
                v2 = self.get_vector_input("vector 2")
                result = v1.dot(v2)
                print(f"Producto punto: {result}")
            elif choice == '4':
                v1 = self.get_vector_input("vector 1")
                v2 = self.get_vector_input("vector 2")
                dot_product = v1.dot(v2)
                magnitude_product = v1.magnitude() * v2.magnitude()
                if magnitude_product != 0:
                    cos_angle = dot_product / magnitude_product
                    angle_rad = math.acos(min(1, max(-1, cos_angle)))
                    angle_deg = math.degrees(angle_rad)
                    print(f"Ángulo entre vectores: {angle_deg:.2f}°")
                else:
                    print("No se puede calcular el ángulo (vector cero)")
            elif choice == '5':
                v1 = self.get_vector_input("vector a proyectar")
                v2 = self.get_vector_input("vector base")
                proj = v1.project_onto(v2)
                print(f"Proyección de {v1} sobre {v2} = {proj}")
            elif choice == '6':
                v = self.get_vector_input("vector")
                normalized = v.normalize()
                print(f"Vector normalizado: {normalized}")
                print(f"Magnitud: {normalized.magnitude():.6f}")
            elif choice == '7':
                p1 = self.get_vector_input("punto 1")
                p2 = self.get_vector_input("punto 2")
                distance = p1.distance_to(p2)
                print(f"Distancia entre {p1} y {p2}: {distance:.2f}")
            elif choice == '8':
                vectors = []
                labels = []
                n = int(input("¿Cuántos vectores desea visualizar? "))
                for i in range(n):
                    v = self.get_vector_input(f"vector {i+1}")
                    vectors.append(v)
                    labels.append(f"Vector {i+1}")
                plot_vectors(*vectors, labels=labels)
            else:
                print("Opción no válida")

# Para usar la calculadora:
# calculator = VectorCalculator()
# calculator.run()
```

---

## 📚 Conceptos Avanzados

### Transformaciones Lineales con Vectores

```python
class LinearTransformation:
    """Clase para realizar transformaciones lineales en vectores 2D"""
    
    def __init__(self, matrix):
        """
        matrix: lista de listas representando una matriz 2x2
        [[a, b], [c, d]]
        """
        self.matrix = matrix
    
    def transform(self, vector):
        """Aplica la transformación lineal al vector"""
        a, b = self.matrix[0]
        c, d = self.matrix[1]
        
        new_x = a * vector.x + b * vector.y
        new_y = c * vector.x + d * vector.y
        
        return Vector2D(new_x, new_y)
    
    def transform_multiple(self, vectors):
        """Transforma múltiples vectores"""
        return [self.transform(v) for v in vectors]

# Transformaciones comunes:

# Rotación de 90° en sentido antihorario
rotation_90 = LinearTransformation([[0, -1], [1, 0]])

# Reflexión sobre el eje x
reflection_x = LinearTransformation([[1, 0], [0, -1]])

# Escalamiento (2x en x, 0.5x en y)
scaling = LinearTransformation([[2, 0], [0, 0.5]])

# Ejemplo de uso
original = Vector2D(3, 4)
rotated = rotation_90.transform(original)
reflected = reflection_x.transform(original)
scaled = scaling.transform(original)

print(f"Original: {original}")
print(f"Rotado 90°: {rotated}")
print(f"Reflejado en X: {reflected}")
print(f"Escalado: {scaled}")
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
