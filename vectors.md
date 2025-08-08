# Documentación Completa sobre Vectores

## 1. Definición de Vector

Un **vector** en 2D es una entidad matemática que tiene **magnitud** (longitud) y **dirección**.  
Se representa como un par ordenado:  
$$
\mathbf{v} = (x, y)
$$

Donde:  
- \(x\) es la componente en el eje \(X\)  
- \(y\) es la componente en el eje \(Y\)  

---

## 2. Operaciones Básicas con Vectores

### 2.1. Suma de Vectores

Dados dos vectores \(\mathbf{A} = (a_x, a_y)\) y \(\mathbf{B} = (b_x, b_y)\),  

$$
\mathbf{A} + \mathbf{B} = (a_x + b_x, \quad a_y + b_y)
$$

---

### 2.2. Multiplicación por Escalar

Dado un vector \(\mathbf{v} = (x, y)\) y un escalar \(k \in \mathbb{R}\),  

$$
k \times \mathbf{v} = (k \times x, \quad k \times y)
$$

**Efectos:**  
- Si \(k > 1\), el vector se alarga.  
- Si \(0 < k < 1\), el vector se acorta.  
- Si \(k = 0\), el vector se convierte en el vector cero \((0,0)\).  
- Si \(k < 0\), el vector cambia de dirección y su magnitud se escala por \(|k|\).

---

## 3. Propiedades Importantes

- **Conmutatividad de la suma:** \(\mathbf{A} + \mathbf{B} = \mathbf{B} + \mathbf{A}\)  
- **Asociatividad de la suma:** \((\mathbf{A} + \mathbf{B}) + \mathbf{C} = \mathbf{A} + (\mathbf{B} + \mathbf{C})\)  
- **Distributividad respecto a escalares:** \(k (\mathbf{A} + \mathbf{B}) = k\mathbf{A} + k\mathbf{B}\)

---

## 4. Magnitud (Norma) de un Vector

La magnitud o norma de un vector \(\mathbf{v} = (x, y)\) es:  

$$
\|\mathbf{v}\| = \sqrt{x^2 + y^2}
$$

---

## 5. Dirección Vectorial

El ángulo \(\theta\) que un vector \(\mathbf{v} = (x, y)\) forma con el eje \(X\) positivo se calcula con:  

$$
\theta = \arctan\left(\frac{y}{x}\right)
$$

Para evitar errores en cuadrantes, se recomienda usar la función:  

$$
\theta = \mathrm{atan2}(y, x)
$$

**Conversión a grados:**  

$$
\theta_{\text{grados}} = \theta_{\text{radianes}} \times \frac{180}{\pi}
$$

---

## 6. Producto Punto (Dot Product)

Dado \(\mathbf{A} = (a_x, a_y)\) y \(\mathbf{B} = (b_x, b_y)\):  

$$
\mathbf{A} \cdot \mathbf{B} = a_x b_x + a_y b_y
$$

**Interpretación geométrica:**  

$$
\mathbf{A} \cdot \mathbf{B} = \|\mathbf{A}\| \|\mathbf{B}\| \cos \theta
$$

Donde \(\theta\) es el ángulo entre \(\mathbf{A}\) y \(\mathbf{B}\).

**Propiedades:**  
- \(\mathbf{A} \cdot \mathbf{B} = \mathbf{B} \cdot \mathbf{A}\) (conmutativo)  
- Si \(\mathbf{A} \cdot \mathbf{B} = 0\), los vectores son ortogonales (perpendiculares).

---

## 7. Proyección Vectorial

La proyección de un vector \(\mathbf{A}\) sobre otro vector \(\mathbf{B}\) es:  

$$
\mathrm{proj}_{\mathbf{B}} \mathbf{A} = \left(\frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{B}\|^2}\right) \mathbf{B}
$$

Representa la componente de \(\mathbf{A}\) en la dirección de \(\mathbf{B}\).

---

## 8. Distancia entre dos vectores (puntos)

Dado \(\mathbf{P} = (p_x, p_y)\) y \(\mathbf{Q} = (q_x, q_y)\), la distancia es:  

$$
d(\mathbf{P}, \mathbf{Q}) = \sqrt{(p_x - q_x)^2 + (p_y - q_y)^2}
$$

---

## 9. Visualización en Python (Ejemplo)

```python
import numpy as np
import matplotlib.pyplot as plt

v1 = np.array([3, 4])
v2 = np.array([-4, 2])
suma = v1 + v2
origin = np.array([0, 0])

plt.quiver(*origin, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector v1')
plt.quiver(*origin, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector v2')
plt.quiver(*origin, suma[0], suma[1], angles='xy', scale_units='xy', scale=1, color='g', label='Suma v1 + v2')

plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.grid(True)
plt.legend()
plt.show()
