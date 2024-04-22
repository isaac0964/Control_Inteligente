# Clasificacion con red neuronal multicapa
# Angel Isaac Gomez Canales 15/03/2024

# Importar librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import MLP_Kalman as MLP
from sklearn.datasets import make_circles

# Crear dataset
X, Y = make_circles(500, noise=0.1, factor=0.5)
X = X.T  # Poner X en el formato correcto ((n_dim, n_muestras))

# Graficar los datos
plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.bwr, edgecolors='k')

# Crear red y entrenarla 
net = MLP.MLP((2, 32, 1), activacion_oculta=MLP.relu, activacion_salida=MLP.logistica, eta=0.1)
net.fit(X, Y.reshape((1, -1)), epochs=100, batch_size=32)

# Obtener minimos y maximos de los datos para graficar area ade clasificacion
xmin, ymin = np.min(X, axis=1) - 0.5
xmax, ymax = np.max(X, axis=1) + 0.5

# Crear malla de puntos
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
data = [xx.ravel(), yy.ravel()]

#Predecir para cada punto de la malla
zz = (net.predict(data)).reshape(xx.shape)
# Pintar area de clasificacion
plt.contourf(xx, yy, zz, alpha=0.4, cmap=plt.cm.bwr)
plt.show()