# Aproximacion de funciones con red neuronal multicapa
# Angel Isaac Gomez Canales 15/03/2024

# Importar librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import MLP
from mayavi import mlab

# Definir funcion a aproximar
def f(X, s=1):
    sum2 = X[0,:]**2 + X[1, :]**2
    Z = (10 / (np.pi * s**4)) * (1 - (1/2 * ((sum2) / s**2))) * np.exp(-(sum2) / 2 * s**2)
    return Z

# Crear dataset mediante una malla y el valor de la funcion en cada putno
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
X = np.array([xx.ravel(), yy.ravel()])
Y = f(X)

# Graficar la superficie
fig = plt.figure(figsize=(12, 6))
fig.suptitle("Aproximacion de Funcion")
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(xx, yy, Y.reshape(xx.shape), cmap=plt.cm.plasma, linewidth=0.2, edgecolor='k', alpha=0.7)
ax.set_title("Funcion Original")

# Crear red y entrenarla 
net = MLP.MLP((2, 32, 16, 8, 1), activacion_oculta=MLP.tanh, activacion_salida=MLP.lineal, eta=0.5)
net.fit(X, Y.reshape((1, -1)), epochs=500, batch_size=32)

#Predecir con la red para cada punto de la malla
zz = (net.predict(X)).reshape(xx.shape)
# Mostrar funcion predicha
ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(xx, yy, zz, cmap=plt.cm.plasma, linewidth=0.2, edgecolor='k', alpha=0.7)
ax.set_title("Funcion predicha")

# Graficar el error
ax = fig.add_subplot(133, projection='3d')
ax.plot_surface(xx, yy, (Y.reshape(xx.shape) - zz), cmap=plt.cm.plasma, linewidth=0, antialiased=False, alpha=0.7)
ax.set_title(r"Error ($deseada - predicha$)")

# Red vs Original
fig = mlab.figure()
ax_ranges = [-5, 5, -5, 5, 0, np.max(zz)+0.5]
ax_scale = [1.0, 1.0, 1.0]
ax_extent = ax_ranges * np.repeat(ax_scale, 2)
l = np.linspace(-5, 5, 50)  # Punros para graficar en x y en y
# Graficar
surf1 = mlab.surf(l, l, Y.reshape(xx.shape), color=(186/255,225/255,255/255))  # Azul
surf2 = mlab.surf(l, l, zz, color=(255/255,179/255,186/255))
surf1.actor.actor.scale = ax_scale
surf2.actor.actor.scale = ax_scale
mlab.points3d(3, 0, 3, color=(186/255,225/255,255/255), scale_factor=0.7, mode='sphere', resolution=8, mask_points=8, name='Original')
mlab.text3d(3.1, 0, 3, 'Original', scale=(0.7, 0.7, 0.7))
mlab.points3d(-3, 0, 3, color=(255/255,179/255,186/255), scale_factor=0.7, mode='sphere', resolution=8, mask_points=8, name='Red')
mlab.text3d(-2.9, 0.1, 3, 'Red', scale=(0.7, 0.7, 0.7))
mlab.view()
# Ajustar vista 
mlab.outline(surf1, color=(.7, .7, .7), extent=ax_extent)
mlab.axes(surf1, color=(.7, .7, .7), extent=ax_extent,
            ranges=ax_ranges,
            xlabel='x', ylabel='y', zlabel='z')
plt.show()
