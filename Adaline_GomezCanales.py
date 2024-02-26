# Angel Isaac Gomez Canales
# Adaline 29/02/2024

# Importar librerias requeridas 
import numpy as np
import matplotlib.pyplot as plt

def standardScaler(X):
    """
    Funcion para escalar los datos
    input: X: data
    output: datos escalados, media y desv estandar
    """
    mean = np.mean(X, axis=-1)
    std = np.std(X, axis=-1)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


# Crear clase para Adaline

class Adaline:
    def __init__(self, n_input, n_output):
        # Inicializar pesos y bias aleatoriamente [-1, 1]
        self.w = np.random.rand(n_output, n_input)
        self.b = np.random.rand(n_output, 1)

    def plot_recta(self, xrecta):
        """
        Esta funcion grafica la recta de la regresion de la red
        input: xrecta: lista con puntos de donde a donde se grafiacara la recta (en x)
        output: recta
        """
        w1, b = self.w[0,0], self.b[0]
        plot_l = plt.plot(xrecta, [(w1 * xrecta[0] + b), (w1 * xrecta[1] + b)], 'r')
        return plot_l
    
    def predict(self, X):
        return np.dot(self.w, X) + self.b
    
    def fit(self, X, Y, batch_size, epochs, eta=0.1):
        # Error inicial
        rmse_ant = np.sqrt(np.sum((Y - self.predict(X)) ** 2) / X.shape[1])
        # Pesos  y bias iniciales
        w_ant, b_ant = self.w, self.b
        # Valores minimo y maximo de x para graficar la recta
        minx = np.min(X, axis=1)
        maxx = np.max(X, axis=1)
        # Graficar recta inicial
        plt.subplot(1, 3, 1)
        plot_l, = self.plot_recta([minx, maxx])
        for epoch in range(epochs):
            # Suma del error cuadrado
            sum_e2 = 0
            # Validar el batch size
            if batch_size < 1 or batch_size > X.shape[1]:
                print(f"Elija un tamano de batch valido, debe ser en el rango [1, {X.shape[1]}]")
                break
            for i in range(int(np.ceil(X.shape[1] / batch_size))):
                # Tomar el lote del conjunto de datos
                # Esto no genera error ya que numpy solo toma los elementos
                # dentro de los limites
                X_batch = X[:, i * batch_size: (i+1) * batch_size]
                Y_batch = Y[:, i * batch_size: (i+1) * batch_size]

                # Actualizar pesos y bias
                Y_pred = self.predict(X_batch)  # Obtener prediccion sobre el batch
                error = Y_batch - Y_pred  # Calcular el error
                e2 = error ** 2 # Error cuadrado
                sum_e2 += np.sum(e2)
                # Actualizacion de pesos y bias
                self.w += eta / batch_size * (error @ X_batch.T)
                self.b += eta / batch_size * np.sum(error, axis=1)
            
            # Graficar recta
            plt.subplot(1, 3, 1)
            plot_l.remove()
            plot_l, = self.plot_recta([minx, maxx])
            #plt.pause(0.001)
            
            # Calcular RMSE de la epoca para graficarlo
            plt.subplot(1, 3, 2)
            rmse = np.sqrt(sum_e2 / X.shape[1])
            plt.plot([epoch, epoch+1], [rmse_ant, rmse], 'r')
            plt.axis([0, epochs, -2, 2])
            #plt.pause(0.001)
            rmse_ant = rmse

            # Graficar pesos
            plt.subplot(1, 3 , 3)
            plt.plot([epoch, epoch+1], [w_ant[0, 0], self.w[0,0]], 'g', label="w")
            plt.plot([epoch, epoch+1], [b_ant[0, 0], self.b[0,0]], 'b', label="bias")
            if epoch == 0:
                plt.legend()
            plt.axis([0, epochs, -2, 2])
            plt.pause(0.01)
            w_ant, b_ant = self.w, self.b

# Datos (problema de regresion)
X = np.array([16.68, 11.50, 12.03, 14.88, 13.75, 18.11, 8, 17.83, 79.24, 21.50, 40.33, 21, 13.50, 
              17.75, 24, 29, 15.35, 19, 9.50, 35.10, 17.90, 52.32, 18.75, 19.83, 10.75]).reshape(1, -1)  # Reshape para darle forma adecuada (m, p)
Y = np.array([7, 3, 3, 4, 6, 7, 2, 7, 30, 5, 16, 10, 4, 6, 9, 10, 6, 7, 3, 17, 10, 26, 9, 8, 4]).reshape(1, -1)  # Reshape para darle forma adecuada (n, p)

# Escalar los datos
X_scaled, _, _ = standardScaler(X)
Y_scaled, _, _ = standardScaler(Y)

# Mezclar los datos
idx = np.random.permutation(X_scaled.shape[1])
X_scaled = X_scaled[:, idx]
Y_scaled = Y_scaled[:, idx]


# Graficar los datos
plt.figure(figsize=(14, 5)) # w x h
plt.subplot(1, 3, 1)
plt.title("Grafica de los datos y regresion")
plt.plot(X_scaled[0, :], Y_scaled[0, :], 'k.')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.subplot(1, 3, 2)
plt.title("RMSE por epoca")
plt.xlabel("Epoca")
plt.ylabel("RMSE")

plt.subplot(1, 3, 3)
plt.title("Actualizacion peso y bias por epoca")
plt.xlabel("Epoca")

net = Adaline(1, 1)
net.fit(X_scaled, Y_scaled, 25, 250, eta=0.01)
plt.show()

# Graficar despues de cada epoca o iteracion?
# Estan bien las graficas
# Cualquier criterio de paro?