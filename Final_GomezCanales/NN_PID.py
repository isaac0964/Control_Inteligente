#! /usr/bin/env python
# coding=utf-8

import numpy as np

def tanh(z, beta=1, alpha=0.05, derivative=False):
    a = beta * np.tanh(alpha * z)
    if derivative:
        da = beta * alpha * ((1 - np.tanh(alpha * z)) * (1 + np.tanh(alpha * z)))
        return a, da
    return a

# Crar clase para red multicapa
class NN_PID():
    def __init__(self, n_in=2, alpha=0.05, beta=1, eta=0.01):
        # alpha y beta son para la tanh
        # Asumimos que hay tres entradas (e, i_e, d_e) y una salida
        self.n_in = n_in  # Neuronas de entrada
        self.eta = eta
        self.alpha = alpha
        self.beta = beta

        # Inicializar pesos aleatoriamente entre -1 y 1
        self.w = 2* np.random.rand(1, n_in) - 1
        
        # Numero de pesos
        self.nW = n_in 

    # Metodo para hacer predicciones
    def predict(self, X):
        z = np.dot(self.w, X)
        y = tanh(z, beta=self.beta, alpha=self.alpha)
        return y
    
    # Metodo para entrenar la red
    def fit(self, X, e, p=0.5, q=0, r=0.1):
        # Crear Matrices P, Q y R
        P = p * np.eye(self.nW)  # Tamaño de Npesos x Npesos (Li)
        Q = q * np.eye(self.nW)  # Tamaño de Npesos x Npesos (Li)
        R = r * np.eye(1)  # Tamaño de Nsalidas x Nsalias (m)

        # Obtener las derivadas de la funcion de activacion y la salida
        y, dphi = tanh((np.dot(self.w, X)), beta=self.beta, alpha=self.alpha, derivative=True)

        # Entrenamiento en linea

        # Obtener matriz H (derivada de la salida respecto a los pesos)
        H = (dphi * X).T

        # Ganancia de Kalman
        K = P.dot(H.T).dot(np.linalg.pinv(R + H.dot(P).dot(H.T)))  # Usar pseudoinversa para evitar errores


        # Calcular delta para actualizar pesos
        dw = self.eta * K.dot(e)

        # Actualizar los pesos
        self.w = self.w + dw.reshape(self.w.shape)

        # Actualizar matriz P
        P = P - np.dot(K, H.dot(P))
        if Q.any():  # Agregar Q solo si es diferente de 0
            P = P + Q
