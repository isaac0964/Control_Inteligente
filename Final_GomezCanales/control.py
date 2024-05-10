#! /usr/bin/env python
# coding=utf-8

# Importar librerias requeridas
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg  import Odometry
import numpy as np
from NN_PID import NN_PID

# Definir la posicion deseada en x y y
pos_deseada = np.array([5, 3])
D = 0.08

# Errores anteriores en x y y
eAx = 0
eAy = 0

# Instanciar los controladores neuronales para x y y 
PDx = NN_PID()
PDy = NN_PID()

bandera_control = 1

# Definir el callback para realizar el control
def Callback(msg, args):
	# Se le pasan como argumentos el arreglo con la posicion deseada y la d

	global eAx
	global eAy
	global bandera_control

	# Solo hacer el control si no se estan procesando ya los datos

	if bandera_control:
		bandera_control = 0  # Hacer la bandera 0 para no procesar mas datos
		# Obtener la posicion en x y y
		x = msg.pose.pose.position.x
		y = msg.pose.pose.position.y

		# Obtener el quaternion de la rotacion para despues convertirlo a angulo
		q0 = msg.pose.pose.orientation.w
		q1 = msg.pose.pose.orientation.x
		q2 = msg.pose.pose.orientation.y
		q3 = msg.pose.pose.orientation.z
		theta = np.arctan2(2*(q0*q3 + q1*q2), (1 - 2*(q2**2 + q3**2)))

		# Obtener la posicion deseada en x y y
		xd = pos_deseada[0]
		yd = pos_deseada[1]

		# Calcular las proyecciones de x y y al frente del robot
		xp = x + D * np.cos(theta)
		yp = y + D * np.sin(theta)

		# Calcular los errores en x y en y
		ex = xd - xp
		ey = yd - yp

		# Calcular la derivada del error en x y y
		d_ex = ex - eAx
		eAx = ex  # Actualizar el error 
		errorx = np.array([[ex], [d_ex]])

		d_ey = ey - eAy
		eAy = ey  # Actualizar el error anterior
		errory = np.array([[ey], [d_ey]])

		# Obtener el control de la red neuronal
		ux = PDx.predict(errorx)[0]
		uy = PDy.predict(errory)[0]

		# Actualizar pesos de la red
		PDx.fit(errorx, ex, p=10, q=1, r=0.1)
		PDy.fit(errory, ey, p=10, q=1, r=0.1)

		# matriz u
		u = np.array([ux, uy])

		# Matriz T del modelo
		T = np.array([[np.cos(theta), -D * np.sin(theta)], [np.sin(theta), D * np.cos(theta)]])

		# Obtener arreglo con la velocidad lineal y angular
		vw = np.linalg.pinv(T).dot(u)

		print(vw)

		# Generar el mensaje con la velocidad lineal y angular
		msg_out = Twist()
		msg_out.linear.x = vw[0]
		msg_out.linear.y = 0
		msg_out.linear.z = 0
		msg_out.angular.x = 0
		msg_out.angular.y = 0
		msg_out.angular.z = vw[1]


		# Parar si el error está en un rango deseado
		if ex < 0.01 and ey < 0.01:
			rospy.signal_shutdown("Llegue al punto deseado! :)")

		# Publicar el mensaje
		pub.publish(msg_out)
		print("x=", x, "y=", y)

		# Al terminar de procesar los datos, hacer la bandera true para volver a procesar datos
		bandera_control = 1




rospy.init_node("topic_publisher", anonymous=True)

# Generar suscriptor y publicador
pub =  rospy.Publisher('/tb3_0/cmd_vel',Twist, queue_size=1)
rospy.Subscriber("/tb3_0/odom", Odometry, Callback, (pos_deseada, D))
rospy.spin()
