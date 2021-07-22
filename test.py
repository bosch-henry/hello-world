import numpy as np
import math
from shapely import geometry


# def if_inPoly(polygon, Points):
#     line = geometry.LineString(polygon)
#     point = geometry.Point(Points)
#     polygon = geometry.Polygon(line)
#     return polygon.contains(point)
#
#
# square = [(0, 0), (1, 0), (1, 1), (0, 1)]  # 多边形坐标
# pt1 = (2, 2)  # 点坐标
# pt2 = (0.5, 0.5)
# print(if_inPoly(square, pt1))
# print(if_inPoly(square, pt2))
#
#
#
# Alfa = 0  * math.pi/180     #rote by x axle, roll angle
# Beta = -1  * math.pi/180     #rote by y axle, pich angle
# Gama = -0.4  * math.pi/180     #rote by z axle, yaw angle
#
# R_Alfa = np.array([[1, 0 ,0],[0, math.cos(Alfa), -math.sin(Alfa)], [0, math.sin(Alfa), math.cos(Alfa)]]).astype("float32")
# R_Beta = np.array([[math.cos(Beta), 0, math.sin(Beta)], [0, 1, 0], [-math.sin(Beta), 0, math.cos(Beta)]]).astype("float32")
# R_Gama = np.array([[math.cos(Gama), -math.sin(Gama), 0],[math.sin(Gama), math.cos(Gama), 0], [0, 0, 1]]).astype("float32")
#
# R = np.zeros((3, 4))
# T = np.array([1.354, 0, 1.452])
# R[:3, :3] = np.array([R_Alfa.dot(R_Beta).dot(R_Gama)])
# R[:, 3] = T.T
#
# # P = np.vstack((R, np.array([0, 0, 0, 1])))
# #
# data = np.load ("/media/henry/LINUX/GT_2021-07-19-18-20-30.npz")
#
# print(data['point_righty'][0])
# # data['point_raw_righty']
# #
# # print(P)
# #
# dx = np.arange(-12.5, 37, 5)
# print(np.linspace(-15, 40, 11))
# print(dx)