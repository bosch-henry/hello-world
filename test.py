import numpy as np
import matplotlib.pyplot as plt

# 以自然数序列作为多项式的系数
#func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
func = np.poly1d(np.array([1, 0, 0]).astype(float))

# x 的横坐标
x = np.linspace(-10, 10, 30)
# 得到y的对应值
y = func(x)
#绘图
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y(x)')
# 显示函数图像
plt.show()
