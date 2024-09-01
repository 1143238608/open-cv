import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
print(x)
y = np.sin(x)
print(y)

plt.figure()
plt.plot(x, y)
plt.yticks(ticks=[-1, -0.8, -0.5, -0.1, 1], labels=['a', 'b', 'c', 'd', 'e'])
ax = plt.gca()
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('None')
ax.spines['left'].set_color('None')
ax.spines['bottom'].set_color('None')

ax.xaxis.set_ticks_position('top')
ax.yaxis.set_ticks_position('right')
plt.show()
