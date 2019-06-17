import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('razao_de_linha1.txt')


for t in data:
    x = t[0]
    y = t[1]
    plt.scatter(y,x)

x1 = np.linspace(-2, 0.25)

plt.plot(x1 , 0.61/(x1 - 0.02 - 0.1833*2.5 )+ 1.2 + 0.03*2.5)
plt.xlim(-2, 1)
plt.ylim(-1, 1.5)
plt.xlabel('LOG([NII]/Ha)')
plt.ylabel('LOG([OIII]/Hb)')
plt.savefig('BPT.png')
plt.show()
