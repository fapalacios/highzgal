import numpy as np
import matplotlib.pyplot as plt

#J2221
plt.scatter(-1.1929410082494034, 0.42539790997307675, color='black')
plt.arrow(-1.1929410082494034, 0.42539790997307675, 0, 0.05, width=0.02, color='black')
plt.arrow(-1.1929410082494034, 0.42539790997307675, -0.05, 0, width=0.02, color='black')
#J021247
plt.scatter(0.0, 0.8064502355915049, color='black')
plt.arrow(0.0, 0.8064502355915049, 0, 0.05, width=0.02, color='black')
plt.arrow(0.0, 0.8064502355915049, -0.05, 0, width=0.02, color='black')
#J023251
plt.scatter(-1.2335265181181607, 0.7864913406584415, color='black')
#J084959
plt.scatter(-0.5549430492841291, 0.8488002595502193, color='black')
plt.arrow(-0.5549430492841291, 0.8488002595502193, 0, 0.05, width=0.02, color='black')
plt.arrow(-0.5549430492841291, 0.8488002595502193, -0.05, 0, width=0.02, color='black')
#J090407
plt.scatter(-0.758054161352852, 0.7690785646287618, color='black')
#J095921
plt.scatter(-2, -0.17521629375982367, color='black')
plt.arrow(-2, -0.17521629375982367, 0, 0.05, width=0.02, color='black')
#J221326
plt.scatter(-2, 1.4908240358852602, color='black')
plt.arrow(-2, 1.4908240358852602, 0, 0.05, width=0.02, color='black')

x1 = np.linspace(-2, 0.25)

plt.plot(x1 , 0.61/(x1 - 0.02 - 0.1833*2.5 )+ 1.2 + 0.03*2.5)
plt.xlim(-2.01, 1)
plt.ylim(-1, 1.8)
plt.title('z ~ 2.5')
plt.xlabel('LOG([NII]/Ha)')
plt.ylabel('LOG([OIII]/Hb)')
plt.savefig('BPT.png')
plt.show()
