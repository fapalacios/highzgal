import numpy as np
import matplotlib.pyplot as plt

#J2221
plt.scatter(-1.1929410082494034, 0.42539790997307675)
plt.arrow(-1.1929410082494034, 0.42539790997307675, 0, 0.04, width=0.02, color='black')
#J021247
plt.scatter(0.0, 0.8064502355915049)
plt.arrow(0.0, 0.8064502355915049, 0, 0.04, width=0.02, color='black')
plt.arrow(0.0, 0.8064502355915049, -0.04, 0, width=0.02, color='black')
#J021411
plt.scatter(0.17609125905568124, 0.43884955614766885)
plt.arrow(0.17609125905568124, 0.43884955614766885, 0, 0.04, width=0.02, color='black')
#J023251
plt.scatter(-1.2335265181181607, 0.7864913406584415, color='black')
#J084847
plt.scatter(-1.4854992676388221, 0.2592976792052675, marker='<')
#J084909
plt.scatter(-1.3162888987138777, 0.6912753660039634)
#J084959
plt.scatter(-0.1413577414780052, 0.8488002595502193, marker='^')
#J090106
plt.scatter(0.8518925210140749, 0.38879163378436604, marker='^')
#J090407
plt.scatter(-0.758054161352852, 0.7690785646287618)
#J095921
plt.scatter(-2, -0.17521629375982367, marker='^')
#J221326
plt.scatter(-2, 1.4908240358852602, marker='^')

x1 = np.linspace(-2, 0.25)

plt.plot(x1 , 0.61/(x1 - 0.02 - 0.1833*2.5 )+ 1.2 + 0.03*2.5)
plt.xlim(-2.01, 1)
plt.ylim(-1, 1.6)
plt.xlabel('LOG([NII]/Ha)')
plt.ylabel('LOG([OIII]/Hb)')
plt.savefig('BPT.png')
plt.show()
