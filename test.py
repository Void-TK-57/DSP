import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from signal_lib import *

import scipy.signal as sgn

"""
signal = unit_step(-5, -10, 11) - unit_step(5, -10, 11)

signal = sinusoid(a = 2, o = 0, w = np.pi*0.05, n1 = 0, n2 = 500)

signal.plot()

signal.spectrum()
"""

print(n)

x = np.exp(-1000.0*np.abs(n))

sig = Signal(x, n)

plt.plot(x)

plt.show()

sig.plot()

dft = sig.fft()

dft.plot()

dft = sig.fft(fold = True)

dft.plot()
