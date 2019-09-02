import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from signal_lib import *

import scipy.signal as sgn

signal1 = complex_exp(a = .05, w = np.pi*0.121, n1 = -75, n2 = 75)
signal2 = complex_exp(a = .05, w = np.pi*0.1, n1 = -75, n2 = 75)

signal = signal1 + signal2

signal.plot()

signal = complex_exp(a = 1, w = np.pi*0.1, n1 = -50, n2 = 50)

signal.plot()