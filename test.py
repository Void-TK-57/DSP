import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from signal_lib import *

import scipy.signal as sgn

signal1 = sinusoid(a = .05, w = np.pi*0.121, n1 = -150, n2 = 150)
signal2 = sinusoid(a = .05, w = np.pi*0.1, n1 = -150, n2 = 150)

signal = signal1 + signal2

signal.plot()