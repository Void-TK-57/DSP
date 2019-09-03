import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from signal_lib import *

import scipy.signal as sgn

def plot_analysis(x):
    # create subplot
    plt.subplot(2, 2, 1)
    # plot real values
    plt.plot(np.real(x))
    # create title
    plt.title("Real Values")
    # create subplot        
    plt.subplot(2, 2, 2)
    # plot imaginary values
    plt.plot(np.imag(x))
    # get title
    plt.title("Imaginary Values")
    # create subplot
    plt.subplot(2, 2, 3)
    # plot absolute values
    plt.plot(np.abs(x))
    # get title
    plt.title("Absolute")
    # create subplot
    plt.subplot(2, 2, 4)
    # plot angle
    plt.plot(np.angle(x))
    # get title
    plt.title("Angle")

    #show plot
    plt.show()

"""
signal1 = complex_exp(a = .05, w = np.pi*0.121, n1 = -75, n2 = 75)
signal2 = complex_exp(a = .05, w = np.pi*0.1, n1 = -75, n2 = 75)

signal = signal1 + signal2

signal.plot()

signal = complex_exp(a = 1, w = np.pi*0.1, n1 = -50, n2 = 50)

signal.plot()
"""

n = np.arange(-1, 4)

x = np.arange(1, 6)

k = 500

sig1 = complex_exp(a = 0.9, o = 0, w = np.pi/3.0, n1 = 0,  n2 = 11)

sig1 = real_exp(a = -0.9, n1 = -5, n2 = 6)

sig1 = sinusoid(a = 1, o = 0, w = np.pi/8, n1 = -8, n2 = 9)

sig1.plot()

sig2 = sig1.fft(n = 400, fold = False)

sig2.plot()

sig2.plot_analysis()

# =================================================

t = np.linspace(0, 0.5, 500)
n = np.arange(0, 501)
r = sinusoid(a = 1, o = 0, w = 0.08*np.pi, n1 = 0, n2 = 501) + sinusoid(a = 0.5, o = 0, w = 0.18*np.pi, n1 = 0, n2 = 501)
s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)

fft = np.fft.fft(s)
T = t[1] - t[0]  # sampling interval 
N = s.size

# 1/T = frequency
f = np.linspace(0, 1 / T, N)

plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, width=1.5)  # 1 / N is a normalization factor
plt.show()