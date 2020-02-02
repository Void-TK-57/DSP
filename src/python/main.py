import numpy as np 
import scipy
import pandas as pd 
from matplotlib import pyplot as plt 
import wavio

from datetime import datetime, time, date, timedelta, MINYEAR
from signal_lib import *

import math

import sys



# function to creat spectrum
def spectrogram(signal, fps = 60.0):
    signal_per_frames = slice_signal(signal, math.floor( signal.total_seconds*fps) )
    total_frames = len( signal_per_frames )
    # create subplots
    fig, ax = plt.subplots()
    frequency, power = [], []
    line, = plt.plot([], [], 'ro')

    def init():
        return line,

    def update(frame):
        # get signal
        signal = signal_per_frames[ frame % total_frames ]
        # get fft abs
        spectrum = signal.fft().abs()
        # get data itself
        data = spectrum.data
        # get frequency
        frequency = data.index
        # get power
        power = data.values
        # set data of line plot
        line.set_data(frequency, power)
        # return line
        return line,

    # create animation
    ani = FuncAnimation(fig, update, init_func=init, blit=True)

    # show plot
    plt.show()


# create datafrme from wavio object
def to_signal(wavio_object, microsecond=0):
    # get data
    data = wavio_object.data.copy()
    # get index
    # base start
    base = datetime.today()
    #base = base.replace(hour = 0, second=0, minute=0, microsecond=microsecond)
    # increase timedelta
    index = [base, ]
    # for each value
    for i in range(1, wavio_object.data.shape[0]):
        # add to index the last element + timedelta
        index.append( index[-1] + timedelta(seconds=1.0/wavio_object.rate) )
    return [ TimeSignal(pd.Series(data[:, i], index = index), wavio_object.rate) for i in range(data.shape[1]) ]
    
# function to slice the total signal into frames
def slice_signal(signal, total_samples):
    # sample size
    sample_size = len(signal.data) // total_samples
    # slice data
    return list( [ TimeSignal(signal.data[ i*sample_size : (i+1)*sample_size ], signal.sampling_rate) for i in range(total_samples) ] )
    


def main(path = "../../data/8bit-C4.wav"):
    wav_control = wavio.read(path)
    print(wav_control)
    # creat signal
    signals = to_signal(wav_control)
    signal = signals[0]
    signal.plot()
    signal.fft().abs().plot()
    
    # get spectrogram
    spectrogram(signal)


if __name__ == "__main__":
    # get args
    args = sys.argv
    # get path
    file = "8bit-C4.wav" if len(args) == 1 else args[1]
    main("../../data/" + file)