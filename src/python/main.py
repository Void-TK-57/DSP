import numpy as np 
import scipy
import pandas as pd 
from matplotlib import pyplot as plt 
import wavio

from datetime import datetime, time, date, timedelta, MINYEAR
from signal_lib import *


import sys

def main_():
    # read wav file and get sampling rate and data
    fs, data = wavfile.read('data/piano_c5.wav')
    duration = len(data)/fs
    print(duration)
    # number of frames
    frames = 30*duration
    print(frames)
    # reshape the data per frame
    data_per_frame = np.array_split(data, frames)
    # return fft
    def generate(data, fs):
        # fft
        fft = np.absolute( np.fft.rfft(data,  n = len(data)))
        # get frequencies
        freq = np.fft.fftfreq(len(fft), d = 1.0/fs)
        # return fft and freq
        return fft, freq

    fft, freq = generate(data_per_frame[0], fs)
    print(fft)
    print(freq)
    print(fft.shape, freq.shape)
    plt.plot(freq, fft)
    plt.show()

    # create figure
    fig, ax = plt.subplots()
    line_plot, = ax.plot(freq, fft)
    ax.set_xlabel("Frequencies")
    ax.set_ylabel("Magnitude")

    # function to initialize frame
    def init():
        data, _ = generate(data_per_frame[0], fs)
        line_plot.set_ydata( data )
        return line_plot,
    
    def animate(i):
        # change the data for the spectrum at time i, and change title to time i
        data, _ = generate( data_per_frame[ i % len(data_per_frame) ], fs )
        line_plot.set_ydata( data )
        return line_plot,
    # create animation
    ani = animation.FuncAnimation( fig, animate, init_func=init, interval=100/6, blit=True, save_count=50, repeat_delay = 1000)
    #ani.save('movie.mp4')
    plt.show()

# create datafrme from wavio object
def to_dataframe(wavio_object, microsecond=0):
    # get data
    data = wavio_object.data.copy()
    # get index
    # base start
    base = datetime.today()
    base = base.replace(hour = 0, second=0, minute=0, microsecond=microsecond)
    # increase timedelta
    index = [base, ]
    # for each value
    for i in range(1, wavio_object.data.shape[0]):
        # add to index the last element + timedelta
        index.append( index[-1] + timedelta(milliseconds=1000.0/wavio_object.rate) )
    # create dataframe
    dataframe = pd.DataFrame(data, index=index, columns=range(data.shape[1]))
    return dataframe
    

def main(path = "../../data/8bit-C4.wav"):
    wav_control = wavio.read(path)
    print(wav_control)
    # creat signal
    signal = Signal( to_dataframe(wav_control) )
    #signal = sinusoid(w = 2*np.pi, n1 = 0, n2 = 500.0)
    signal = square(0, 10, -5, 50)
    signal.stem()
    print(signal)
    signal_2 = real_exp(0.9, 5, 50)
    signal_2.stem()
    print(signal_2)
    signal.convolution(signal_2).stem()
    

if __name__ == "__main__":
    # get args
    args = sys.argv
    # get path
    file = "8bit-C4.wav" if len(args) == 1 else args[1]
    main("../../data/" + file)