import numpy as np 
import scipy
import pandas as pd 
from matplotlib import pyplot as plt 
import matplotlib
from scipy.io import wavfile
import scipy.signal as signal
import matplotlib.animation as animation

def main():
    # read wav file and get sampling rate and data
    fs, data = wavfile.read('data/rock-120bpm.wav')
    data = data[:, 0]
    plt.plot(data)
    plt.show()
    # get spectrum (frequency interval, time interval and matrix of frequency per time )
    frequency_interval, time_interval, spectrum = signal.spectrogram(data, fs = fs)
    print(frequency_interval.shape, time_interval.shape, spectrum.shape)
    # create figure
    fig, ax = plt.subplots()

    line_plot, = ax.plot(frequency_interval, spectrum[:, 0])

    # function to initialize frame
    def init():
        line_plot.set_ydata( spectrum[:, 0] )
        return line_plot,
    
    def animate(i):
        # change the data for the spectrum at time i, and change title to time i
        line_plot.set_ydata( spectrum[:, i] )
        return line_plot,
    # create animation
    ani = animation.FuncAnimation( fig, animate, init_func=init, interval=2, blit=True, save_count=50, repeat_delay = 1000)
    #ani.save('movie.mp4')
    plt.show()


if __name__ == "__main__":
    main()