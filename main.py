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


if __name__ == "__main__":
    main()