# signal lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import collections
from scipy.signal import lfilter
import seaborn as sns; sns.set()
from scipy.fftpack import fft

from datetime import datetime, time, date, timedelta



class Signal:

    def __init__(self, data = None):
        self.data = data.copy() if data is not None else np.arange(len(10))
        
    # x property
    @property
    def data(self): return self._data 

    @data.setter
    def data(self, v):
        if not (isinstance(v, np.ndarray) or isinstance(v, list) or isinstance(v, tuple) or isinstance(v, pd.Series) ): raise Exception("Values of the Signal must be a Numpy Array or List or Tuple or Pandas Series or Panda DataFrame")
        if isinstance(v, pd.Series):
            self._data = v
        else:
            # set x
            self._data = pd.Series( v.copy() )

    # method to do a linear covolution sum operation
    def convolution(self, other):
        # get other data
        if isinstance(other, Signal):
            other_data = other.data.values
        elif isinstance(other, pd.Series):
            other_data = other.values
        elif isinstance(other, np.ndarray) or isinstance(other, list) or isinstance(other, tuple):
            other_data = other
        else:
            raise Exception("Invalid argument for convolution")
        # get data
        data = self.data.values.copy()
        # return convolve
        return Signal( np.convolve(data, other_data) )

    # method to do a auto correlation
    def autocorrelation(self):
        # return convulation on self
        return self.convolution(self)

    # method to calculate the fast fourier transform of the signal
    def fft(self):
        # calculate the fft
        fft_data = fft(self.data.values)
        return Signal( fft_data[:len(fft_data)//2] )
        
    # method to calculate the spectrum of the signal
    def spectrum(self):
        # get sampling interval 
        return self.fft().abs()
        
    # method to apply a function
    def apply(self, function = lambda x: x, in_place = True):
        # check if it is in place
        if in_place:
            # set self.data to the output of the functon passed
            self.data = function(self.data)
            # return self
            return self
        else:
            # else, create another data
            data = function( self.data.copy() )
            # create and return another signal
            return Signal(data, index=self.data.index)

    # method to plot on a subplot given
    def subplot(self, y, ax = plt):
        ax.stem(y, linefmt = 'C3-', markerfmt = 'C3.', basefmt='C0:', use_line_collection = True)
        # return plt
        return ax

    # method to plot on a plot given 88
    def plot(self):
        # call data plot
        self.data.plot()
        # show plot
        plt.show()

    # method to stem plot
    def stem(self):
        # get data
        data = self.data.values
        # call subplot
        self.subplot(data, ax)
        # show plot
        plt.show()
    
    # method to calculate the energy of the signal
    def energy(self):
        # return the sum of the abs of the x squared
        data = self.data.copy()
        return np.sum( np.square( np.abs(data) ) )

    # method to calculate the sample sum
    def sum(self):
        # return the sum of the abs of the x squared
        data = self.data.copy()
        return np.sum( data )

    # method to return the real part signal
    def real(self):
        # return the sum of the abs of the x squared
        data = self.data.copy()
        return Signal( data.apply( lambda x: np.real( x ) ) )

    # method to return the imag part signal
    def imag(self):
        # return the sum of the abs of the x squared
        data = self.data.copy()
        return Signal( data.apply( lambda x: np.imag( x ) ) )

    # method to return the abs part signal
    def abs(self):
        # return the sum of the abs of the x squared
        data = self.data.copy()
        return Signal( data.apply( lambda x: np.abs( x ) ) )

    # method to return the angle part signal
    def angle(self):
        # return the sum of the abs of the x squared
        data = self.data.copy()
        return Signal( data.apply( lambda x: np.angle( x ) ) )

    # method to convert to string
    def __str__(self): return "Signal:\n" + str(self.data)

    # method to calculate the length of the signal
    def __len__(self): return len( self.data.values )

    # method to get the value of the signal at index given
    def __getitem__(self, index): return self.data.iloc[index]

    # method to add
    def __add__(self, other):
        if isinstance(other, Signal):
            other = other.data
        return Signal( self.data.add(other, fill_value = 0.0) )

    # method to add
    def __sub__(self, other):
        if isinstance(other, Signal):
            other = other.data
        return Signal( self.data.sub(other, fill_value = 0.0) )

    # method to add
    def __mul__(self, other):
        if isinstance(other, Signal):
            other = other.data
        return Signal( self.data.mul(other, fill_value = 0.0) )

    # method to add
    def __div__(self, other):
        if isinstance(other, Signal):
            other = other.data
        return Signal( self.data.div(other, fill_value = 0.0) )

    # method to add
    def __pow__(self, other):
        if isinstance(other, Signal):
            other = other.data
        return Signal( self.data.pow(other, fill_value = 0.0) )

class TimeSignal(Signal):

    def __init__(self, data, sampling_rate):
        self._sampling_rate = sampling_rate
        super().__init__(data)

    @property
    def sampling_rate(self): return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, i):
        pass
    
    @property
    def total_seconds(self):
        return ( self.data.index[-1] - self.data.index[0] ).total_seconds()

#==========================================================================================================================#
#==========================================================================================================================#

# method do equalize range
def _equalize_range(s1, s2):
    # create n by combining both n
    n = np.mgrid[min(min(s1.n), min(s2.n)):max(max(s1.n), max(s2.n)) + 1]
    # creatw y1 with n length
    y1 = np.zeros(len(n), dtype = np.complex64)
    # create with y2 with length n by copying y1
    y2 = y1.copy()
    # set y1 as s1 x with length n
    y1[ np.where( ( n >= np.min(s1.n) ) & ( n <= np.max(s1.n) )  ) ] = s1.x
    # set y2 as s2 x with length n
    y2[ np.where( ( n >= np.min(s2.n) ) & ( n <= np.max(s2.n) )  ) ] = s2.x
    #return n adn both news ys
    return n, y1, y2

# method to add to signal
def _sigadd(s1, s2):
    # equalize range of both signals
    n, y1, y2 = _equalize_range(s1, s2)
    # final y = y1 + y2
    y = y1 + y2
    #return new singl with x = y and n = n
    return Signal(y, n)

# method to add to signal
def _sigsub(s1, s2):
    # equalize range of both signals
    n, y1, y2 = _equalize_range(s1, s2)
    # final y = y1 + y2
    y = y1 - y2
    #return new singl with x = y and n = n
    return Signal(y, n)

# method to mult 2 signal
def _sigmult(s1, s2):
    # equalize range of both signals
    n, y1, y2 = _equalize_range(s1, s2)
    # final y = y1 * y2
    y = y1 * y2
    #return new singl with x = y and n = n
    return Signal(y, n)

# method to div 2 signals
def _sigdiv(s1, s2):
    # equalize range of both signals
    n, y1, y2 = _equalize_range(s1, s2)
    # final y = y1 * y2
    y = y1 / y2
    #return new singl with x = y and n = n
    return Signal(y, n)

# method to div 2 signals
def _sigexp(s1, s2):
    # equalize range of both signals
    n, y1, y2 = _equalize_range(s1, s2)
    # final y = y1 * y2
    y = y1 ** y2
    #return new singl with x = y and n = n
    return Signal(y, n)

# method to fold an array by half
def _fold_half(array):
    # indices
    idx = list( range( len(array)//2, len(array) ) ) + list( range( len(array)//2 ) )
    # return the array based on the folded idx
    return array[idx]


#==========================================================================================================================#
#==========================================================================================================================#

# function to make a Time Series
def make_time_series(x, n, unit = 'milliseconds'):
    # get now time date
    now = datetime.today()
    # get index based on time step in the unit passed
    index = now + pd.to_timedelta(n, unit=unit)
    # create series
    series = pd.Series(x, index=index)
    # return series
    return series
    
# function to create a unit step signal
def unit_step(n0 = 0, n1 = 0, n2 = 10):
    n = np.arange(n1, n2)
    x = np.zeros(n2-n1)
    x[n0-n1:] = 1
    return Signal( make_time_series(x, n) )
        
# function to create a unit sample signal
def unit_sample(n0 = 0, n1 = 0, n2 = 10):
    n = np.arange(n1, n2)
    x = np.zeros(n2-n1)
    x[n0-n1] = 1
    return Signal( make_time_series(x, n) )

# function to creare sinusoid
def sinusoid(a = 1, o = 0, w = np.pi, n1 = 0, n2 = 10, sampling_rate=0.01):
    n = np.arange(n1, n2, sampling_rate)
    x = a*np.sin(o + w*n)
    return Signal( make_time_series(x, n) )

# function to create random signal
def random_signal(n1 = 0, n2 = 10):
    n = np.arange(n1, n2)
    x = np.random.rand(n2-n1)
    return Signal( make_time_series(x, n) )

# function to creare real exponential
def real_exp(a = 1, n1 = 0, n2 = 10):
    n = np.arange(n1, n2)
    x = a**n
    return Signal( make_time_series(x, n) )

# function to creare complex exponential
def complex_exp(a = 1, o = 0, w = np.pi, n1 = 0, n2 = 10):
    n = np.arange(n1, n2)
    x = a*np.exp( (o + w*1j)*n/100.0 )
    return Signal( make_time_series(x, n) )

# function to create square signal
def square(n0=2, n1=8, n2=0, n3=10):
    return unit_step(n0, n2, n3) - unit_step(n1, n2, n3)

# main function
if __name__ == "__main__":
    signal = complex_exp(w = np.pi/2, n1=0, n2 = 400)
    signal.plot()
    # plot signal
    print(signal)
