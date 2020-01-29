# signal lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import collections
from scipy.signal import lfilter
import seaborn as sns; sns.set()

from datetime import datetime, time, date, timedelta

class Signal:

    def __init__(self, x = None):
        self.x = x.copy() if x is not None else np.arange(len(10))
         
    # x property
    @property
    def x(self): return self._x 

    @x.setter
    def x(self, v):
        if not (isinstance(v, np.ndarray) or isinstance(v, list) or isinstance(v, tuple) or isinstance(v, pd.Series)): raise Exception("Values of the Signal must be a Numpy Array or List or Tuple or Pandas Series")
        if isinstance(v, pd.Series):
            self._x = v
        else:
            # create time series
            now = datetime.today()
            # index by time
            index = now + pd.to_timedelta(v, unit='ms') - now 
            # set x
            self._x = pd.Series(v.copy(), index = index)

    # method to do a linear covolution sum operation
    def convolution(self, other):
        # check if other is instance of Signal
        assert (isinstance(other, Signal) ), "Convulation Operation must be done with another Signal"
        # new x = convolution of both x
        x = np.convolve(self.x, other.x)
        # get min values of n of boths signals
        n_min = self.n[0] + other.n[0]
        # get max values of n of boths signals
        n_max = self.n[-1] + other.n[-1]
        # new n = range of n min to n max
        n = np.arange(n_min, n_max+1)

        # ceate and return new signal with x and n 
        return Signal(x, n)

    # method to do a auto correlation
    def autocorrelation(self):
        # return convulation on self
        return self.convolution(self)

    # method to calculate the even signal component
    def even(self):
        # check if dtype of self.x is complex
        assert (self.x.dtype != np.complex128 and self.x.dtype != np.complex64), "Cant Get Even with complex values"
        # sum new signal identical to self + fold
        s = Signal(self.x, self.n) + self.fold()
        # return new signal s times 0.5
        return s*0.5

    # method to calculate the even signal component
    def odd(self):
        # check if dtype of self.x is complex
        assert (self.x.dtype != np.complex128 and self.x.dtype != np.complex64), "Cant Get Even with complex values"
        # sum new signal identical to self + fold
        s = Signal(self.x, self.n) - self.fold()
        # return new signal s times 0.5
        return s*0.5

    # method to apply a function
    def apply(self, function = lambda x: x, in_place = True):
        # check if it is in place
        if in_place:
            # set self.x to the output of the functon passed
            self.x = function(self.x)
            # return self
            return self
        else:
            # else, create another x
            x = function(self.x.copy())
            # create and return another signal
            return Signal(x, self.n)

    # method to plot on a subplot given
    def subplot(self, x, y, ax = plt):
        ax.stem(x, y, linefmt = 'C3-', markerfmt = 'C3.', basefmt='C0:', use_line_collection = True)
        # return plt
        return ax

    # method to plot on a plot given
    def plot(self, ax = None):
        # check if plt is None
        if ax is None:
            ax = plt.subplot(1, 1, 1)
        
        # call sublot and get plt plotted
        self.subplot(self.x.index.microseconds/1000, self.x, ax)
        # show plot
        plt.show()

    # method to calculate the fast fourier transform of the signal
    def fft(self, n = None, fold = False):
        # calculate the fft
        x = np.fft.fft(self.x, n)
        # check fold
        if fold:
            x = np.concatenate( [ x[len(x)//2:], x[:len(x)//2] ] )
        # else, dont change it
        # create n array based of length of x
        n = np.arange(len(x))
        # return signal creted
        sig = Signal(x, n)
        # check if folded
        if fold:
            return _fold_signal(sig)
        else:
            return sig
    
    # method to calculate the spectrum of the signal
    def spectrum(self):
        # calcualte the fast furier transform
        fft = self.fft()
        # get sampling interval 
        T = (self.n[1] - self.n[0]) / len(self.n)
        # get number of frenquencies  
        N = self.x.size
        # 1/T = frequency
        f = np.linspace(0, 1 / T, N)
        # plot
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency [Hz]")
        # plot as bar
        plt.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, width=0.4)  # 1 / N is a normalization factor
        # show plot
        plt.show()
    
    # method to calculate the full plot analisys
    def plot_analysis(self, n = None, fold = False):
        # check if fold is True
        if fold:
            x = _fold_half(self.x)
        else:
            x = self.x
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
    
    # method to calculate the energy of the signal
    def energy(self):
        # return the sum of the abs of the x squared
        return np.sum( np.square( np.abs(self.x) ) )

    # method to calculate the sample sum
    def sum(self):
        # return sum of values of the signal
        return np.sum(self.x)

    # method to calculate the sample prod
    def prod(self):
        # return prod of values of the signal
        return np.prod(self.x)

    # method to do a fold operation
    def fold(self):
        # copy x and n
        x = self.x.copy()
        n = self.n.copy()
        #flip x and n
        x = x[::-1]
        n = n[::-1]
        # reverse n
        n = n*-1
        # create and return new signal
        return Signal(x, n)

    # method to return the real part signal
    def real(self):
        return Signal(np.real(self.x), self.n)

    # method to return the imag part signal
    def imag(self):
        return Signal(np.imag(self.x), self.n)

    # method to return the abs part signal
    def abs(self):
        return Signal(np.abs(self.x), self.n)

    # method to return the angle part signal
    def angle(self):
        return Signal(np.angle(self.x), self.n)

    # method to to a shift of the signal
    def __rshift__(self, k):
        # increase indices of n (so the values in k will be shifted to the right)
        n = self.n.copy() + k
        # create and return new signal with x = self.x and n = new n
        return Signal(self.x.copy(), n)

    # method to to a shift of the signal    
    def __lshift__(self, k):
        # decrease indices of n (so the values in k will be shifted to the left)
        n = self.n.copy() - k
        # create and return new signal with x = self.x and n = new n
        return Signal(self.x.copy(), n)

    # method to compare the signal    
    def __eq__(self, other):
        if isinstance(other, Signal):
            return self.n == other.n and self.x == other.n
        return False

    # method to compare the signal    
    def __ne__(self, other):
        # call == method operator and return its negation
        return not self.__eq__(other)

    # method to convert to string
    def __str__(self): return str(self.x)

    # method to calculate the length of the signal
    def __len__(self): return len(self.n)

    # method to get the value of the signal at index given
    def __getitem__(self, index): return self.x.iloc[index]

    # method to add
    def __add__(self, other):
        # check if other is a scalar
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray):
            # mult self.x by the scalar other to be the new x
            x = self.x + other
            # create and return new signal
            return Signal(x, self.n)
        elif isinstance(other, Signal):
            # else, call sigmult and return teh new signal created
            return _sigadd(self, other)
        else:
            raise Exception("Arithmetic Opeartion must be with another int, float, numpy array or signal")

    # method to sub
    def __sub__(self, other):
        # check if other is a scalar
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray):
            # mult self.x by the scalar other to be the new x
            x = self.x - other
            # create and return new signal
            return Signal(x, self.n.copy())
        elif isinstance(other, Signal):
            # else, call sigmult and return teh new signal created
            return _sigsub(self, other)
        else:
            raise Exception("Arithmetic Opeartion must be with another int, float, numpy array or signal")

    # method to add
    def __div__(self, other):
        # check if other is a scalar
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray):
            # mult self.x by the scalar other to be the new x
            x = self.x / other
            # create and return new signal
            return Signal(x, self.n.copy())
        elif isinstance(other, Signal):
            # else, call sigmult and return teh new signal created
            return _sigdiv(self, other)
        else:
            raise Exception("Arithmetic Opeartion must be with another int, float, numpy array or signal")

    # method to mult
    def __mul__(self, other):
        # check if other is a scalar
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray):
            # mult self.x by the scalar other to be the new x
            x = self.x * other
            # create and return new signal
            return Signal(x, self.n.copy())
        elif isinstance(other, Signal):
            # else, call sigmult and return teh new signal created
            return _sigmult(self, other)
        else:
            raise Exception("Arithmetic Opeartion must be with another int, float, numpy array or signal")



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
    index = now + pd.to_timedelta(n, unit=unit) - now 
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
def sinusoid(a = 1, o = 0, w = np.pi, n1 = 0, n2 = 10):
    n = np.arange(n1, n2)
    x = a*np.sin(o + w*n/100.0)
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
    x = a*np.exp( (o + w*1j)*n )
    return Signal( make_time_series(x, n) )

# main function
if __name__ == "__main__":
    signal = unit_step(100, n1=0, n2 = 200)
    # plot signal
    print(signal)
    print(signal.x.index)
    signal.plot()
    