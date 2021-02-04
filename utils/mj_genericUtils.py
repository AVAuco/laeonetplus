"""
Utils module

(c) MJMJ/2018
"""
import sys


def mj_isDebugging():
    '''
    Check if program is running in a debugger
    :return: True if debugger is detected, False otherwise
    '''
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        isD = False
    elif gettrace():
        isD = True
    else:
        isD = False

    return isD


import numpy


def mj_smooth(x, window_len=3, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

    Based on http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """

    if x.ndim != 1:
        raise ValueError #, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError #, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError #, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    #print(s)
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')
    padval = int(window_len/2)
    #s = numpy.pad(s, (padval, padval), 'edge')

    y = numpy.convolve(w / w.sum(), s, mode='valid')

    return y[padval:-padval]


def mj_bbarray2bblist(bbarray):
    return [(bbarray[0], bbarray[1]), (bbarray[2], bbarray[3])]

def mj_bblist2bbarray(bblist):
    return numpy.array([bblist[0][0], bblist[0][1], bblist[1][0], bblist[1][1]])
