"""
The :mod:`dic.smooth` module allows a vector of data to be smoothed, often alleviating some of the undesired spikes that
occurs during data acquisition.

"""
from __future__ import absolute_import, division, print_function
import numpy as np

__all__ = ["smooth"]


def smooth(x, window_len=10, window='hanning'):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) at both ends of the data so that transient parts are minimized
    in the beginning and end of the output signal.

    Parameters
    ----------
    x : List[T]
        The input signal.
    window_len : int, optional
        The dimension of the smoothing window. Default is ``10``.
    window : {'moving_average', 'hanning', 'hamming', 'bartlett', 'blackman'}, optional
        The type of window. Default is ``'hanning'``.

    Returns
    -------
    ``numpy.ndarray``
        The smoothed signal.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(-2, 2, 10)
    >>> x = np.sin(t) + np.random.randn(len(t)) * 0.1
    >>> smooth(x)
    array([-0.97986086, -0.76226712, -0.51086275, -0.22217585,  0.08607285,
            0.37779472,  0.61419799,  0.772406  ,  0.85738291])

    References
    ----------
    This function was modified from this source_.

    .. _source: https://raw.githubusercontent.com/scipy/scipy-cookbook/master/ipython/SignalSmooth.ipynb`_.
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("Only 1 dimensional arrays are accepted.")
    if x.size < window_len:
        raise ValueError("Input vector must be bigger than window size.")
    if window_len < 3:
        return x
    if window not in ['moving_average', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window {} is not valid. Choose from 'moving_average', 'hanning', "
                         "'hamming', 'bartlett', 'blackman'".format(window))

    if window == 'moving_average':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)

    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    y = np.convolve(w / w.sum(), s, mode='same')

    return y[window_len - 1:-window_len + 1]
