"""
The :mod:`dic.extensometer` module provides a means of easily creating and analyzing extensometers. An extensometer is
defined by two nodal locations connected by a straight line. Metrics can be extracted from extensometers such as
the length, angle, strain, etc. If the same metric needs to be extracted from the same extensometer for several
DIC files, then the :func:`dic.extensometer.extensometer_sequence` function allows a set of DIC filenames, extensometers and metric to be provided,
and the relevant metric will be extracted for the given extensometers for all the DIC files. Placing extensometers
can be accomplished by using the :func:`dic.extensometers.place_extensometers` function which, when given the reference image and DIC filenames,
allows extensometer placement by clicking on a Matplotlib figure.

"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple, Iterable
import itertools
from math import sqrt
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from scipy.interpolate import UnivariateSpline
import sys
from tqdm import tqdm
import warnings
from .dic_io import load_dic_data
from .dic_utils import point_to_position, point_to_indices, get_displacement, get_initial_position
from .geometry_utils import norm, distance_to
from .plot import plot_overlay
from .smooth import smooth

__all__ = ["Extensometer", "extensometer_to_position", "extensometer_length",
           "extensometer_rectified_length", "extensometer_transverse_displacement",
           "extensometer_strain", "extensometer_neutral_axis_strain", "extensometer_angle",
           "extensometer_curvature", "extensometer_sequence", "place_extensometers"]


class Extensometer(namedtuple("Extensometer", ["pt1", "pt2"])):
    """
    An extensometer that is defined between two points in pixel space.

    Attributes
    ----------
    pt1 : (x, y)
        Two-dimensional coordinates of the first pixel in the reference image.
    pt2 : (x, y)
        Two-dimensional coordinates of the second pixel in the reference image.
    """


def extensometer_to_position(dic_data, extensometer, add_displacement=True):
    """
    Converts an extensometer from pixel to coordinate space, optionally adding the displacement of the end points.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:`dic.extensometer.Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.
    add_displacement : bool, optional
        Whether to add displacement to the undeformed position. Default is ``True``.

    Returns
    -------
    (x1, y1, z1), (x2, y2, z2) : ``numpy.ndarray``, ``np.ndarray``
        Two arrays of three values. Each specifies the ``(x, y, z)`` location (in mm) of the extensometer end point.
    """
    pt1, pt2 = extensometer
    pt1_pos = point_to_position(dic_data, pt1, add_displacement=add_displacement)
    pt2_pos = point_to_position(dic_data, pt2, add_displacement=add_displacement)
    return pt1_pos, pt2_pos


def extensometer_length(dic_data, extensometer, add_displacement=True):
    """
    Calculates the linear distance between the two endpoints of the extensometer.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:`dic.extensometer.Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.
    add_displacement : bool, optional
        Whether to add displacement to the undeformed position. Default is ``True``.

    Returns
    -------
    float
        Length of the extensometer.
    """
    pt1_pos, pt2_pos = extensometer_to_position(dic_data, extensometer, add_displacement=add_displacement)
    delta_pos = pt2_pos - pt1_pos
    return norm(delta_pos)


def _interpolate_point(extensometer, t):
    """
    Linearly interpolates between the two endpoints of the extensometer.

    Parameters
    ----------
    extensometer : :class:`dic.extensometer.Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.
    t : float, [0., 1.]
        At ``0.`` the first point in the extensometer is returned.
        At ``1.`` the second point is returned. In between the pixel coordinates
        are linearly interpolated between the endpoints.

    Returns
    -------
    (x, y) : (float, float)
        Position (in pixels) of the interpolated point.
    """
    pt1, pt2 = extensometer
    pt1 = np.asarray(pt1)
    pt2 = np.asarray(pt2)
    return pt1 + (pt2 - pt1) * t


def _convert_fractional_points_to_array(fractional_points):
    """
    Converts fractional points input parameter to a Numpy ``ndarray``.

    Parameters
    ----------
    fractional_points : int or List[int]
        If ``int``, uniformly spaced intervals on the interval ``[0, 1]`` with the number of sample points
        determined by ``fractional_points``. If ``List[float]``, the given items in the list will be returned as an ``ndarray``.

    Returns
    -------
    ``numpy.ndarray``
        List of fractional points on the range ``[0, 1]``
    """
    if isinstance(fractional_points, Iterable):
        return np.asarray(fractional_points)
    return np.linspace(0, 1, fractional_points)


def _extensometer_interpolated_points(dic_data, extensometer, fractional_points=50, window_len=10, add_displacement=True):
    """
    Interpolates and smoothes the 3D points between extensometer endpoints. Uncorrelated intermediary points are removed
    from the data set. Each row in the returned array contains``[t, x, y, z]`` where ``t`` is the fractional
    length along the extensometer on the range ``[0, 1]`` and ``x, y, z`` represent the corresponding interpolated point.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:`dic.extensometer.Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.
    fractional_points : int or List[int], optional
        If ``int``, uniformly spaced intervals on the interval ``[0, 1]`` with the number of sample points
        determined by ``fractional_points``. If ``List[float]``, then the given items in the list will be used directly
        to compute displacement. A value of ``0`` corresponds to ``pt1`` of the extensometer and ``1``corresponds
        to ``pt2`` of the extensometer. Default is ``50``.
    window_len : int, optional
        The dimension of the smoothing window. Default is ``10``.
    add_displacement : bool, optional
        Whether to add displacement to the undeformed position. Default is ``True``.

    Returns
    -------
    txyz : ``numpy.ndarray``
        Array of ``[t, x, y, z]`` data points where ``t`` is the fractional length along the extensometer and ``x,y,z``
        represent the interpolated coordinate.
    """
    # get the interpolated position of the extensometer in 3D:
    fractional_points = _convert_fractional_points_to_array(fractional_points)
    extensometer_pos = np.empty((len(fractional_points), 4))
    extensometer_pos[:, 0] = fractional_points
    for i, t in enumerate(fractional_points):
        pt = _interpolate_point(extensometer, t)
        extensometer_pos[i, 1:] = point_to_position(dic_data, pt, add_displacement=add_displacement)

    # remove any uncorrelated points:
    # if an interpolated point is uncorrelated then all (x, y, z)
    # values will be nan, so we only need to check isnan on
    # one column (in the case I take the x-direction).
    is_valid = np.logical_not(np.isnan(extensometer_pos[:, 1]))
    valid_pos = extensometer_pos[is_valid]
    num_valid_points = len(valid_pos)

    if num_valid_points < window_len:
        new_window_len = num_valid_points - 1
        what = "The number of valid points along the extensometer ({0:d}) is less than the window_len ({1:d}). " \
               "The window_len for the current extensometer has been shortened to {0:d}.".format(num_valid_points,
                                                                                                 window_len,
                                                                                                 new_window_len)
        warnings.warn(what)
        window_len = new_window_len

    # smooth the data along each axis
    smoothed_pos = np.empty_like(valid_pos)
    smoothed_pos[:, 0] = valid_pos[:, 0]
    for i in range(1, valid_pos.shape[-1]):
        smoothed_pos[:, i] = smooth(valid_pos[:, i], window_len=window_len)

    return smoothed_pos


def extensometer_rectified_length(dic_data, extensometer, fractional_points=50, window_len=10, add_displacement=True):
    """
    Approximates the length of an extensometer by connecting a finite amount of points along the extensometer
    and summing the length of each line segment. If the path between the extensometer end points is a straight line
    or ``fractional_points = 2``, this function and :func:`dic.extensometer.extensometer_length` will yield the same results.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:`dic.extensometer.Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.
    fractional_points : int or List[int]
        If ``int``, uniformly spaced intervals on the interval ``[0, 1]`` with the number of sample points
        determined by ``fractional_points``. If ``List[float]``, the given items in the list will be returned
        as an ``ndarray``. Adjacent fractional points will be joined to form the line segments that will
        be used to determine the length of the extensometer.
    window_len : int, optional
        The dimension of the smoothing window. Default is ``10``.
    add_displacement : bool, optional
        Whether to add displacement to the undeformed position. Default is ``True``.

    Returns
    -------
    float
        Length of the extensometer.
    """
    smoothed_pos = _extensometer_interpolated_points(dic_data, extensometer,
                                                     fractional_points=fractional_points,
                                                     window_len=window_len,
                                                     add_displacement=add_displacement)

    # find total length by summing the lengths of each individual segment
    pt1 = smoothed_pos[:-1, 1:]
    pt2 = smoothed_pos[1:, 1:]
    segment_lengths = np.linalg.norm(pt2 - pt1, axis=1)
    return np.sum(segment_lengths)


def extensometer_transverse_displacement(dic_data, extensometer, fractional_points=10):
    """
    Calculates the displacement transverse to the deformed end-points of the extensometer.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:`dic.extensometer.Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.
    fractional_points : int or List[float], optional
        If ``int``, the transverse displacement will be calculated in uniformly spaced intervals
        on the interval ``[0, 1]`` with the number of sample points determined by ``fractional_points``.
        If ``List[float]``, then the given items in the list will be used directly to compute displacement.
        A value of ``0`` corresponds to ``pt1`` of the extensometer and ``1``corresponds to ``pt2`` of the extensometer,
        i.e. the fractional coordinates are the fraction along the extensometer's length at which the transverse
        displacement should be calculated. Default is ``10``.

    Returns
    -------
    List[float]
        List of transverse displacements.
    """
    pt1_pos, pt2_pos = extensometer_to_position(dic_data, extensometer)
    fractional_points = _convert_fractional_points_to_array(fractional_points)
    transverse_displacement = np.empty(len(fractional_points))
    for i, t in enumerate(fractional_points):
        mid_pt = _interpolate_point(extensometer, t)
        mid_pos = point_to_position(dic_data, mid_pt)
        transverse_displacement[i] = distance_to(mid_pos, pt1_pos, pt2_pos)
    return transverse_displacement


def extensometer_strain(dic_data, extensometer):
    """
    Calculates the strain between the specified points. Points should be specified by their pixel location.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:`dic.extensometer.Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.

    Returns
    -------
    float
        Strain, i.e. ``deformed_length / initial_length - 1.0``.
    """
    pt1, pt2 = extensometer
    row1, col1 = point_to_indices(dic_data, pt1)
    row2, col2 = point_to_indices(dic_data, pt2)
    pos1 = get_initial_position(dic_data, row1, col1)
    pos2 = get_initial_position(dic_data, row2, col2)
    delta_pos = pos2 - pos1

    initial_length = sqrt(delta_pos.dot(delta_pos))

    disp1 = get_displacement(dic_data, row1, col1)
    disp2 = get_displacement(dic_data, row2, col2)

    pos1 += disp1
    pos2 += disp2
    delta_pos = pos2 - pos1

    deformed_length = norm(delta_pos)

    return deformed_length / initial_length - 1.0


def extensometer_neutral_axis_strain(dic_data, extensometer, fractional_points=50, window_len=10):
    """
    Calculates the strain of the neutral axis of the extensometer by comparing the rectified extensometer
    length with and without displacements applied. If the extensometer is approximately straight,
    this function will yield nearly the same value as :func:`dic.extensometer.extensometer_strain`
    which only considers nodal end points of the extensometer.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:`dic.extensometer.Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.
    fractional_points : int or List[int]
        If ``int``, the transverse displacement will be calculated in uniformly spaced intervals
        on the interval ``[0, 1]`` with the number of sample points determined by ``fractional_points``.
        If ``List[float]``, the given items in the list will be used directly to compute strain.
        A value of ``0`` corresponds to ``pt1`` of the extensometer and ``1``corresponds to ``pt2`` of the extensometer.
        Default is ``50``.
    window_len : int, optional
        The dimension of the smoothing window. Default is ``10``.

    Returns
    -------
    float
        Strain of the neutral axis
    """
    initial_length = extensometer_rectified_length(dic_data, extensometer,
                                                   fractional_points=fractional_points,
                                                   add_displacement=False,
                                                   window_len=window_len)
    deformed_length = extensometer_rectified_length(dic_data, extensometer,
                                                    fractional_points=fractional_points,
                                                    add_displacement=True,
                                                    window_len=window_len)
    return deformed_length / initial_length - 1.0


def extensometer_angle(dic_data, extensometer, add_displacement=True):
    """
    Calculates the angle (in radians) between the specified extensometer and the x-axis.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:`dic.extensometer.Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.

    Returns
    -------
    float
        Angle in radians.
    """
    pt1_pos, pt2_pos = extensometer_to_position(dic_data, extensometer, add_displacement=add_displacement)
    delta_pos = pt2_pos - pt1_pos
    deformed_length = norm(delta_pos)
    x_axis = np.array([1.0, 0.0, 0.0])
    c = np.dot(delta_pos, x_axis) / deformed_length
    return np.arccos(np.clip(c, -1.0, 1.0))


def extensometer_curvature(dic_data, extensometer, fractional_points=50, window_len=10):
    """
    Calculates the curvature along the length of the given extensometer.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:`dic.extensometer.Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.
    fractional_points : int or List[int]
        If ``int``, the transverse displacement will be calculated in uniformly spaced intervals
        on the interval ``[0, 1]`` with the number of sample points determined by ``fractional_points``.
        If ``List[float]``, then the given items in the list will be used directly to compute displacement.
        A value of ``0`` corresponds to ``pt1`` of the extensometer and ``1``corresponds to ``pt2`` of the extensometer,
        Default is ``50``.
    window_len : int, optional
        The dimension of the smoothing window. Default is ``10``.

    Returns
    -------
    ``numpy.ndarray``
        Curvature at the specified ``fractional_points``.
    """
    t, x, y, z = _extensometer_interpolated_points(dic_data, extensometer,
                                                   fractional_points=fractional_points,
                                                   window_len=window_len,
                                                   add_displacement=True).T

    fx = UnivariateSpline(t, x)
    fy = UnivariateSpline(t, y)
    fz = UnivariateSpline(t, z)

    dfx_dt = fx.derivative(1)(t)
    dfy_dt = fy.derivative(1)(t)
    dfz_dt = fz.derivative(1)(t)

    d2fx_dt2 = fx.derivative(2)(t)
    d2fy_dt2 = fy.derivative(2)(t)
    d2fz_dt2 = fz.derivative(2)(t)

    return np.sqrt((d2fz_dt2 * dfy_dt - d2fy_dt2 * dfz_dt) ** 2 +
                   (d2fx_dt2 * dfz_dt - d2fz_dt2 * dfx_dt) ** 2 +
                   (d2fy_dt2 * dfx_dt - d2fx_dt2 * dfy_dt) ** 2) / (dfx_dt ** 2 + dfy_dt ** 2 + dfz_dt ** 2) ** 1.5


def extensometer_sequence(dic_filenames, extensometers, metric, description="Processing extensometers"):
    """
    Extracts the specified metric from the extensometers for all DIC files.

    Parameters
    ----------
    dic_filenames : str
        Names of DIC files to analyze.
    extensometers : List[:class:`dic.extensometer.Extensometer`]
        List of extensometers to
    metric : callable
        A callable object that provides the signature ``metric(dict, Extensometer) -> float``, i.e. takes as input
        a ``dict`` of dic_data and an :class:`.Extensometer` and returns a ``float``.
    description : str, optional
        Description of the computation that will be printed along with the progress bar.
        Default is ``"Processing extensometers"``.

    Returns
    -------
    ``numpy.ndarray``
        Array of size ``len(dic_filenames)`` by ``len(extensometers)``
    """
    num_dic_filenames = len(dic_filenames)
    num_extensometers = len(extensometers)
    extensometer_data = np.empty((num_dic_filenames, num_extensometers))

    args = zip(
        dic_filenames,
        itertools.repeat(extensometers, num_dic_filenames),
        itertools.repeat(metric, num_dic_filenames)
    )

    pool = mp.Pool()
    for i, row in enumerate(tqdm(pool.imap(_extensometer_sequence_worker, args),
                                 total=num_dic_filenames, file=sys.stdout, desc=description)):
        extensometer_data[i] = row
    pool.close()
    pool.join()

    return extensometer_data


def _extensometer_sequence_worker(args):
    """
    Worker for extracting the extensometer metric from a single DIC data file.

    Parameters
    ----------
    args : (str, List[:class:`dic.extensometer.Extensometer`], callable)
        Tuple of arguments needed for each calculation, i.e. ``(dic_filename, extensometers, metric)``.

    returns
    -------
    List[float]
        The provided metric calculated for each extensometer.
    """
    dic_filename, extensometers, metric = args
    dic_data = load_dic_data(dic_filename, variable_names=("x", "y", "X", "Y", "Z", "U", "V", "W", "sigma"))
    output = []
    for extensometer in extensometers:
        output.append(metric(dic_data, extensometer))

    return output


def place_extensometers(reference_image_filename, reference_dic_filename):
    """
    Places extensometers on the given set of reference data. Clicking the figure will add a point. Adding two points
    will create an extensometer.

    Parameters
    ----------
    reference_image_filename : str
        Name of the file containing the reference image, e.g. the first image of the DIC data.
    reference_dic_filename : str
        Name of the file containing the reference DIC data, e.g. the DIC data calculated from the reference image.

    Returns
    -------
    List[:class:`.Extensometer`]
        List of extensometers added by the user.
    """
    fig, ax = plot_overlay(reference_image_filename, load_dic_data(reference_dic_filename), "sigma", add_colorbar=True)

    current_extensometer_points = []
    extensometers = []

    ax.set_title("Left click to add point")

    def on_click(event):
        if not event.inaxes:
            return
        current_extensometer_points.append([event.xdata, event.ydata])
        ax.plot([event.xdata], [event.ydata], marker="o", color="#F44336", markersize=10)

        if len(current_extensometer_points) == 2:
            pt2 = current_extensometer_points.pop()
            pt1 = current_extensometer_points.pop()
            extensometers.append(Extensometer(pt1=pt1, pt2=pt2))
            xdata = [pt1[0], pt2[0]]
            ydata = [pt1[1], pt2[1]]
            ax.plot(xdata, ydata, color="#F44336", linestyle="-", linewidth=2)

        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    return extensometers
