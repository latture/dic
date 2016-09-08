from __future__ import absolute_import, division, print_function
from collections import namedtuple, Iterable
import itertools
from math import sqrt
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import scipy.optimize as so
import sys
from tqdm import tqdm
from .dic_io import load_dic_data
from .dic_utils import point_to_position, point_to_indices, get_displacement, get_initial_position
from .geometry_utils import norm, distance_to
from .plot import plot_overlay
from .smooth import smooth

__all__ = ["Extensometer", "extensometer_length", "extensometer_transverse_displacement",
           "extensometer_strain", "extensometer_angle", "extensometer_curvature",
           "extensometer_sequence", "place_extensometers"]


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


def extensometer_length(dic_data, extensometer, add_displacement=True):
    """
    Calculates the length of the extensometer.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:.`Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.
    add_displacement : bool, optional
        Whether to add displacement to the undeformed position. Default is ``True``.

    Returns
    -------
    float
        Length of the extensometer.
    """
    pt1_pos = point_to_position(dic_data, extensometer.pt1, add_displacement)
    pt2_pos = point_to_position(dic_data, extensometer.pt2, add_displacement)
    delta_pos = pt2_pos - pt1_pos
    return norm(delta_pos)


def extensometer_transverse_displacement(dic_data, extensometer, fractional_points=10):
    """
    Calculates the displacement transverse to the deformed end-points of the extensometer.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:.`Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.
    fractional_points : int or List[float], optional
        If ``int``, the transverse displacement will be calculated in uniformly spaced intervals
        on the interval ``[0, 1]`` with the number of sample points determined by ``fractional_points``.
        If ``List[float]``, then the given items in the list will be used directly to compute displacement.
        If a ``List`` is given, a value of ``0`` corresponds to ``pt1`` of the extensometer and ``1``
        corresponds to ``pt2`` of the extensometer, i.e. the fractional coordinates are the fraction
        along the extensometer's length at which the transverse displacement should be calculated.
        Default is ``10``.

    Returns
    -------
    List[float]
        List of transverse displacements.
    """
    pt1 = np.asarray(extensometer.pt1)
    pt2 = np.asarray(extensometer.pt2)
    pt1_pos = point_to_position(dic_data, pt1)
    pt2_pos = point_to_position(dic_data, pt2)

    if isinstance(fractional_points, Iterable):
        fractional_points = np.asarray(fractional_points)
    else:
        fractional_points= np.linspace(0, 1, fractional_points)

    transverse_displacement = np.empty(len(fractional_points))

    for i, t in enumerate(fractional_points):
        mid_pt = pt1 + (pt2 - pt1) * t
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
    extensometer : :class:.`Extensometer`
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

    deformed_length = sqrt(delta_pos.dot(delta_pos))

    return deformed_length / initial_length - 1.0


def extensometer_angle(dic_data, extensometer):
    """
    Calculates the angle (in radians) between the specified extensometer and the x-axis.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:.`Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.

    Returns
    -------
    float
        Angle in radians.
    """
    pt1, pt2 = extensometer
    pos1 = point_to_position(dic_data, pt1)
    pos2 = point_to_position(dic_data, pt2)
    delta_pos = pos2 - pos1
    deformed_length = sqrt(delta_pos.dot(delta_pos))

    x_axis = np.array([1.0, 0.0, 0.0])

    c = np.dot(delta_pos, x_axis) / deformed_length
    return np.arccos(np.clip(c, -1.0, 1.0))


def _distance_from_center(center, x, y):
    """
    Calculates the distance of each 2D point from the ``center = (x_center, y_center)``

    Parameters
    ----------
    center: (T, T)
        Two-dimensional data point ``(x, y)`` corresponding to the center of the circle.
    x : T, List[T]
        Point or list of ``x`` data points to calculate the distance to.
    y : T, List[T]
        Point or list of ``y`` data points to calculate the distance to.

    Returns
    -------
    List[float]
        List of distances from the center point.
    """
    x_center, y_center = center
    return np.sqrt((x - x_center)**2 + (y - y_center)**2)


def _leastsq_circle_objective(center, x, y):
    """
    Calculates the algebraic distance from the ``(x, y)`` data points
    and the mean circle centered at ``center = (x_center, y_center)``.

    Parameters
    ----------
    center: (T, T)
        Two-dimensional data point ``(x, y)`` corresponding to the center of the circle.
    x : T, List[T]
        Point or list of ``x`` data points to calculate the distance to.
    y : T, List[T]
        Point or list of ``y`` data points to calculate the distance to.

    Returns
    -------
    List[float]
        List of distances.
    """
    radii = _distance_from_center(center, x, y)
    return radii - radii.mean()


def _fit_leastsq_circle(x, y):
    """
    Fits a circle to the given dataset.

    Parameters
    ----------
    x : List[T]
        x data points.
    y : List[T]
        y data points

    Returns
    -------
    (x_center, y_center, radius, residual) : (float, float, float, float)
        Parameters and residual of the circular fit.
    """
    # coordinates of the barycenter form the initial guess
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    center_estimate = x_mean, y_mean

    center, ier = so.leastsq(_leastsq_circle_objective, center_estimate, args=(x,y))
    x_center, y_center = center
    radii = _distance_from_center(center, x, y)
    mean_radius = radii.mean()
    residual = np.sum((radii - mean_radius)**2)

    return x_center, y_center, mean_radius, residual


def extensometer_curvature(dic_data, extensometer):
    """
    Calculates the curvature of the given extensometer. Curvature is calculated
    by fitting a circle to the data points that are within a distance of
    +/- ``extensometer_length / 4`` of the point with the maximum
    transverse displacement.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    extensometer : :class:.`Extensometer`
        ``(x, y)`` coordinates of the extensometer points in pixel space.

    Returns
    -------
    float
        Curvature.
    """
    # sample 50 points along the extensometer
    num_points = 50
    fractional_points = np.linspace(0, 1, num_points)
    length = extensometer_length(dic_data, extensometer, add_displacement=True)

    transverse_displacement = extensometer_transverse_displacement(dic_data, extensometer, fractional_points)
    masked_transverse_displacement = np.ma.masked_invalid(transverse_displacement, copy=False)
    masked_transverse_displacement = np.ma.masked_invalid(smooth(masked_transverse_displacement))
    max_disp_idx = np.argmax(masked_transverse_displacement)

    # number of indices that correspond to a distance of L / 4
    # where L = extensometer length
    quarter_idx = num_points // 4

    # get the indices of the points that are +/- L/4 away from the
    # maximum displacement. Bound indices on the range [0, num_points]
    right_idx = min(num_points, max_disp_idx + quarter_idx)
    left_idx = max(0, max_disp_idx - quarter_idx)

    # extract the (x, y) data that will be used for the circular fit
    x_circular = length * fractional_points[left_idx:right_idx]
    y_circular = masked_transverse_displacement[left_idx:right_idx]

    x_center, y_center, radius, residual = _fit_leastsq_circle(x_circular, y_circular)

    return 1.0 / radius


def extensometer_sequence(dic_filenames, extensometers, metric, description="Processing extensometers"):
    """
    Extracts the specified metric from the extensometers for all DIC files.

    Parameters
    ----------
    dic_filenames : str
        Names of DIC files to analyze.
    extensometers : List[:class:`.Extensometer`]
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
    args : (str, List[:class:`.Extensometer`], callable)
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
