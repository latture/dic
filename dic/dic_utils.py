"""
``dic_utils`` contains several utility functions used when analyzing DIC data, e.g. determining the step size,
going from pixel to millimeter coordinates, and determining deformations.

"""
import numpy as np

__all__ = ["get_step", "point_to_indices", "get_initial_position", "get_displacement", "point_to_position"]


def get_step(dic_data):
    """
    Returns the step size of the DIC data

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.

    Returns
    -------
    int
        Step size.
    """
    return dic_data["x"][0, 1] - dic_data["x"][0, 0]


def point_to_indices(dic_data, pt):
    """
    Transforms ``(x, y)`` in pixel coordinates into the corresponding ``(row, col)`` to access the closest data point
    in the specified DIC data.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    pt : (x, x)
        Two-dimensional coordinates of the pixel in global space.

    Returns
    -------
    (row, col) : (int, int)
        The row and column in ``dic_data`` that corresponds to the given pixel point.
    """
    step = get_step(dic_data)
    col = int(round((pt[0] - dic_data["x"].min()) / step))
    row = int(round((pt[1] - dic_data["y"].min()) / step))
    return row, col


def get_initial_position(dic_data, row, col):
    """
    Retrieves the initial position (in mm) held at the specified row and column.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    row : int
        Row in the DIC data to access.
    col : int
        Column in the DIC data to access.

    Returns
    -------
    ``numpy.ndarray``
        Initial ``(x, y, z)`` position in mm.
    """
    return np.array([dic_data["X"][row, col], dic_data["Y"][row, col], dic_data["Z"][row, col]])


def get_displacement(dic_data, row, col):
    """
    Retrieves the displacement (in mm) held at the specified row and column.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    row : int
        Row in the DIC data to access.
    col : int
        Column in the DIC data to access.

    Returns
    -------
    ``numpy.ndarray``
        Displacements ``(u, v, w)`` in mm.
    """
    return np.array([dic_data["U"][row, col], dic_data["V"][row, col], dic_data["W"][row, col]])


def point_to_position(dic_data, pt, add_displacement=True):
    """
    Transforms a point in pixel space into its displaced coordinates in mm.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    pt : (x, y)
        Two-dimensional coordinates of the pixel in global space.
    add_displacement : bool, optional
        Whether to add deformation to the undeformed position. Default is ``True``.

    Returns
    -------
    ``numpy.ndarray``
        ``(x, y, z)`` position in mm of the point.
    """
    row, col = point_to_indices(dic_data, pt)
    pos = get_initial_position(dic_data, row, col)
    if add_displacement:
        pos += get_displacement(dic_data, row, col)
    return pos
