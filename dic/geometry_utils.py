"""
The ``geometry_utils`` module contain utility functions (not specific to DIC) that aid in applying geometric transformations
or determining spatial relationships between data point, usually in 3D space.

"""

from math import acos, cos, sin, sqrt
import numpy as np
from numba import jit

__all__ = ["norm", "apply_rotation", "distance_between", "distance_to",
           "get_angle", "get_transformation_matrix", "apply_transformation", "cross3d"]


@jit
def norm(x):
    """
    Calculates the norm of a vector.

    Parameters
    ----------
    x : array_like
        Input vector.

    Returns
    -------
    float
        Norm of the vector.
    """
    x = np.asarray(x)
    return sqrt(x.dot(x))


@jit
def cross3d(a, b):
    """
    Calculates the cross product between two 3D vectors.

    Parameters
    ----------
    a : array_like
        ``(x, y, z)`` components of the first vector.

    b : array_like
        ``(x, y, z)`` components of the second vector.

    Returns
    -------
    ``numpy.ndarray``
        Vector cross product.
    """
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


@jit
def apply_rotation(x, y, theta):
    """
    Rotates the given ``(x, y)`` indices by an amount ``theta``.

    Parameters
    ----------
    x : int
        Coordinate with respect to the x-axis.
    y : int
        Coordinate with respect to the y-axis.
    theta : T
        Angle in radians.

    Returns
    -------
    (x_new, y_new) : (int, int)
        Rotated coordinates.
    """
    s = sin(theta)
    c = cos(theta)
    x_new = int(x * c - y * s)
    y_new = int(x * s + y * c)
    return x_new, y_new


def distance_between(pt1, pt2):
    """
    Calculates the distance between the given points.

    Parameters
    ----------
    pt1 : List[T]
        Coordinate of the first point.
    pt2 : List[T]
        Coordinate of the second point.

    Returns
    -------
    float
        Distance
    """
    pt1 = np.asarray(pt1)
    pt2 = np.asarray(pt2)
    delta = pt2 - pt1
    return norm(delta)


def distance_to(pt0, pt1, pt2):
    """
    Computes the distance between ``pt0`` and the line that passes through
    ``pt1`` and ``pt2``.

    Parameters
    ----------
    pt0 : ``numpy.ndarray``
        Point to calculate distance to.
    pt1 : ``numpy.ndarray``
        First point that defines the line.
    pt2 : ``numpy.ndarray``
        Second point that defines the line.

    Returns
    -------
    float
        Distance.
    """
    d_21 = pt2 - pt1
    d_10 = pt1 - pt0
    l_21 = norm(d_21)
    l_21_10 = norm(cross3d(d_21, d_10))
    return l_21_10 / l_21


def get_angle(pt1, pt2):
    """
    Calculates the angle between the given two-dimensional points and the x-axis.

    Parameters
    ----------
    pt1 : List[T]
        (x, y) coordinate of the first point.
    pt2 : List[T]
        (x, y) coordinate of the second point.

    Returns
    -------
    float
        Angle in radians.
    """
    dx = pt2[0] - pt1[0]
    length = distance_between(pt1, pt2)
    return acos(dx / length)


def get_transformation_matrix(axis, theta, translation, scale):
    """
    Returns the 4x4 transformation matrix that applies the specified
    scale, translation, and rotation.

    Parameters
    ----------
    axis : array_like
        Axis of rotation.
    theta : float
        Angle of rotation in radians.
    translation : array_like
        Translation vector.
    scale : float
        Uniform scale factor.

    Returns
    -------
    ``numpy.ndarray``
        4x4 transformation matrix.
    """
    axis = np.asarray(axis)
    axis /= norm(axis)
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    tx, ty, tz = translation
    return np.array([[aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac),   tx],
                     [2*(bc-ad),   aa+cc-bb-dd, 2*(cd+ab),   ty],
                     [2*(bd+ac),   2*(cd-ab),   aa+dd-bb-cc, tz],
                     [0.,          0.,          0.,          1. / scale]])


def apply_transformation(pt, transformation_matrix, offset):
    """
    Applies the affine transformation to the point.

    Parameters
    ----------
    pt : array_like
        ``(x, y, z)`` coordinate.
    transformation_matrix : array_like
        4x4 transformation_matrix.
    offset : array_like
        Offset from the origin.

    Returns
    -------
    ``numpy.ndarray``
        Tranformed ``(x, y, z)`` coordinate.
    """
    x = np.array([pt[0], pt[1], pt[2], 1.])
    x[:3] -= offset
    x = transformation_matrix.dot(x)
    x = x[:-1] / x[-1]
    x += offset
    return x
