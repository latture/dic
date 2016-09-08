from __future__ import absolute_import, division, print_function
import numpy as np

__all__ = ["px_per_mm", "mm_per_px"]


def px_per_mm(dic_data):
    """
    Calculates the average pixels per millimeter.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
        The dictionary must contain the keys ``x``, ``y``, ``X`` and ``Y``.

    Returns
    -------
    float
        Pixels per millimeter.
    """
    # x, y [px] data is not masked. Remove uncorrelated regions before
    # calculating delta:
    valid_data_mask = np.logical_not(dic_data["X"].mask)
    valid_x = dic_data["x"][valid_data_mask]
    valid_y = dic_data["y"][valid_data_mask]

    valid_dx = valid_x - valid_x[0]
    valid_dy = valid_y - valid_y[0]

    valid_X = dic_data["X"].compressed()
    valid_Y = dic_data["Y"].compressed()

    valid_dX = valid_X - valid_X[0]
    valid_dY = valid_Y - valid_Y[0]

    length_px = np.sqrt(np.multiply(valid_dx, valid_dx) + np.multiply(valid_dy, valid_dy)).mean()
    length_mm = np.sqrt(np.multiply(valid_dX, valid_dX) + np.multiply(valid_dY, valid_dY)).mean()

    return length_px / length_mm


def mm_per_px(dic_data):
    """
    Calculates the average millimeters per pixel.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
        The dictionary must contain the keys ``x``, ``y``, ``X`` and ``Y``.

    Returns
    -------
    flost
        Millimeters per pixel.
    """
    return 1.0 / px_per_mm(dic_data)
