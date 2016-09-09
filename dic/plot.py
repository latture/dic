"""
The :mod:`dic.plot` module contains the functions needed to display DIC overlays using :func:`dic.plot.plot_overlay` and ``(x,y)`` data using
:func:`dic.plot.plot_overlay`. Because the default colorbars from Matplotlib can be somewhat underwhelming (and produce unintended
effects when adding them to an existing figure) the :func:`dic.plot.plot_colorbar` function allows explicit creation of the desired
colorbar.

"""
from __future__ import absolute_import, division, print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, savefig
import matplotlib.image as mpimg
import numpy as np
from .scale import px_per_mm

__all__ = ["plot_overlay", "plot_xy", "plot_colorbar", "show", "savefig"]


def plot_overlay(image_filename, dic_data, key, fig=None, ax=None, contour_options=None, add_colorbar=False,
                 colorbar_title=None, colorbar_options=None):
    """
    Plots a filled contour overlay on top of the specified image.

    Parameters
    ----------
    image_filename : str
        Name of file containing the background image.
    dic_data : dict
        Dictionary containing the DIC data (usually loaded from a ``.mat`` file).
    key : str
        Name of the key to use as the overlay variable. The key must be present in the data loaded from ``dic_filename``.
    fig : ``matplotlib.figure.Figure``, optional
        Figure to add overlay to. If not specified, the figure will be taken from the specified axes. If neither this
        nor ``ax`` is specified, a new figure and axis will be created.
    ax : ``matplotlib.axes.Axes``, optional
        Axes object to add the overlay to. If not specified the axes will be taken from the given figure's current
        axes. If neither this nor ``fig`` is specified a new axes object will be created.
    contour_options : dict, optional
        Keyword arguments to use when calling ``matplotlib.pyplot.contourf``. Default is ``None``
        Valid keys are ``corner_mask``, ``alpha``, ``cmap``, ``levels``, ``extend``, etc.
        See ``matplotlib.pyplot.contourf`` for all valid keys.
    add_colorbar : bool, optional
        Whether to add a colorbar to the overlay axes. Default is ``False``.
    colorbar_title : str, optional
        Title to add to the colorbar. Default is ``None``. If ``None`` is provided, the title is set to the ``key``.
    colorbar_options : dict, optional
        Keyword arguments to use when adding a colorbar to the overlay. Default is ``None``. If ``None`` is provided
        the colorbar will be shrunk slightly such that the color bar is roughly the same height as the overlay.
        Valid key word arguments include ``ticks``, ``format``, ``orientation``, etc.
        See ``matplotlib.figure.colorbar`` for all options.

    Returns
    -------
    ``matplotlib.figure.Figure``, ``matplotlib.axes.Axes``
        Figure and axes containing the background image and contour overlay.
    """
    image = mpimg.imread(image_filename)

    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()

    ax.set_axis_off()
    scale = px_per_mm(dic_data)
    x = dic_data["x"] + dic_data["U"] * scale
    y = dic_data["y"] - dic_data["V"] * scale
    z = dic_data[key]

    if contour_options is None:
        contour_options = {}
    if "levels" not in contour_options and z.min() < z.max():
        contour_options["levels"] = np.linspace(z.min(), z.max(), 10)
    if "cmap" not in contour_options:
        contour_options["cmap"] = plt.cm.viridis
    if "extend" not in contour_options:
        contour_options["extend"] = "both"

    ax.imshow(image, cmap=plt.cm.gray)
    cs = ax.contourf(x, y, z, **contour_options)
    if add_colorbar:
        if colorbar_options is None:
            colorbar_options = {
                "shrink": 0.9
            }
        cbar = fig.colorbar(cs, ax=ax, **colorbar_options)
        if colorbar_title is None:
            cbar.ax.set_title(key)
        else:
            cbar.ax.set_title(colorbar_title)

    return fig, ax


def plot_xy(x, y, fig=None, ax=None, figure_options=None, axes_options=None, plot_options=None):
    """
    Plots the given x, y data.

    Parameters
    ----------
    x : List[T]
        X data points.
    y : List[T]
        Y data points.
    fig : ``matplotlib.figure.Figure``, optional
        Figure to add overlay to. If not specified, the figure will be taken from the specified axes. If neither this
        nor ``ax`` is specified, a new figure and axis will be created.
    ax : ``matplotlib.axes.Axes``, optional
        Axes object to add the overlay to. If not specified the axes will be taken from the given figure's current
        axes. If neither this nor ``fig`` is specified a new axes object will be created.
    figure_options : dict, optional
        Keyword arguments to use when creating the figure. Default is ``None``.
        Valid keys are ``figsize``, ``dpi``, ``facecolor``, ``edgecolor``, ``linewidth``, ``frameon``, ``subplotpars``,
        and ``tight_layout``.
    axes_options : dict, optional
        Keyword arguments to use when adding an axis to the figure. Default is ``None``.
        Valid keys are legal ``matplotlib.axes.Axes`` kwargs plus projection, which chooses a projection type
        for the axes.
    plot_options : dict, optional
        Keyword arguments to use when plotting the data on the axis. Default is ``None``.
        Valid keys are ``matplotlib.lines.Line2D`` properties, e.g. ``alpha``, ``color``, ``label``, etc.

    Returns
    -------
    ``matplotlib.figure.Figure``, ``matplotlib.axes.Axes``
        Figure and axes containing the x-y data.
    """
    if fig is None and ax is None:
        if figure_options is None:
            figure_options = {}
        if axes_options is None:
            axes_options = {}
        fig = plt.figure(**figure_options)
        ax = fig.add_subplot(111, **axes_options)
        ax.tick_params(width=1, length=7)
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()

    if plot_options is None:
        plot_options = {}
    ax.plot(x, y, **plot_options)

    return fig, ax


def plot_colorbar(cmap, vmin, vmax, fig=None, ax=None, figure_options=None, axes_options=None, colorbar_options=None):
    """
    Creates a colorbar of the specified name and normalization.

    Parameters
    ----------
    cmap : str
        Name of the Matplotlib colorbar.
    vmin : float
        Lower limit of the normalization.
    vmax : float
        Upper limit of the normalization.
    fig : ``matplotlib.figure.Figure``, optional
        Figure to add overlay to. If not specified, the figure will be taken from the specified axes. If neither this
        nor ``ax`` is specified, a new figure and axis will be created.
    ax : ``matplotlib.axes.Axes``, optional
        Axes object to add the overlay to. If not specified the axes will be taken from the given figure's current
        axes. If neither this nor ``fig`` is specified a new axes object will be created.
    figure_options : dict, optional
        Keyword arguments to use when creating the figure. Default is ``None``.
        Valid keys are ``figsize``, ``dpi``, ``facecolor``, ``edgecolor``, ``linewidth``, ``frameon``, ``subplotpars``,
        and ``tight_layout``.
    axes_options : dict, optional
        Keyword arguments to use when adding an axis to the figure. Default is ``None``.
        Valid keys are legal ``matplotlib.axes.Axes`` kwargs plus projection, which chooses a projection type
        for the axes.
    colorbar_options : dict, optional
        Keyword arguments to use when adding the colorbar to the axes. Valid keys are ``orientation``, ``ticks``, etc.
        See ``matplotlib.colorbar.ColorbarBase`` for more valid keyword arguments.

    Returns
    -------
    ``matplotlib.figure.Figure``, ``matplotlib.axes.Axes``
        Figure and axes containing the colorbar.
    """
    if fig is None and ax is None:
        if figure_options is None:
            figure_options = {}
        if axes_options is None:
            axes_options = {}
        fig = plt.figure(**figure_options)
        ax = fig.add_subplot(111, **axes_options)
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()

    if colorbar_options is None:
        colorbar_options = {}
    if "orientation" not in colorbar_options:
        colorbar_options["orientation"] = "vertical"

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, **colorbar_options)

    return fig, ax
