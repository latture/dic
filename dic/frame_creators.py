from __future__ import absolute_import, division, print_function
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from .dic_io import load_dic_data
from .plot import plot_overlay, plot_colorbar, plot_xy

__all__ = ["OverlayWithStressStrainFrameCreator"]


class OverlayWithStressStrainFrameCreator(object):
    """
    This class creates a frames of a contour overlay, colorbar and stress-strain plot.

    Attributes
    ----------
    image_filenames : List[str]
        List of filenames containing the background images for the overlay plots.
    dic_filenames : str
        List of filenames containing the DIC data files.
    variable : str
        Variable from the DIC data to plot on the overlay.
    xy_data : List[T], List[T]
        Tuple of ``(xdata, ydata)`` to plot on the 2D plot. The length of ``xdata`` and ``ydata`` must be equal and the
        same as the number of images and DIC files.
    figure_width : float, optional
        Width of the figure in inches. The height will be chosen automatically based on the aspect ratio of the overlays.
        Default is ``10.0``.
    fractional_padding : float, optional
        Fraction of the image width to pad between the right of the colorbar and the left of the x-y data plot.
        Default is ``0.5``
    overlay_contour_options : dict, optional
        Keyword arguments to use when calling ``matplotlib.pyplot.contourf``. Default is ``None``
        Valid keys are ``corner_mask``, ``alpha``, ``cmap``, ``levels``, ``extend``, etc.
        See ``matplotlib.pyplot.contourf`` for all valid keys.
    xy_axes_options : dict, optional
        Keyword arguments to the x-y data axes. Default is ``None``.
        Valid keys are legal ``matplotlib.axes.Axes`` kwargs plus projection, which chooses a projection type
        for the axes.
    xy_plot_options : dict, optional
        Keyword arguments to use when plotting the data on the x-y data axis. Default is ``None``.
        Valid keys are ``matplotlib.lines.Line2D`` properties, e.g. ``alpha``, ``color``, ``label``, etc.
    colorbar_options : dict, optional
        Keyword arguments to the colorbar axes. Default is ``None``.
        Valid keys are legal ``matplotlib.axes.Axes`` kwargs plus projection, which chooses a projection type
        for the axes.
    point_plot_options : dict, optional
        Keyword arguments to use when plotting the current data point in the x-y data series on the x-y axes.
        If ``None`` is supplied a purple circle is used.

    Methods
    -------
    __call__(i)
        Returns the ``i``th figure.
    __len__()
        Returns the total number of frames to create.
    """

    def __init__(self, image_filenames, dic_filenames, variable, xy_data, figure_width=10.0, fractional_padding=0.5,
                 overlay_contour_options=None, xy_axes_options=None, xy_plot_options=None, colorbar_options=None,
                 point_plot_options=None):
        assert len(dic_filenames) == len(image_filenames), "The number of images ({}) does not match the " \
                                                           "number of DIC files ({}).".format(len(image_filenames),
                                                                                              len(dic_filenames))
        self._image_filenames = image_filenames
        self._dic_filenames = dic_filenames
        self._num_frames = len(image_filenames)
        self._variable = variable
        self._x_data = xy_data[0]
        self._y_data = xy_data[1]

        self._image_height, self._image_width = self._image_resolution(image_filenames[0])
        self._height_padding = int(0.05 * self._image_height)

        colorbar_lim, overlay_lim, xy_lim = self._axes_limits(self._image_width, fractional_padding)
        self._colorbar_lim = colorbar_lim
        self._overlay_lim = overlay_lim
        self._xy_lim = xy_lim

        self._grid_spec = GridSpec(self._image_height + self._height_padding, xy_lim[-1])
        self._fig_width_in_inches = figure_width
        self._fig_height_in_inches = self._image_height / xy_lim[-1] * self._fig_width_in_inches

        self._init_options(colorbar_options, overlay_contour_options, point_plot_options, xy_axes_options,
                           xy_plot_options)

    def _init_options(self, colorbar_options, overlay_contour_options, point_plot_options, xy_axes_options,
                      xy_plot_options):
        if overlay_contour_options is None:
            overlay_contour_options = {}
        if "cmap" in overlay_contour_options:
            self._cmap = overlay_contour_options["cmap"]
        else:
            self._cmap = "viridis"
            overlay_contour_options["cmap"] = self._cmap
        if "levels" in overlay_contour_options:
            self._levels = overlay_contour_options.pop("levels")
        else:
            self._levels = None
        if overlay_contour_options is None:
            self._overlay_contour_options = {}
        else:
            self._overlay_contour_options = overlay_contour_options
        if xy_axes_options is None:
            self._xy_axes_options = {}
        else:
            self._xy_axes_options = xy_axes_options
        self._xy_plot_options = xy_plot_options
        self._colorbar_options = colorbar_options

        if "title" in self._colorbar_options:
            self._colorbar_title = self._colorbar_options.pop("title")
        else:
            self._colorbar_title = self._variable

        if point_plot_options is None:
            self._point_plot_options = {}
        else:
            self._point_plot_options = point_plot_options
        if "marker" not in self._point_plot_options:
            self._point_plot_options["marker"] = "o"
        if "color" not in self._point_plot_options:
            self._point_plot_options["color"] = "#9C27B0"

    def __call__(self, i):
        if plt.get_backend().lower() == "agg":
            fig = Figure(figsize=(self._fig_width_in_inches, self._fig_height_in_inches))
        else:
            fig = plt.figure(figsize=(self._fig_width_in_inches, self._fig_height_in_inches))

        overlay_ax = fig.add_subplot(
            self._grid_spec[:self._image_height - self._height_padding, self._overlay_lim[0]:self._overlay_lim[1]])

        cbar_ax = fig.add_subplot(
            self._grid_spec[:self._image_height - self._height_padding, self._colorbar_lim[0]:self._colorbar_lim[1]])
        cbar_ax.set_title(self._colorbar_title, y=1.04)

        xy_ax = fig.add_subplot(
            self._grid_spec[:self._image_height - self._height_padding, self._xy_lim[0]:self._xy_lim[1]],
            **self._xy_axes_options)

        image_filename = self._image_filenames[i]
        dic_filename = self._dic_filenames[i]

        vmax, vmin = self._set_contour_levels(dic_filename)

        plot_overlay(image_filename, load_dic_data(dic_filename), self._variable, ax=overlay_ax, contour_options=self._overlay_contour_options)
        plot_colorbar(self._cmap, vmin, vmax, ax=cbar_ax, colorbar_options=self._colorbar_options)
        plot_xy(self._x_data, self._y_data, ax=xy_ax, axes_options=self._xy_axes_options, plot_options=self._xy_plot_options)
        plot_xy([self._x_data[i]], [self._y_data[i]], ax=xy_ax, plot_options=self._point_plot_options)
        return fig

    def _set_contour_levels(self, dic_filename):
        if self._levels is None:
            variable_names = [self._variable]
            if self._variable != "sigma":
                variable_names.append("sigma")
            dic_data = load_dic_data(dic_filename, variable_names=variable_names)
            vmin = np.min(dic_data[self._variable])
            vmax = np.max(dic_data[self._variable])
            if vmax > vmin:
                self._overlay_contour_options["levels"] = np.linspace(vmin, vmax, 10)
            else:
                self._overlay_contour_options["levels"] = None
        else:
            vmin = np.min(self._levels)
            vmax = np.max(self._levels)
            self._overlay_contour_options["levels"] = self._levels
        return vmax, vmin

    def __len__(self):
        return self._num_frames



    @staticmethod
    def _image_resolution(image_filename):
        """
        Retrieves the resolution of the specified image.

        Parameters
        ----------
        image_filename : str
            Name of image to load.

        Returns
        -------
        (height, width) : (int, int)
            Image resolution in pixels.
        """
        img = mpimg.imread(image_filename)
        return img.shape

    @staticmethod
    def _axes_limits(image_width, fractional_padding=0.5):
        """
        Calculates the widths of each axes object in the frame.

        Parameters
        ----------
        image_width : int
            Width of the overlay background image
        fractional_padding : float, optional
        Fraction of the image width to pad between the right of the colorbar and the left of the x-y data plot.
        Default is ``0.5``

        Returns
        -------
        colorbar_lim, overlay_lim, xy_lim : List[(int, int)]
            Limits for each element.
        """
        # calculate widths and padding for each item
        overlay_width = image_width
        colorbar_width = int(0.05 * image_width)
        xy_width = image_width
        overlay_colorbar_padding_width = int(0.05 * image_width)
        colorbar_xy_padding_width = int(fractional_padding * image_width)

        # set limits based on item sizes
        left_lim = 0
        right_lim = overlay_width
        overlay_lim = (left_lim, right_lim)

        left_lim = right_lim + overlay_colorbar_padding_width
        right_lim = left_lim + colorbar_width
        colorbar_lim = (left_lim, right_lim)

        left_lim = right_lim + colorbar_xy_padding_width
        right_lim = left_lim + xy_width
        xy_lim = (left_lim, right_lim)

        return colorbar_lim, overlay_lim, xy_lim
