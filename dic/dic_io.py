from __future__ import absolute_import, division, print_function
import itertools
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import scipy.io as spio
import sys
import warnings
from .tqdm import tqdm

__all__ = ["load_dic_data", "load_csv_data", "get_filenames", "get_image_filenames", "update_dic_data"]


def get_filenames(directory, extension, prepend_directory=False):
    """
    Retrieves a sorted list of file names in the specified directory with the given extension.

    Parameters
    ----------
    directory : str
        Directory to search.
    extension : str
        Extension to search for (including the period, if applicable).
    prepend_directory : bool, optional
        Whether to prepend the directory to the filenames. Default = ``False``.

    Returns
    -------
    List[str]
        Sorted list of file names that match the given extension
    """
    output = []
    for f in os.listdir(directory):
        fbase, fext = os.path.splitext(f)
        if fext == extension:
            output.append(f)
    output.sort()

    if prepend_directory:
        for i, f in enumerate(output):
            output[i] = os.path.join(directory, f)

    return output


def get_image_filenames(directory, extension=".tif", prepend_directory=False):
    """
    Sorts and retrieves the camera images in the specified directory. This function assumes that the left camera images
    end in ``_0.ext``, where ``.ext`` is the provided extension. Similarly, the right camera images are assumed to
    end in ``_1.ext``.

    Parameters
    ----------
    directory : str
        Directory to search.
    extension : str, optional
        Image extension. Default = ``".tif"``.
    prepend_directory : bool, optional
        Whether to prepend the directory to the filenames. Default = ``False``.

    Returns
    -------
    (left_camera_filenames, right_camera_filenames) : (List[str], List[str])
        List of filenames belonging to each camera.
    """
    filenames = get_filenames(directory, extension, prepend_directory=prepend_directory)
    left_camera_filenames = []
    right_camera_filenames = []

    left_camera_required_text = "_0{}".format(extension)
    right_camera_required_text = "_1{}".format(extension)

    for f in filenames:
        if f[-len(left_camera_required_text):] == left_camera_required_text:
            left_camera_filenames.append(f)
        elif f[-len(right_camera_required_text):] == right_camera_required_text:
            right_camera_filenames.append(f)
        else:
            warnings.warn("Unable to categorize file {} as belonging to the left or right camera.".format(f))

    return left_camera_filenames, right_camera_filenames


def _mask_bad_data(dic_data, mask_key="sigma", bad_value=-1.0):
    """
    Masks (i.e. removes) the uncorrelated data for each key in the dictionary. The positions of bad values are
    determined by the entries of ``dic_data[mask_key]`` that are equal to ``bad_value``. The same mask is
    applied to all keys in the dictionary that reference correlated data. The only keys not updated are ``x`` and ``y``
    --those correspond to the pixel positions and are not dependent on whether the correlation was successful--
    as well as double underscore keys, e.g. ``__key__``. This assumes that all correlated values in the
    dictionary have the same shape. The dictionary is updated in-place. The keys remain the same, but the values are
    masked after the function call.

    Parameters
    ----------
    dic_data : dict
        Dictionary of ``{key: value}`` pairs.
    mask_key : str
        Key in ``dic_data`` to search for ``bad_value``s
    bad_value : float
        Value that corresponds to uncorrelated data for the specified ``mask_key``.
    """
    bad_mask = np.isclose(dic_data[mask_key], bad_value)
    unmasked_keys = ["x", "y"]
    for key, value in iter(dic_data.items()):
        if key[:2] != '__' and key not in unmasked_keys:
            if key == mask_key:
                fill_value = bad_value
            else:
                fill_value = None
            dic_data[key] = np.ma.masked_array(value, mask=bad_mask, fill_value=fill_value)


def _unmask_bad_data(dic_data):
    """
    Unmasks the data for each key in the dictionary by replacing all ``numpy.MaskedArray``s with the their unmasked data.
    The keys remain the same, but the all ``numpy.MaskedArray`` instances are converted to ``numpy.ndarray``.

    Parameters
    ----------
    dic_data : dict
        Dictionary of ``{key: value}`` pairs.
    """

    for key, value in iter(dic_data.items()):
        if isinstance(value, np.ma.MaskedArray):
            dic_data[key] = np.ma.filled(value)


def load_dic_data(filename, variable_names=None):
    """
    Loads the DIC data specified by the given filename. Uncorrelated regions of the dataset are masked (i.e. removed)
    before the data is returned. Prior to loading the data must be exported into the Matlab (``.mat``) file format.

    Parameters
    ----------
    filename : str
        Name of the DIC data file to load.
    variable_names : None or sequence
        If ``None`` (the default) - read all variables in file. Otherwise variable_names should be a sequence of
        strings, giving names of the matlab variables to read from the file. The reader will skip any variable
        with a name not in this sequence, possibly saving some read processing.

    Returns
    -------
    dict
        A dictionary of ``{key: value}`` pairs where each key is the variable name,
        e.g. ``X``, ``U``, ``Z``, etc., and the value is a 2D numpy array containing the
        exported DIC results.
    """
    dic_data = spio.loadmat(filename, variable_names=variable_names)
    _mask_bad_data(dic_data)
    return dic_data


def _get_csv_header_names(filename):
    """
    Returns the column headers for the given filename.

    Parameters
    ----------
    filename : str
        Name of the ``.csv`` file to search.

    Returns
    -------
    List[str] or None
        List of header names or ``None`` is no header was found.

    Raises
    ------
    RuntimeError
        If the first entry of the first row is not 0 or Count.
    """
    with open(filename, "r") as csvfile:
        first_line = csvfile.readline()
        names = first_line.strip("\n").split(",")
        if names[0] == '0':
            return None
        elif names[0] == "Count":
            return names
        else:
            raise RuntimeError("Could not determine header.")


def load_csv_data(filename, column, scale=None):
    """
    Loads the specifed column of the csv data from the given filename. Values are multiplied by ``scale`` before the
    data is returned.

    Parameters
    ----------
    filename : str
        Name of the ``.csv`` file to load.
    column : int
        Column index to load from the file.
    scale : float, optional
        Value to scale the data by before returning. For example, when loading MTS data the scale variable
        can be used to change the native output (Volts) to force (N) by providing ``scale`` that is equal to
        the Newtons/Volts. Default is ``None``.

    Returns
    -------
    ``numpy.ndarray``
        1D array of data values.
    """
    header_names = _get_csv_header_names(filename)
    if header_names is not None:
        header_row = 0
    else:
        header_row = None

    csv_dataframe = pd.read_csv(filename, delimiter=",", header=header_row, names=header_names)
    csv_values = csv_dataframe.ix[:,column].values
    if scale is not None:
        csv_values *= scale
    return csv_values


def _update_dic_data_worker(worker_args):
    """
    Processes a single DIC file and saves the output.

    Parameters
    ----------
    worker_args : (str, callable, tuple, str, str, bool)
        Arguments needed to process a single DIC file,
        namely ``(dic_filename, function, args, input_directory, output_directory, compress)``
    """
    dic_filename, function, args, input_directory, output_directory, compress = worker_args
    dic_data = load_dic_data(os.path.join(input_directory, dic_filename))
    if function is not None:
        function(dic_data, *args)
    _unmask_bad_data(dic_data)
    output_filename = os.path.join(output_directory, dic_filename)
    spio.savemat(output_filename, dic_data, do_compression=compress)


def update_dic_data(input_directory, output_directory, function=None, args=(), compress=False, processes=None):
    """
    Calls ``function`` on each DIC file and saves the new version into the output directory.

    Parameters
    ----------
    input_directory : str
        Path to uncompressed ``.mat`` files.
    output_directory : str
        Where to save the compressed files.
    function : callable, optional
        Function that accepts a ``dict`` and modifies the object in place. Default is ``None``, i.e. no function
        is called.
    args : tuple, optional
        Extra arguments passed to function, i.e. ``f(dic_data, *args)``.
    output_directory : str
        Directory to store the results.
    compress : bool, optional
        Whether to compress the DIC files when saving to the output directory.
    processes : int, optional
        The number of processes to use when converting the files. Default is ``None``.
        If ``None`` is provided then the number of processes will be set to the value
        returned by ``multiprocessing.cpu_count()``.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if not isinstance(args, tuple):
        args = (args,)

    dic_filenames = get_filenames(input_directory, ".mat")
    num_dic_filenames = len(dic_filenames)

    worker_args = zip(
            dic_filenames,
            itertools.repeat(function, num_dic_filenames),
            itertools.repeat(args, num_dic_filenames),
            itertools.repeat(input_directory, num_dic_filenames),
            itertools.repeat(output_directory, num_dic_filenames),
            itertools.repeat(compress, num_dic_filenames)
        )

    if processes is None:
        processes = mp.cpu_count()

    pool = mp.Pool(processes=processes)

    for _ in tqdm(pool.imap_unordered(_update_dic_data_worker, worker_args),
                  total=num_dic_filenames, file=sys.stdout, desc="Processing DIC files"):
        pass
    pool.close()
    pool.join()

    print("DIC files updated.")
