from __future__ import absolute_import, division, print_function
import itertools
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import subprocess
import sys
from tqdm import tqdm

__all__ = ["export_frames", "image_sequence_to_video"]


def _save_frame_worker(args):
    """
    Saves a single frame in the video sequence.

    Parameters
    ----------
    args: (int, callable, str, dict)
        Tuple containing ``(i, frame_creator, ftemplate, savefig_options)``, i.e. all the data
        required to save the ``i``th frame.
    """
    i, frame_creator, ftemplate, savefig_options = args
    fig = frame_creator(i)
    canvas = FigureCanvas(fig)
    canvas.print_figure(ftemplate.format(i), **savefig_options)
    plt.close(fig)


def export_frames(frame_creator, output_directory=None, output_filename="frame.png", savefig_options=None):
    """
    Exports video frames by calling ``frame_creator(i)`` where ``i`` is the iteration number
    belonging to the range ``[0, len(frame_creator)]``. Therefore, the ``frame_creator`` must be
    callable and accept a single argument ``i``. The ``frame_creator`` must also define the
    ``__len__`` method that returns the number frames that the creator intends to create.

    Parameters
    ----------
    frame_creator : callable
        Object responsible for creating the frame. As input the ``frame_creator`` will be called with the frame number
        ``i`` and is expected to return a figure of type ``matplotlib.figure.Figure``. The object must define
        ``__call__(self, i)`` and ``__len(self)__`` methods.
    output_directory : str, optional
        Directory to place the frames into, If ``None`` is specified the frames will be placed in the current
        working directory.
    output_filename : str, optional
        Filename to save the frames as. The frame number will be appending to the output automatically.
    savefig_options : dict, optional
        Keyword arguments to pass to ``matplotlib.pyplot.savefig``.
    """
    fbase, fext = os.path.splitext(output_filename)
    ftemplate = fbase + "_{:03d}" + fext
    num_frames = len(frame_creator)

    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        ftemplate = os.path.join(output_directory, ftemplate)
    if savefig_options is None:
        savefig_options = {}

    # create a single iterable to send arguments to each process
    args = zip(range(num_frames),
        itertools.repeat(frame_creator, num_frames),
        itertools.repeat(ftemplate, num_frames),
        itertools.repeat(savefig_options, num_frames))

    if plt.get_backend().lower() == "agg":
        num_processes = mp.cpu_count()
    else:
        num_processes = 1

    plt.ioff()
    pool = mp.Pool(processes=num_processes, maxtasksperchild=1)
    for _ in tqdm(pool.imap(_save_frame_worker, args), total=num_frames, file=sys.stdout, desc="Exporting frames"):
        pass
    pool.close()
    pool.join()
    plt.ion()

    print("Frames successfully exported.")


def _scale_to_ffmpeg_arg(scale):
    """
    Returns the FFMPEG command line argument to set the scale to a value where both the width and height are
    divisible by 2.

    Parameters
    ----------
    scale : (int, int) or None
        Video scale in the format ``(width, height)`` in pixels. If ``None`` is specified then the scale is set
        such that the dimensions are divisible by 2 but kept as close to the original resolution as possible.

    Returns
    -------
    str
        FFMPEG command line scale.
    """
    if scale is None:
        return "\"scale=trunc(iw/2)*2:trunc(ih/2)*2\""

    if all(i == -1 for i in scale):
        raise RuntimeError("Both width and height cannot be set to -1. "
                           "Use None if you would like to auto-scale both dimensions.")

    if scale[0] != -1:
        width = "{:d}".format(scale[0])
    else:
        width = "trunc(oh*a/2)*2"
    if scale[1] != -1:
        height = "{:d}".format(scale[1])
    else:
        height = "trunc(ow/a/2)*2"

    return "\"scale={:s}:{:s}\"".format(width, height)


def image_sequence_to_video(input_template, output_filename, crf=23, scale=None):
    """
    Converts an image sequence into a video using FFMPEG.

    Parameters
    ----------
    input_template : str
        The ``-i`` parameter passed to FFMPEG. This should specify the naming convention of the image sequence, e.g. ``frame_%03d.png``.
    output_filename : str
        Name of file to save the movie to, e.g. ``movie.mp4``.
    scale : (int, int), optional
        Video scale in the format ``(width, height)`` in pixels. If you'd like to keep the aspect ratio and scale only
        one dimension, set the value for the direction desired and use the value ``-1`` for the other dimension,
        e.g. ``(720, -1)`` for 720p video resolution. The default is ``None`` which resizes the image such that the width
        and height are divisible by 2 (a requirement of the video encoder) but tries to keep the resolution as
        close to the original image as possible.
    crf : int [0-51], optional
        The range of the quantizer scale is 0-51: where 0 is lossless, 23 is default, and 51 is worst possible.
        A lower value is a higher quality and a subjectively sane range is 18-28. Consider 18 to be visually lossless
        or nearly so: it should look the same or nearly the same as the input but it isn't technically lossless.
        The range is exponential, so increasing the CRF value +6 is roughly half the bitrate while -6 is roughly
        twice the bitrate. General usage is to choose the highest CRF value that still provides an acceptable quality.
        If the output looks good, then try a higher value and if it looks bad then choose a lower value.
        (credit: https://trac.ffmpeg.org/wiki/Encode/H.264)
    """
    scale_arg = _scale_to_ffmpeg_arg(scale)

    args = ["ffmpeg",
            "-i", input_template,
            "-crf", "{:d}".format(crf),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", scale_arg,
            "-y",
            output_filename]
    cmd = " ".join(args)
    print("Converting images into video...")
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    with p.stdout:
        for line in iter(p.stdout.readline, b''):
            sys.stdout.write(line)
            sys.stdout.flush()
    p.wait()
