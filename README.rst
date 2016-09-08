DIC
===
This repository contains the tools needed to process digital image correlation (DIC) files. This includes:

- loading, saving and compressing ``*.MAT`` files
- placing and analyzing extensometers
- plotting (x,y) data and contour overlays
- creating video frames and turning the image sequence into a video

The reference IPython notebook shown `here <https://nbviewer.jupyter.org/github/latture/dic/blob/master/reference_dic_notebook.ipynb>`_
describes how to create a video that matches a sequence of DIC overlays to the corresponding stress-strain curve.
To ensure that all the required packages are available, navigate to the ``dic`` folder in the terminal and execute::

    >>> pip install -r requirements.txt

See the `documentation website <https://latture.github.io/dic/>`_ for details on all the available functions.

Compressing files
-----------------
The ``compress.py`` script is included as a convenient way of compressing ``.MAT`` files (those exported from the VIC
software are not by default). By running this script on the native output from the VIC software, total file size should
reduce anywhere from 4-5x. To compress your exported files navigate to the root DIC folder in the terminal, i.e. the folder
containing this README. Then execute ``python compress.py -i /path/to/input_files/ -o /path/to/output_directory/``
If the output directory does not exist, it will be created.
