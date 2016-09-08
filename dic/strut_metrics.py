from __future__ import absolute_import, division, print_function
from .extensometer import extensometer_angle, extensometer_sequence

__all__ = ["nodal_rotation"]


def nodal_rotation(dic_filenames, extensometers):
    """
    Calculates the rotation of each extensometer for every DIC file.

    Parameters
    ----------
    dic_filenames : List[str]
        List of filenames containing the DIC data.
    extensometers : List[:class:`.Extensometer`]
        List of pixel coordinates for each extensometer.

    Returns
    -------
    ``numpy.ndarray``
        Nodal rotation. Shape of the returned array is ``(len(dic_filenames), len(extensometers))``.
    """

    extensometer_angles = extensometer_sequence(dic_filenames, extensometers, metric=extensometer_angle,
                                                description="Calculating nodal rotation")
    extensometer_rotation = extensometer_angles - extensometer_angles[0]
    print("Nodal rotation calculation completed.")
    return extensometer_rotation


