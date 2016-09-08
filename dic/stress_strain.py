from __future__ import absolute_import, division, print_function
import numpy as np
from .extensometer import extensometer_strain, extensometer_sequence

__all__ = ["stress_strain"]


def stress_strain(dic_filenames, force, area, extensometers):
    """
    Calculates the stress-strain response using the force data and the strain data calculated from
    the DIC files in the specified directory. Strain in calculated for each of the given extensometers, and the
    average at each data point is returned.

    Parameters
    ----------
    dic_filenames : List[str]
        List of filenames containing the DIC data.
    force : List[float]
        List of force data points. Number of data points must eq
    area : float
        Area of the sample.
    extensometers : List[:class:`.Extensometer`]
        List of pixel coordinates for each extensometer.

    Returns
    -------
    (stress, strain) : (``numpy.ndarray``, ``numpy.ndarray``)
        Stress-strain data.
    """
    force = np.asarray(force)
    stress = force / area
    assert len(stress) == len(dic_filenames), "The number stress data points ({}) does not match the " \
                                              "number of DIC files ({})".format(len(stress), len(dic_filenames))

    extensometer_strains = extensometer_sequence(dic_filenames, extensometers, metric=extensometer_strain,
                                                 description="Calculating stress-strain")
    average_strain = np.mean(extensometer_strains, axis=-1)
    print("Strain calculation completed.")

    return stress, average_strain
