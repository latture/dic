from __future__ import absolute_import, division, print_function
from math import acos
import numpy as np
from .dic_utils import point_to_indices, point_to_position, get_initial_position, get_displacement
from .geometry_utils import apply_rotation, apply_transformation, distance_between, \
                            distance_to, get_angle, get_transformation_matrix, norm, cross3d

__all__ = ["add_transverse_displacement"]


def _get_node_mask(dic_data, nodes, radius):
    """
    Gets the mask to be applied to the correlated data (represented as masked arrays)
    from the combination of the current mask and the removed data near the nodal locations.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data (usually loaded from a ``.mat`` file).
    nodes : List[List[int, int]]
        Two-dimensional list of ``(x, y)`` nodal location. Units should be in pixels
        and given with respect to the global pixel coordinate, i.e. the pixel location
        in the reference camera image.
    radius : int
        Radius of the nodes in pixels.

    Returns
    -------
    ``numpy.ndarray(dtype=bool)``
        Updated mask.
    """
    old_mask = dic_data["sigma"].mask
    new_mask = np.copy(old_mask)
    global_x_indices = np.array(dic_data["x"][0], dtype=np.int)
    global_y_indices = np.array(dic_data["y"][:, 0], dtype=np.int)[:, np.newaxis]

    for node in nodes:
        x_px, y_px = node
        local_x_indices = global_x_indices - x_px
        local_y_indices = global_y_indices - y_px
        current_node_mask = local_x_indices**2 + local_y_indices**2 <= radius**2
        new_mask = np.ma.mask_or(new_mask, current_node_mask)

    return new_mask


def _get_elem_lut(dic_data, nodes, elems, strut_radius):
    """
    Associates regions of the correlated dataset with the appropriate strut direction.
    Each ``(i, j)`` index in the returned lookup table corresponds to an ``int`` that
    represents the element index the data point belongs to. If no element is associated
    with a data point, the ``(i, j)`` index that data point will be masked.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    nodes : List[(T, T)]
        List of ``(x, y)`` nodal coordinates in pixel space.
    elems : List[(int, int)]
        List of indices into ``nodes``
    strut_radius : int
        Radius of each strut in pixels.

    Returns
    -------
    ``numpy.ma.MaskedArray(dtype=int)``
        Lookup table mapping an index ``(i, j)`` in the DIC data to the element index.
    """
    elem_lut = np.full_like(dic_data["sigma"].mask, -1, dtype=np.int)
    num_rows, num_cols = elem_lut.shape
    mask = dic_data["sigma"].mask

    # dy is constant because strut radius is fixed for all struts:
    bounding_box_dy = np.arange(-strut_radius, strut_radius + 1, dtype=np.int)

    for i, elem in enumerate(elems):
        nn_1, nn_2 = elem
        node_1 = nodes[nn_2]
        node_2 = nodes[nn_1]

        if node_1[0] > node_2[0]:
            node_1, node_2 = node_2, node_1

        node_1_idx = np.asarray(point_to_indices(dic_data, node_1), dtype=np.int)
        node_2_idx = np.asarray(point_to_indices(dic_data, node_2), dtype=np.int)

        # check that both nodal locations are correlated
        if not mask[node_1_idx[0], node_1_idx[1]] and not mask[node_2_idx[0], node_2_idx[1]]:

            theta = get_angle(node_1_idx, node_2_idx)
            length = distance_between(node_1_idx, node_2_idx)
            offset = node_1_idx

            bounding_box_dx = np.arange(0, length, dtype=np.int)
            for dx in bounding_box_dx:
                for dy in bounding_box_dy:
                    row, col = apply_rotation(dx, dy, theta)
                    row += offset[0]
                    col += offset[1]
                    if row < num_rows and col < num_cols:
                        elem_lut[row, col] = i

    mask = elem_lut < 0
    return np.ma.masked_array(elem_lut, mask=mask)


def _get_elem_positions(dic_data, nodes, elems, add_displacement=True):
    """
    Returns the location (in mm) of each element endpoint.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    nodes : List[(T, T)]
        List of ``(x, y)`` nodal coordinates in pixel space.
    elems : List[(int, int)]
        List of indices into ``nodes``
    add_displacement : bool, optional
        Whether to add deformation to the undeformed position. Default is ``True``.

    Returns
    -------
    ``numpy.ndarray(dtype=float)``
        Array of displaced locations. The shape is ``(len(elems), 2, 3)``.
        The last dimension contains the ``(x, y, z)`` location (in mm) of
        the point.
    """
    nodes = np.asarray(nodes)
    elems = np.asarray(elems, dtype=np.int)
    elem_pos_px = nodes[elems]
    num_elems, nodes_per_elem, point_dimensionality = elem_pos_px.shape
    elem_pos_mm = np.empty((num_elems, nodes_per_elem, 3))

    for i, (node_1, node_2) in enumerate(elem_pos_px):
        elem_pos_mm[i, 0] = point_to_position(dic_data, node_1, add_displacement=add_displacement)
        elem_pos_mm[i, 1] = point_to_position(dic_data, node_2, add_displacement=add_displacement)

    return elem_pos_mm


def _get_elem_transformation_matrix(initial_elem_position, deformed_elem_position):
    """
    Returns the 4x4 transformation matrix that maps a point in the undeformed
    space to the corresponding point in the deformed coordinate space.

    Parameters
    ----------
    initial_elem_position : (pt1, pt2)
        Initial end points of the element. End points should be in the form
        ``((x1, y1, z1), (x2, y2, z2))``.
    deformed_elem_position : (pt1, pt2)
        End points of the deformed element. End points should be in the form
        ``((x1, y1, z1), (x2, y2, z2))``.

    Returns
    -------
    ``numpy.ndarray``
        4x4 transformation matrix.
    """
    pt1_0, pt2_0 = initial_elem_position
    pt1_1, pt2_1 = deformed_elem_position

    delta_0 = pt2_0 - pt1_0
    length_0 = norm(delta_0)

    delta_1 = pt2_1 - pt1_1
    length_1 = norm(delta_1)
    axis = cross3d(delta_0, delta_1)
    if np.allclose(axis, np.array([0., 0., 0.])):
        return np.eye(4, dtype=np.float)
    theta = acos(delta_0.dot(delta_1) / (length_0 * length_1))

    scale = length_1 / length_0
    translation = pt1_1 - pt1_0

    return get_transformation_matrix(axis, theta, translation, scale)


def _get_elem_transformation_matrices(initial_elem_positions, deformed_elem_positions):
    """
    Calculates the transformation matrix that maps from initial to deformed position
    for each element in the list of positions.

    initial_elem_positions : List[(pt1, pt2)]
        List of initial (undeformed) element positions. Positions should be
        in the form ``((pt1, pt2), ...)`` where each ``pt`` is a set of
        ``(x, y, z)`` coordinates of type ``(float, float, float)``.
    """
    num_elems = len(initial_elem_positions)
    assert num_elems == len(deformed_elem_positions), "The number of initial and deformed elements are not equal."
    transformation_matrices = np.empty((num_elems, 4, 4))
    for i, (initial_elem_position, deformed_elem_position) in enumerate(zip(initial_elem_positions, deformed_elem_positions)):
        transformation_matrices[i] = _get_elem_transformation_matrix(initial_elem_position, deformed_elem_position)
    return transformation_matrices


def add_transverse_displacement(dic_data, nodes, elems, strut_radius):
    """
    Adds the ``transverse_displacement`` key to the DIC data. Transverse displacement is
    defined as displacement perpendicular to the strut axis. Data near the nodal regions
    is removed from the correlated data. Strut axis is determined by the data points that
    are within +/- the strut radius along the vector between nodal locations. DIC data
    is updated in-place.

    Parameters
    ----------
    dic_data : dict
        Dictionary containing the DIC data.
    nodes : List[(T, T)]
        List of ``(x, y)`` nodal coordinates in pixel space.
    elems : List[(int, int)]
        List of indices into ``nodes``
    strut_radius : int
        Radius of each strut in pixels.

    Returns
    -------
    dic_data : dict
        DIC data with transverse displacement added.
    """
    node_mask = _get_node_mask(dic_data, nodes, 4 * strut_radius)
    elem_lut = _get_elem_lut(dic_data, nodes, elems, strut_radius)
    combined_mask = np.ma.mask_or(node_mask, elem_lut.mask)
    initial_elem_pos = _get_elem_positions(dic_data, nodes, elems, add_displacement=False)
    deformed_elem_pos = _get_elem_positions(dic_data, nodes, elems)
    elem_transformation_matrices = _get_elem_transformation_matrices(initial_elem_pos, deformed_elem_pos)

    dic_data["sigma"].mask = combined_mask

    transverse_displacement = np.empty_like(dic_data["sigma"])

    rows, cols = np.where(np.logical_not(combined_mask))
    for i, j in zip(rows, cols):
        elem_idx = elem_lut[i, j]
        m = elem_transformation_matrices[elem_idx]
        offset = initial_elem_pos[elem_idx, 0]
        pt0_initial = get_initial_position(dic_data, i, j)
        pt0_transformed = apply_transformation(pt0_initial, m, offset)

        pt1, pt2 = deformed_elem_pos[elem_idx]
        pt0 = pt0_initial + get_displacement(dic_data, i, j)
        transverse_displacement[i, j] = abs(distance_to(pt0, pt1, pt2) - distance_to(pt0_transformed, pt1, pt2))

    dic_data["transverse_displacement"] = np.ma.masked_array(transverse_displacement, mask=combined_mask)
    return dic_data
