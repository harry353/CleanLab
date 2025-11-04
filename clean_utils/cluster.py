def cluster_to_first_ps(new_y, new_x, existing_coords, radius=2):
    """
    Merge newly fitted point-source coordinates with existing clusters.

    Checks whether the new point-source position `(new_y, new_x)` lies within
    a specified radius of any previously identified source coordinates.
    If so, it returns the coordinates of the first matching cluster to
    maintain positional consistency. Otherwise, the new coordinate is added
    to the list of existing cluster centers.

    Parameters
    ----------
    new_y, new_x : float
        Coordinates (y, x) of the newly detected or fitted point source.
    existing_coords : list of tuple
        List of previously accepted cluster coordinates as (y, x) tuples.
        The list is updated in place if the new coordinate does not match
        an existing cluster.
    radius : float, optional
        Maximum distance threshold for considering two sources as belonging
        to the same cluster (default: 2 pixels).

    Returns
    -------
    cluster_y, cluster_x : float
        Coordinates of the cluster center corresponding to the new source.

    Notes
    -----
    - This function is used in cluster-based CLEAN strategies to prevent
      repeated subtraction of overlapping point sources.
    - The first matching coordinate within the radius is reused, ensuring
      stable clustering behavior across iterations.
    """
    for y, x in existing_coords:
        if (new_y - y) ** 2 + (new_x - x) ** 2 <= radius ** 2:
            return y, x

    existing_coords.append((new_y, new_x))
    return new_y, new_x
