#collected from: https://github.com/MDAnalysis/mdanalysis/blob/75706b44a6988748d063b6f7a9d86faf04498893/package/MDAnalysis/analysis/psa.py#L331
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import warnings
import numbers
def sqnorm(v, axis=None):
    """Compute the sum of squares of elements along specified axes.
    Parameters
    ----------
    v :  numpy.ndarray
         coordinates
    axes : None / int / tuple (optional)
         Axes or axes along which a sum is performed. The default
         (*axes* = ``None``) performs a sum over all the dimensions of
         the input array.  The value of *axes* may be negative, in
         which case it counts from the last axis to the zeroth axis.
    Returns
    -------
    float
          the sum of the squares of the elements of `v` along `axes`
    """
    return np.sum(v*v, axis=axis)

def get_msd_matrix(P, Q, axis=None):
    r"""Generate the matrix of pairwise mean-squared deviations between paths.
    The MSDs between all pairs of points in `P` and `Q` are
    calculated, each pair having a point from `P` and a point from
    `Q`.
    `P` (`Q`) is a :class:`numpy.ndarray` of :math:`N_p` (:math:`N_q`) time
    steps, :math:`N` atoms, and :math:`3N` coordinates (e.g.,
    :attr:`MDAnalysis.core.groups.AtomGroup.positions`). The pairwise MSD
    matrix has dimensions :math:`N_p` by :math:`N_q`.
    Parameters
    ----------
    P : numpy.ndarray
        the points in the first path
    Q : numpy.ndarray
        the points in the second path
    Returns
    -------
    msd_matrix : numpy.ndarray
         matrix of pairwise MSDs between points in `P` and points
         in `Q`
    Notes
    -----
    We calculate the MSD matrix
    .. math::
       M_{ij} = ||p_i - q_j||^2
    where :math:`p_i \in P` and :math:`q_j \in Q`.
    """
    return np.asarray([sqnorm(p - Q, axis=axis) for p in P])


def reshaper(path, axis):
    """Flatten path when appropriate to facilitate calculations
    requiring two dimensional input.
    """
    if len(axis) > 1:
        path = path.reshape(len(path), -1)
    return path

def get_coord_axes(path):
    """Return the number of atoms and the axes corresponding to atoms
    and coordinates for a given path.
    The `path` is assumed to be a :class:`numpy.ndarray` where the 0th axis
    corresponds to a frame (a snapshot of coordinates). The :math:`3N`
    (Cartesian) coordinates are assumed to be either:
    1. all in the 1st axis, starting with the x,y,z coordinates of the
       first atom, followed by the *x*,*y*,*z* coordinates of the 2nd, etc.
    2. in the 1st *and* 2nd axis, where the 1st axis indexes the atom
       number and the 2nd axis contains the *x*,*y*,*z* coordinates of
       each atom.
    Parameters
    ----------
    path : numpy.ndarray
         representing a path
    Returns
    -------
    (int, (int, ...))
         the number of atoms and the axes containing coordinates
    """
    path_dimensions = len(path.shape)
    if path_dimensions == 3:
        N = path.shape[1]
        axis = (1,2) # 1st axis: atoms, 2nd axis: x,y,z coords
    elif path_dimensions == 2:
        # can use mod to check if total # coords divisible by 3
        N = path.shape[1] / 3
        axis = (1,) # 1st axis: 3N structural coords (x1,y1,z1,...,xN,xN,zN)
    else:
        raise ValueError("Path must have 2 or 3 dimensions; the first "
                         "dimensions (axis 0) must correspond to frames, "
                         "axis 1 (and axis 2, if present) must contain atomic "
                         "coordinates.")
    return N, axis


def hausdorff(P, Q):
    r"""Calculate the symmetric Hausdorff distance between two paths.
    The metric used is RMSD, as opposed to the more conventional L2
    (Euclidean) norm, because this is convenient for i.e., comparing
    protein configurations.
    *P* (*Q*) is a :class:`numpy.ndarray` of :math:`N_p` (:math:`N_q`) time
    steps, :math:`N` atoms, and :math:`3N` coordinates (e.g.,
    :attr:`MDAnalysis.core.groups.AtomGroup.positions`). *P* (*Q*) has
    either shape |3Dp| (|3Dq|), or |2Dp| (|2Dq|) in flattened form.
    Note that reversing the path does not change the Hausdorff distance.
    Parameters
    ----------
    P : numpy.ndarray
        the points in the first path
    Q : numpy.ndarray
        the points in the second path
    Returns
    -------
    float
        the Hausdorff distance between paths `P` and `Q`
    Example
    -------
    Calculate the Hausdorff distance between two halves of a trajectory:
     #>>> from MDAnalysis.tests.datafiles import PSF, DCD
     #>>> u = Universe(PSF,DCD)
     #>>> mid = len(u.trajectory)/2
     #>>> ca = u.select_atoms('name CA')
     #>>> P = numpy.array([
     ...                ca.positions for _ in u.trajectory[:mid:]
     ...              ]) # first half of trajectory
     #>>> Q = numpy.array([
     ...                ca.positions for _ in u.trajectory[mid::]
     ...              ]) # second half of trajectory
     #>>> hausdorff(P,Q)
     4.7786639840135905
     #>>> hausdorff(P,Q[::-1]) # hausdorff distance w/ reversed 2nd trajectory
     4.7786639840135905
    Notes
    -----
    :func:`scipy.spatial.distance.directed_hausdorff` is an optimized
    implementation of the early break algorithm of [Taha2015]_; the
    latter code is used here to calculate the symmetric Hausdorff
    distance with an RMSD metric
    References
    ----------
    .. [Taha2015] A. A. Taha and A. Hanbury. An efficient algorithm for
       calculating the exact Hausdorff distance. IEEE Transactions On Pattern
       Analysis And Machine Intelligence, 37:2153-63, 2015.
    """
    N_p, axis_p = get_coord_axes(P)
    N_q, axis_q = get_coord_axes(Q)

    if N_p != N_q:
        raise ValueError("P and Q must have matching sizes")

    P = reshaper(P, axis_p)
    Q = reshaper(Q, axis_q)

    return max(directed_hausdorff(P, Q)[0],
               directed_hausdorff(Q, P)[0]) / np.sqrt(N_p)


def hausdorff_wavg(P, Q):
    r"""Calculate the weighted average Hausdorff distance between two paths.
    *P* (*Q*) is a :class:`numpy.ndarray` of :math:`N_p` (:math:`N_q`) time
    steps, :math:`N` atoms, and :math:`3N` coordinates (e.g.,
    :attr:`MDAnalysis.core.groups.AtomGroup.positions`). *P* (*Q*) has
    either shape |3Dp| (|3Dq|), or |2Dp| (|2Dq|) in flattened form. The nearest
    neighbor distances for *P* (to *Q*) and those of *Q* (to *P*) are averaged
    individually to get the average nearest neighbor distance for *P* and
    likewise for *Q*. These averages are then summed and divided by 2 to get a
    measure that gives equal weight to *P* and *Q*.
    Parameters
    ----------
    P : numpy.ndarray
        the points in the first path
    Q : numpy.ndarray
        the points in the second path
    Returns
    -------
    float
        the weighted average Hausdorff distance between paths `P` and `Q`
    Example
    -------
     #>>> from MDAnalysis import Universe
     #>>> from MDAnalysis.tests.datafiles import PSF, DCD
     #>>> u = Universe(PSF,DCD)
     #>>> mid = len(u.trajectory)/2
     #>>> ca = u.select_atoms('name CA')
     #>>> P = numpy.array([
     ...                ca.positions for _ in u.trajectory[:mid:]
     ...              ]) # first half of trajectory
     #>>> Q = numpy.array([
     ...                ca.positions for _ in u.trajectory[mid::]
     ...              ]) # second half of trajectory
     #>>> hausdorff_wavg(P,Q)
     2.5669644353703447
     #>>> hausdorff_wavg(P,Q[::-1]) # weighted avg hausdorff dist w/ Q reversed
     2.5669644353703447
    Notes
    -----
    The weighted average Hausdorff distance is not a true metric (it does not
    obey the triangle inequality); see [Seyler2015]_ for further details.
    """
    N, axis = get_coord_axes(P)
    d = get_msd_matrix(P, Q, axis=axis)
    out = 0.5*( np.mean(np.amin(d,axis=0)) + np.mean(np.amin(d,axis=1)) )
    return ( out / N )**0.5


def hausdorff_avg(P, Q):
    r"""Calculate the average Hausdorff distance between two paths.
    *P* (*Q*) is a :class:`numpy.ndarray` of :math:`N_p` (:math:`N_q`) time
    steps, :math:`N` atoms, and :math:`3N` coordinates (e.g.,
    :attr:`MDAnalysis.core.groups.AtomGroup.positions`). *P* (*Q*) has
    either shape |3Dp| (|3Dq|), or |2Dp| (|2Dq|) in flattened form. The nearest
    neighbor distances for *P* (to *Q*) and those of *Q* (to *P*) are all
    averaged together to get a mean nearest neighbor distance. This measure
    biases the average toward the path that has more snapshots, whereas weighted
    average Hausdorff gives equal weight to both paths.
    Parameters
    ----------
    P : numpy.ndarray
        the points in the first path
    Q : numpy.ndarray
        the points in the second path
    Returns
    -------
    float
        the average Hausdorff distance between paths `P` and `Q`
    Example
    -------
     #>>> from MDAnalysis.tests.datafiles import PSF, DCD
     #>>> u = Universe(PSF,DCD)
     #>>> mid = len(u.trajectory)/2
     #>>> ca = u.select_atoms('name CA')
     #>>> P = numpy.array([
     ...                ca.positions for _ in u.trajectory[:mid:]
     ...              ]) # first half of trajectory
     #>>> Q = numpy.array([
     ...                ca.positions for _ in u.trajectory[mid::]
     ...              ]) # second half of trajectory
     #>>> hausdorff_avg(P,Q)
     2.5669646575869005
     #>>> hausdorff_avg(P,Q[::-1]) # hausdorff distance w/ reversed 2nd trajectory
     2.5669646575869005
    Notes
    -----
    The average Hausdorff distance is not a true metric (it does not obey the
    triangle inequality); see [Seyler2015]_ for further details.
    """
    N, axis = get_coord_axes(P)
    d = get_msd_matrix(P, Q, axis=axis)
    out = np.mean( np.append( np.amin(d,axis=0), np.amin(d,axis=1) ) )
    return ( out / N )**0.5


def hausdorff_neighbors(P, Q):
    r"""Find the Hausdorff neighbors of two paths.
    *P* (*Q*) is a :class:`numpy.ndarray` of :math:`N_p` (:math:`N_q`) time
    steps, :math:`N` atoms, and :math:`3N` coordinates (e.g.,
    :attr:`MDAnalysis.core.groups.AtomGroup.positions`). *P* (*Q*) has
    either shape |3Dp| (|3Dq|), or |2Dp| (|2Dq|) in flattened form.
    Parameters
    ----------
    P : numpy.ndarray
        the points in the first path
    Q : numpy.ndarray
        the points in the second path
    Returns
    -------
    dict
        dictionary of two pairs of numpy arrays, the first pair (key
        "frames") containing the indices of (Hausdorff) nearest
        neighbors for `P` and `Q`, respectively, the second (key
        "distances") containing (corresponding) nearest neighbor
        distances for `P` and `Q`, respectively
    Notes
    -----
    - Hausdorff neighbors are those points on the two paths that are separated by
      the Hausdorff distance. They are the farthest nearest neighbors and are
      maximally different in the sense of the Hausdorff distance [Seyler2015]_.
    - :func:`scipy.spatial.distance.directed_hausdorff` can also provide the
      hausdorff neighbors.
    """
    N, axis = get_coord_axes(P)
    d = get_msd_matrix(P, Q, axis=axis)
    nearest_neighbors = {
        'frames' : (np.argmin(d, axis=1), np.argmin(d, axis=0)),
        'distances' : ((np.amin(d,axis=1)/N)**0.5, (np.amin(d, axis=0)/N)**0.5)
    }
    return nearest_neighbors

def discrete_frechet(P, Q):
    r"""Calculate the discrete Fréchet distance between two paths.
    *P* (*Q*) is a :class:`numpy.ndarray` of :math:`N_p` (:math:`N_q`) time
    steps, :math:`N` atoms, and :math:`3N` coordinates (e.g.,
    :attr:`MDAnalysis.core.groups.AtomGroup.positions`). *P* (*Q*) has
    either shape |3Dp| (|3Dq|), or :|2Dp| (|2Dq|) in flattened form.
    Parameters
    ----------
    P : numpy.ndarray
        the points in the first path
    Q : numpy.ndarray
        the points in the second path
    Returns
    -------
    float
        the discrete Fréchet distance between paths *P* and *Q*
    Example
    -------
    Calculate the discrete Fréchet distance between two halves of a
    trajectory.
     #>>> u = Universe(PSF,DCD)
     #>>> mid = len(u.trajectory)/2
     #>>> ca = u.select_atoms('name CA')
     #>>> P = np.array([
     ...                ca.positions for _ in u.trajectory[:mid:]
     ...              ]) # first half of trajectory
     #>>> Q = np.array([
     ...                ca.positions for _ in u.trajectory[mid::]
     ...              ]) # second half of trajectory
     #>>> discrete_frechet(P,Q)
     4.7786639840135905
     #>>> discrete_frechet(P,Q[::-1]) # frechet distance w/ 2nd trj reversed 2nd
     6.8429011177113832
    Note that reversing the direction increased the Fréchet distance:
    it is sensitive to the direction of the path.
    Notes
    -----
    The discrete Fréchet metric is an approximation to the continuous Fréchet
    metric [Frechet1906]_ [Alt1995]_. The calculation of the continuous
    Fréchet distance is implemented with the dynamic programming algorithm of
    [EiterMannila1994]_ [EiterMannila1997]_.
    References
    ----------
    .. [Frechet1906] M. Fréchet. Sur quelques points du calcul
       fonctionnel. Rend. Circ. Mat. Palermo, 22(1):1–72, Dec. 1906.
    .. [Alt1995] H. Alt and M. Godau. Computing the Fréchet distance between
       two polygonal curves. Int J Comput Geometry & Applications,
       5(01n02):75–91, 1995. doi: `10.1142/S0218195995000064`_
    .. _`10.1142/S0218195995000064`: http://doi.org/10.1142/S0218195995000064
    .. [EiterMannila1994] T. Eiter and H. Mannila. Computing discrete Fréchet
       distance. Technical Report CD-TR 94/64, Christian Doppler Laboratory for
       Expert Systems, Technische Universität Wien, Wien, 1994.
    .. [EiterMannila1997] T. Eiter and H. Mannila. Distance measures for point
       sets and their computation. Acta Informatica, 34:109–133, 1997. doi: `10.1007/s002360050075`_.
    .. _10.1007/s002360050075: http://doi.org/10.1007/s002360050075
    """

    N, axis = get_coord_axes(P)
    Np, Nq = len(P), len(Q)
    d = get_msd_matrix(P, Q, axis=axis)
    ca = -np.ones((Np, Nq))

    def c(i, j):
        """Compute the coupling distance for two partial paths formed by *P* and
        *Q*, where both begin at frame 0 and end (inclusive) at the respective
        frame indices :math:`i-1` and :math:`j-1`. The partial path of *P* (*Q*)
        up to frame *i* (*j*) is formed by the slicing ``P[0:i]`` (``Q[0:j]``).
        :func:`c` is called recursively to compute the coupling distance
        between the two full paths *P* and *Q*  (i.e., the discrete Frechet
        distance) in terms of coupling distances between their partial paths.
        Parameters
        ----------
        i : int
            partial path of *P* through final frame *i-1*
        j : int
            partial path of *Q* through final frame *j-1*
        Returns
        -------
        dist : float
            the coupling distance between partial paths `P[0:i]` and `Q[0:j]`
        """
        if ca[i,j] != -1 :
            return ca[i,j]
        if i > 0:
            if j > 0:
                ca[i,j] = max( min(c(i-1,j),c(i,j-1),c(i-1,j-1)), d[i,j] )
            else:
                ca[i,j] = max( c(i-1,0), d[i,0] )
        elif j > 0:
            ca[i,j] = max( c(0,j-1), d[0,j] )
        else:
            ca[i,j] = d[0,0]
        return ca[i,j]

    return (c(Np-1, Nq-1) / N)**0.5


def dist_mat_to_vec(N, i, j):
    """Convert distance matrix indices (in the upper triangle) to the index of
    the corresponding distance vector.
    This is a convenience function to locate distance matrix elements (and the
    pair generating it) in the corresponding distance vector. The row index *j*
    should be greater than *i+1*, corresponding to the upper triangle of the
    distance matrix.
    Parameters
    ----------
    N : int
        size of the distance matrix (of shape *N*-by-*N*)
    i : int
        row index (starting at 0) of the distance matrix
    j : int
        column index (starting at 0) of the distance matrix
    Returns
    -------
    int
        index (of the matrix element) in the corresponding distance vector
    """

    if not (isinstance(N, numbers.Integral) and isinstance(i, numbers.Integral)
            and isinstance(j, numbers.Integral)):
        raise ValueError("N, i, j all must be of type int")

    if i < 0 or j < 0 or N < 2:
        raise ValueError("Matrix indices are invalid; i and j must be greater "
                         "than 0 and N must be greater the 2")

    if (j > i and (i > N - 1 or j > N)) or (j < i and (i > N or j > N - 1)):
        raise ValueError("Matrix indices are out of range; i and j must be "
                         "less than N = {0:d}".format(N))
    if j > i:
        return (N*i) + j - (i+2)*(i+1) // 2  # old-style division for int output
    elif j < i:
        warnings.warn("Column index entered (j = {:d} is smaller than row "
                      "index (i = {:d}). Using symmetric element in upper "
                      "triangle of distance matrix instead: i --> j, "
                      "j --> i".format(j, i))
        return (N*j) + i - (j+2)*(j+1) // 2  # old-style division for int output
    else:
        raise ValueError("Error in processing matrix indices; i and j must "
                         "be integers less than integer N = {0:d} such that"
                         " j >= i+1.".format(N))
