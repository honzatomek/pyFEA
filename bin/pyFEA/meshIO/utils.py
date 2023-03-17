
import typing
from math import pi, sin, cos, tan, asin, acos, atan2, floor, ceil
import numpy as np
import scipy
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt

import pdb


def pad_with_zeros(*args, dtype=float) -> tuple[np.ndarray]:
    """
    Pads the shorter vectors with zeros so that both have the same length
    """
    maxlen = 0
    vv = []
    for v in args:
        vv.append(np.array(v, dtype=dtype))
        maxlen = len(v) if len(v) > maxlen else maxlen

    for i in range(len(vv)):
        if len(vv[i]) < maxlen:
            vv[i] = np.hstack((vv[i], np.zeros(maxlen - len(vv[i]))))

    return vv



def distance(point1: list | np.ndarray,
             point2: list | np.ndarray) -> float:
    """
    Returns the distance between two n-dimensional points.
    The shorter coordinate vector gets padded with zeros.
    """
    v1, v2 = pad_with_zeros(point1, point2)
    return np.linalg.norm(v2 - v1)



def angle2v(vector1: list | np.ndarray,
            vector2: list | np.ndarray, out: str = "radians") -> float:
    """
    Returns the angle between two n-dimensional vectors.
    """
    v1, v2 = pad_with_zeros(vector1, vector2)
    angle = acos(max(min(np.dot(v1, v2) /
                         (np.linalg.norm(v1) * np.linalg.norm(v2)), 1), -1))
    if out == "degrees":
        return degrees(angle)
    else:
        return angle



def angle3p(point1: list | np.ndarray,
            point2: list | np.ndarray,
            point3: list | np.ndarray, out: str = "radians") -> float:
    """
    Returns the angle between three n-dimensional points with the angle being
    at the 1st point.
    """
    p1, p2, p3 = pad_with_zeros(point1, point2, point3)
    v1 = p2 - p1
    v2 = p3 - p1
    angle = acos(max(min(np.dot(v1, v2) /
                         (np.linalg.norm(v1) * np.linalg.norm(v2)), 1), -1))
    if out == "degrees":
        return degrees(angle)
    else:
        return angle



def normal2v(vector1: list | np.ndarray,
             vector2: list | np.ndarray, norm: bool = False) -> np.ndarray:
    """
    Returns a vector normal to the supplied vectors using right-hand rule.
    """
    v1, v2 = pad_with_zeros(vector1, vector2)
    if norm:
        return unit(np.cross(v1, v2))
    else:
        return np.cross(v1, v2)



def normal3p(point1: list | np.ndarray,
             point2: list | np.ndarray,
             point3: list | np.ndarray, norm: bool = False) -> np.ndarray:
    """
    Returns a vector normal to the plane defined by three points.
    """
    p1, p2, p3 = pad_with_zeros(point1, point2, point3)
    v1 = p2 - p1
    v2 = p3 - p1

    if norm:
        return unit(np.cross(v1, v2))
    else:
        return np.cross(v1, v2)



def unit(vector: list | np.ndarray) -> np.ndarray:
    """
    Returns unit vector.
    """
    v = np.array(vector, dtype=float)
    return v / np.linalg.norm(v)



def project_point_to_line(P: list | np.ndarray,
                          A: list | np.ndarray,
                          B: list | np.ndarray) -> np.ndarray:
    """
    Returns the projection of point P on a line defined by points A and B

    vector in direction of the line:   n = |B - A|
    projection of AP onto n:          ap = |P - A| . n / ||n||
    point closest to P on line:       P1 = A + ap * n
    """
    p, a, b = pad_with_zeros(P, A, B)
    n = unit(b - a)
    return a + np.dot(p - a, n) * n



def project_point_to_plane(P: list | np.ndarray,
                           A: list | np.ndarray,
                           B: list | np.ndarray,
                           C: list | np.ndarray) -> float:
    """
    Returns the projection of point P to a plane defined by points A, B and C

    vector in 1st direction of the plane:   n1 = |B - A|
    vector in 2nd direction of the plane:   n2 = |C - A|
    projection of AP onto n1:              ap1 = |P - A| . n1 / ||n1||
    projection of AP onto n2:              ap2 = |P - A| . n2 / ||n2||
    point closest to P on plane:            P1 = A + ap1 * n1 + ap2 * n2
    """
    p, a, b, c = pad_with_zeros(P, A, B, C)
    n1 = unit(b - a)
    n2 = unit(c - a)
    return a + np.dot(p - a, n1) * n1 + np.dot(p - a, n2) * n2



def closest_line_to_line(A1: list | np.ndarray,
                         A2: list | np.ndarray,
                         B1: list | np.ndarray,
                         B2: list | np.ndarray) -> tuple[np.ndarray]:
    """
    Finds two closes points on two lines defined by two points each.
    If the points are the same returns just one point and None, that
    means the lines intersect.
    """
    a1, a2, b1, b2 = pad_with_zeros(A1, A2, B1, B2)
    a = unit(a2 - a1)
    b = unit(b2 - b1)
    # first check if parrallel (b is a linear combination of a)
    if np.dot(a, b) == 1.0:
        return None, None

    n = normal2v(a, b, norm = True)
    # TODO:
    # t . v = 0
    # u . v = 0
    # a1 + t * a + v * n = b1 + u * b
    # from: https://math.stackexchange.com/questions/846054/closest-points-on-two-line-segments
    R1 = sum((a2 - a1) ** 2)
    R2 = sum((b2 - b1) ** 2)
    D4321 = sum((b2 - b1) * (a2 - a1))
    D3121 = sum((b1 - a1) * (a2 - a1))
    D4331 = sum((b2 - b1) * (b1 - a1))

    t = (D4321 * D4331 + D3121 * R2) / (R1 * R2 + D4321 ** 2)
    u = (D4321 * D3121 + D4331 * R1) / (R1 * R2 + D4321 ** 2)

    P1 = a1 + t * a
    P2 = b1 + u * b
    # check for line intersection
    if np.array_equal(P1, P2):
        return P1, None
    else:
        return P1, P2



def distance_point_to_line(P: list | np.ndarray,
                           A: list | np.ndarray,
                           B: list | np.ndarray) -> float:
    """
    Returns the distance from point P to a line defined by points A and B

    vector in direction of the line:   n = |B - A|
    projection of AP onto n:          ap = |P - A| . n / ||n||
    point closest to P on line:       P1 = A + ap * n
    distance:                          d = ||P1 - P||
    """
    return distance(P, project_point_to_line(P, A, B))



def distance_point_to_plane(P: list | np.ndarray,
                            A: list | np.ndarray,
                            B: list | np.ndarray,
                            C: list | np.ndarray) -> float:
    """
    Returns the distance from point P to a plane defined by points A, B and C

    vector in 1st direction of the plane:   n1 = |B - A|
    vector in 2nd direction of the plane:   n2 = |C - A|
    projection of AP onto n1:              ap1 = |P - A| . n1 / ||n1||
    projection of AP onto n2:              ap2 = |P - A| . n2 / ||n2||
    point closest to P on plane:            P1 = A + ap1 * n1 + ap2 * n2
    distance:                                d = ||P1 - P||
    """
    return distance(P, project_point_to_plane(P, A, B, C))



def distance_line_to_line(A1: list | np.ndarray,
                          A2: list | np.ndarray,
                          B1: list | np.ndarray,
                          B2: list | np.ndarray) -> float:
    """
    returns the distance between the closest points on tho lines
    """
    P1, P2 = closest_line_to_line(A1, A2, B1, B2)
    if P1 is None:    # parallel
        return distance_point_to_line(A1, B1, B2)
    elif P2 is None:  # intersecting
        return 0.
    else:
        return np.linalg.norm(P1, P2)


def reverse_cuthill_mckee(A: np.ndarray, reorder: bool = False):
    """
    Reverse Cuthill-McKee algorithm for reordering matrix for smallest
    badwidth

    The square A matrix format is such that 1 at position (i, j)
    means that node i is connected to variable j

    A = [[1. 0. 0. 0. 1. 0. 0. 0.]
         [0. 1. 1. 0. 0. 1. 0. 1.]
         [0. 1. 1. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 1. 0.]
         [1. 0. 1. 0. 1. 0. 0. 0.]
         [0. 1. 0. 0. 0. 1. 0. 1.]
         [0. 0. 0. 1. 0. 0. 1. 0.]
         [0. 1. 0. 0. 0. 1. 0. 1.]]

    To create the connectivity matrix above one can do:

    >>> A = np.diag(np.ones(8))
    >>> nzc = [[4], [2, 5, 7], [1, 4], [6], [0, 2], [1, 7], [3], [1, 5]]
    >>> for i in range(len(nzc)):
    >>>     for j in nzc[i]:
    >>>         A[i, j] = 1
    """

    def getAdjacency(Mat: np.ndarray):
        """
        return the adjacncy matrix for each node
        """
        adj = [0] * Mat.shape[0]
        for i in range(Mat.shape[0]):
            q = np.flatnonzero(Mat[i])
            q = list(q)
            q.pop(q.index(i))
            adj[i] = q
        return adj

    def getDegree(Graph: np.ndarray):
        """
        find the degree of each node. That is the number
        of neighbours or connections.
        (number of non-zero elements) in each row minus 1.
        Graph is a Cubic Matrix.
        """
        degree = [0]*Graph.shape[0]
        for row in range(Graph.shape[0]):
            degree[row] = len(np.flatnonzero(Graph[row]))-1
        return degree

    def RCM_loop(deg, start, adj, pivots, R):
        """
        Reverse Cuthil McKee ordering of an adjacency Matrix
        """
        digar = np.array(deg)
        # use np.where here to get indecies of minimums
        if start not in R:
            R.append(start)
        Q = adj[start]
        for idx, item in enumerate(Q):
            if item not in R:
                R.append(item)
        Q = adj[R[-1]]
        if set(Q).issubset(set(R)) and len(R) < len(deg) :
            p = pivots[0]
            pivots.pop(0)
            return RCM_loop(deg, p, adj, pivots, R)
        elif len(R) < len(deg):
            return RCM_loop(deg, R[-1], adj, pivots, R)
        else:
            R.reverse()
            return R

    # define the Result queue
    R = ["C"] * A.shape[0]
    adj = getAdjacency(A)
    degree = getDegree(A)
    digar = np.array(degree)
    pivots = list(np.where(digar == digar.min())[0])
    inl = []

    P = np.array(RCM_loop(degree, 0, adj, pivots, inl))

    # permute the matrix A if needed
    if reorder:
        B = np.array(A)

        for i in range(B.shape[0]):
            B[:, i] = B[P, i]

        for i in range(B.shape[0]):
            B[i, :] = B[i, P]
        return P, B
    else:
        return P



def show_matrix(A: list, title: list):
    """
    ncol = nrow + 2
    nrow * (nrow + 2) = nplots
    nplots = nrow^2 + 2 nrow
    0 = nrow^2 + 2 nrow - nplots
    nrow = (-2 +- (2 ^ 2 + 4 nplots) ** 0.5) / 2
    nrow = ceil((-2 + (4 + 4 * nplots) ** 0.5) / 2)

    Use:
    >>> A = np.diag(np.ones(8))
    >>> nzc = [[4], [2, 5, 7], [1, 4], [6], [0, 2], [1, 7], [3], [1, 5]]
    >>> for i in range(len(nzc)):
    >>>     for j in nzc[i]:
    >>>         A[i, j] = 1
    >>> P, B = reverse_cuthill_mckee(A, reorder=True)
    >>> show_matrix([A, B], ["orig", "ordered"])
    """
    if type(A) is list:
        nplots = len(A)
        nrow = ceil((-2 + (4 + 4 * nplots) ** 0.5) / 2)
        ncol = nplots / nrow
        if ncol != ceil(ncol):
            ncol = ceil(ncol)
        nrow = int(nrow)
        ncol = int(ncol)
        fig, axs = plt.subplots(nrow, ncol, layout="tight")
        if nrow == 1:
            for n in range(nplots):
                axs[n].spy(A[n])
                axs[n].set_title(title[n])
        else:
            for i in range(nrow):
                for j in range(ncol):
                    if i * ncol + j >= nplots:
                        axs[i][j].axis("off")
                    else:
                        axs[i][j].spy(A[i * ncol + j])
                        axs[i][j].set_title(title[i * ncol + j])
    else:
        fig, ax = plt.subplots(1, 1)
        ax.spy(A)
        ax.set_title(title)

    plt.show()



def gramm_schmidt(A: np.ndarray) -> np.ndarray:
    """
    Performs Gramm-Schmidt orthonormalisation of the vector base, where
    each column of matrix A is a separate vector that should be orthonormalised
    to the ones with lower column numbers
    """
    (n, m) = A.shape

    for i in range(m):
        q = A[:, i] # i-th column of A

        for j in range(i):
            q = q - np.dot(A[:, j], A[:, i]) * A[:, j]

        if np.array_equal(q, np.zeros(q.shape)):
            raise np.linalg.LinAlgError("The column vectors are not linearly independent")

        # normalize q
        q = q / np.sqrt(np.dot(q, q))

        # write the vector back in the matrix
        A[:, i] = q



def MAC(vector1: list | dict | np.ndarray, vector2: list | dict | np.ndarray) -> float:
    """
    Returns the MAC value of two vectors

    MAC = (A . B)^2 / ((A . A) * (B . B))
    """
    if type(vector1) is dict:
        A = np.array([v for k, v in vector1.items()], dtype = float).flatten()
    else:
        A = np.array(vector1, dtype=float).flatten()

    if type(vector1) is dict:
        B = np.array([v for k, v in vector2.items()], dtype = float).flatten()
    else:
        B = np.array(vector2, dtype=float).flatten()

    mac = (np.dot(A, B) ** 2) / (np.dot(A, A) * np.dot(B, B))
    return mac



def MACMatrix(bData: dict, cData: dict) -> np.ndarray:
    """
    Creates the MAC matrix between two sets of vectors

    Returns:
        bkeys - row headers
        ckeys - column headers
        macM  - np.ndarray MAC matrix
    """
    bkeys = list(bData.keys())
    ckeys = list(cData.keys())

    macM = []
    for bkey in bkeys:
        bkeyData = bData[bkey]
        v = []
        for ckey in ckeys:
            ckeyData = cData[ckey]
            v.append(MAC(bkeyData, ckeyData))
        macM.append(v)
    return bkeys, ckeys, np.array(macM, dtype=float)



def mapping(source: np.ndarray, values: np.ndarray, target: np.ndarray,
            cube_scale: float = 20., distances: bool = False,
            max_distance: float = None) -> np.ndarray:
    """
    Map scalar/vector/tensor from source coordinates to target coordinates

    In:
        source       - source coordinates [np.ndarray]
        values       - values on source coordinates [np.ndarray]
        target       - target coordinates [np.ndarray]
        cube_scale   - model extents multiplier to create bounding box of
                       average values around the model so as to not extrapolate
        distances    - calculate the closest distances between the two sets of
                       coordinates
        max_distance - maximum distance for the mapping
    """
    spoints = np.array(source, dtype=float)
    tpoints = np.array(target, dtype=float)

    if source.shape[0] != values.shape[0]:
        raise ValueError(f"The number of values must match the number of source " +
                         f"coordinates {values.shape[0]:n} != {source.shape[0]:n}.")

    # create a cuboid for extrapolation
    smin, smax = np.min(source, axis=0), np.max(source, axis=0)
    tmin, tmax = np.min(target, axis=0), np.max(target, axis=0)
    min = np.minimum(smin, tmin).flatten()
    max = np.maximum(smax, tmax).flatten()
    avg = (min + max) / 2.
    cube = np.zeros((8, 3), dtype=float)
    cube[0] = (np.array([max[0], max[1], max[2]], dtype=float) - avg) * cube_scale + avg
    cube[1] = (np.array([min[0], max[1], max[2]], dtype=float) - avg) * cube_scale + avg
    cube[2] = (np.array([min[0], min[1], max[2]], dtype=float) - avg) * cube_scale + avg
    cube[3] = (np.array([max[0], min[1], max[2]], dtype=float) - avg) * cube_scale + avg
    cube[4] = (np.array([max[0], min[1], min[2]], dtype=float) - avg) * cube_scale + avg
    cube[5] = (np.array([min[0], min[1], min[2]], dtype=float) - avg) * cube_scale + avg
    cube[6] = (np.array([max[0], max[1], min[2]], dtype=float) - avg) * cube_scale + avg
    cube[7] = (np.array([min[0], max[1], min[2]], dtype=float) - avg) * cube_scale + avg


    # select the value type
    if len(values.shape) == 1:
        svalues = values.reshape(values.shape[0], 1, 1)
        value_type = "scalar"
    elif len(values.shape) == 2:
        svalues = values.reshape(values.shape[0], 1, values.shape[1])
        value_type = "vector"
    else:
        svalues = values.copy()
        value_type = "tensor"

    # pair original coordinates to scalar values and add the cuboid
    mean = np.mean(svalues, axis=0)
    cube_values = np.array([mean] * cube.shape[0],
                           dtype=float).reshape(-1, svalues.shape[1], svalues.shape[2])
    spoints = np.concatenate((spoints, cube), axis=0)
    svalues = np.concatenate((svalues, cube_values), axis=0)

    # map values to new nodes
    grid = np.empty((tpoints.shape[0], svalues.shape[1], svalues.shape[2]), dtype=float)
    for m in range(svalues.shape[1]):
        for n in range(svalues.shape[2]):
            grid[:,m,n] = scipy.interpolate.griddata(spoints, svalues[:,m,n], tpoints, method="linear")

    # reshape the interpolated values back to the original shape
    if value_type == "scalar":
        grid = grid.reshape(grid.shape[0])
    elif value_type == "vector":
        grid = grid.reshape(grid.shape[0], -1)

    # if closest distances are reuqested
    if distances:
        tree = scipy.spatial.cKDTree(spoints)
        xi = scipy.interpolate.interpnd._ndim_coords_from_arrays(tpoints,
                                                                 ndim=tpoints.shape[1])
        distances, indexes = tree.query(xi)

        # Copy original result but mask missing values with NaNs
        if max_distance:
            grid2 = grid[:]
            if len(grid.shape) > 1:
                grid2[distances > max_distance, :] = np.nan
            else:
                grid2[distances > max_distance] = np.nan
            grid = grid2
        distances = dict(list(zip(tids, distances)))

    else:
        distances = None

    if distances:
        return grid, distances
    else:
        return grid



def sin_variable_frequency(times: np.ndarray, freqs: np.ndarray,
                           signal: np.ndarray) -> np.ndarray:
    """
    Sine funciton with variable frequency

    f(t) = A(t) * sin(2.0 π ∫f(t)dt + φ)
    """
    signal_t = []
    for i, t in enumerate(times):
        if i == 0:
            Iftdt = freqs[i] * t
        else:
            Iftdt += (freqs[i-1] + freqs[i]) / 2.0 * (t - times[i-1])

        if signal.dtype in (complex, np.complex,np.complex64, np.complex128, np.complex256):
            s = signal[i].real * np.sin(2.0 * np.pi * Iftdt + signal[i].imag)
        else:
            s = signal[i] * np.sin(2.0 * np.pi * Iftdt)

        if np.isnan(s):
            raise ValueError(f"NaN value for {t = } {signal[i] = }")

        signal_t.append(s)
    return np.array(signal_t, dtype=float)



def interpolate_sin(times_old: np.ndarray, freqs_old: np.ndarray,
                    times_new: np.ndarray, freqs_new: np.ndarray,
                    signal: np.ndarray):
    shifted = False
    if signal.dtype in (complex, np.complex,np.complex64, np.complex128, np.complex256):
        amplitude_old = signal.real
        phase_old = signal.imag
        if np.min(phase_old) < 0.:
            shifted = True
            phase_old += np.pi
    else:
        amplitude_old = signal
        phase_old = np.zeros(a.shape, dtype=float)

    amplitude_new = np.interp(times_new, times_old, amplitude_old)
    phase_new = np.interp(times_new, times_old, phase_old, period = 2.0 * np.pi)
    if shifted:
        phase_new -= np.pi

    return np.array([complex(a, p) for a, p in zip(amplitude_new, phase_new)], dtype=complex)



def interpolate_sin():
    #              Hz :  mm/s2
    amplitudes = { 10.:  5000.0,
                   20.: 10000.0,
                   30.:  3000.0,
                   50.:  1000.0,
                  100.:   100.0}
    #         Hz :   s
    sweep = { 10.:  30.,
              15.:  60.,
             100.: 600.}
    # frequency
    f = np.linspace(10., 100., 90)
    a = np.interp(f, np.array(list(amplitudes.keys()), dtype=float),
                     np.array(list(amplitudes.values()), dtype=float))
    t = np.interp(f, np.array(list(sweep.keys()), dtype=float),
                     np.array(list(sweep.values()), dtype=float))
    p = a.copy()
    p[:] = 0.
    signal = np.array([complex(a[i], p[i]) for i in range(f.shape[0])])

    print(f"{f = }")
    print(f"{t = }")
    print(f"{a = }")
    print(f"{p = }")
    print(f"{signal = }")

    # plt.plot(f, a, label="original data")
    # plt.legend()
    # plt.show()

    sampling_rate = f[-1] * 10.
    dt = 1. / sampling_rate
    tn = np.linspace(t[0], t[-1], int((t[-1] - t[0]) / dt) + 1)
    fn = np.interp(tn, t, f)
    print(f"{tn = }")

    # TODO:
    Iftdt = 1. / 2. * fn[0] * tn[0]
    for tt in tn:
        idx1 = np.where(t <= tt)[0][-1]
        idx2 = np.where(t >= tt)[0][ 0]
        # print(f"{tt = }, {idx1 = }, {idx2 = }")

        s1 = signal[idx1].real * np.sin(2. * np.pi * Iftdt + signal[idx1].imag)



def PSD(times: np.ndarray, signal: np.ndarray,
        plot: bool = False, show_immediately: bool = True) -> (np.ndarray, np.ndarray):
    if times.shape[0] != signal.shape[0]:
        raise ValueError(f"PSD: time vector and signal vector must have same " +
                         f"length ({times.shape} != {signal.shape}")
    N = signal.shape[0]                              # number of signal values
    fs = 1.0 / (times[1] - times[0])                 # sampling frequency
    fft  = np.fft.fftshift(np.fft.fft(signal))       # fast fourrier transform
    freq = np.fft.fftshift(np.fft.fftfreq(N, 1./fs)) # frequencies with zero in the middle
    fft  = fft[N // 2 + 1:]
    freq = freq[N // 2 + 1:]
    Pxx  = (1 / (fs * N)) * np.abs(fft) ** 2
    Pxx *= 2.

    if plot:
        plt.plot(freq, Pxx, label="PSD")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [mm2/s4]")
        plt.suptitle("PSD estimate using FFT")
        if show_immediately:
            plt.legend()
            plt.show()

    return freq, Pxx


def PSD_scipy(times: np.ndarray, signal: np.ndarray,
        plot: bool = False, show_immediately: bool = True) -> (np.ndarray, np.ndarray):
    fs = 1.0 / (times[1] - times[0])                       # sampling frequency
    freq, Pxx = scipy.signal.periodogram(signal, fs=fs, scaling="density")    # frequency, PSD

    if plot:
        plt.plot(freq, Pxx, label="PSD")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [mm2/s4]")
        plt.suptitle("PSD estimate using FFT")
        if show_immediately:
            plt.legend()
            plt.show()

    return freq, Pxx



def CSD(times: np.ndarray, signal1: np.ndarray, signal2: np.ndarray,
        plot: bool = False, show_immediately: bool = True) -> (np.ndarray, np.ndarray):
    N = signal1.shape[0]             # number of signal values
    dt = times[1] - times[0]
    fs = 1.0 / dt                    # sampling frequency
    Cxy = np.fft.fft(signal1) * np.conjugate(np.fft.fft(signal2))
    Cxy = np.fft.fftshift(Cxy)
    Cxy = Cxy[N // 2 + 1:]
    freq = np.fft.fftshift(np.fft.fftfreq(N, 1./fs)) # frequencies with zero in the middle
    freq = freq[N // 2 + 1:]

    Cxy = 1. / (N * fs) * abs(Cxy)

    if plot:
        plt.plot(freq, Cxy, label="Cross PSD")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Cross PSD [mm2/s4]")
        plt.suptitle("Cross Power Spectral Density")
        if show_immediately:
            plt.legend()
            plt.show()

    return freq, Cxy



def Coherence(times: np.ndarray, signal1: np.ndarray, signal2: np.ndarray,
              plot: bool = False, show_immediately: bool = True) -> (np.ndarray, np.ndarray):
    N = signal1.shape[0]             # number of signal values
    dt = times[1] - times[0]
    fs = 1.0 / dt                    # sampling frequency

    fft1 = np.fft.fft(signal1)
    fft2 = np.fft.fft(signal2[::-1])

    Cxy = np.abs(np.fft.fftshift(fft1 * np.conjugate(fft2))[N // 2 + 1:]) / (N * fs)
    # Pxx = np.abs(np.fft.fftshift(fft1 * np.conjugate(fft1))[N // 2 + 1:]) / (N * fs)
    # Pyy = np.abs(np.fft.fftshift(fft2 * np.conjugate(fft2))[N // 2 + 1:]) / (N * fs)
    f1, Pxx = PSD(times, signal1)
    f2, Pyy = PSD(times, signal2)

    freq = np.fft.fftshift(np.fft.fftfreq(N, 1./fs)) # frequencies with zero in the middle
    freq = freq[N // 2 + 1:]

    coherence = Cxy ** 2 / (Pxx * Pyy)

    if plot:
        f2, c2 = scipy.signal.coherence(signal1, signal2, fs=fs)

        f31, c31 = scipy.signal.csd(signal1, signal2, fs=fs)
        f32, p32 = scipy.signal.csd(signal1, signal1, fs=fs)
        f33, p33 = scipy.signal.csd(signal2, signal2, fs=fs)

        c3 = np.abs(c31) ** 2 / (p32 * p33)

        plt.plot(freq, coherence, label="manual")
        plt.plot(f2, c2, label="scipy")
        plt.plot(f31, c3, label="scipy 2")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Coherence [-]")
        plt.suptitle("Coherence")
        if show_immediately:
            plt.legend()
            plt.show()

    return freq, coherence

def crossSpectrum(x, y, nperseg=1000):
	# from: https://stackoverflow.com/questions/51258394/compute-coherence-in-python
	#-------------------Remove mean-------------------
	cross = numpy.zeros(nperseg, dtype='complex128')
	for ind in range(x.size / nperseg):

		xp = x[ind * nperseg: (ind + 1)*nperseg]
		yp = y[ind * nperseg: (ind + 1)*nperseg]
		xp = xp - numpy.mean(xp)
		yp = yp - numpy.mean(xp)

		# Do FFT
		cfx = numpy.fft.fft(xp)
		cfy = numpy.fft.fft(yp)

		# Get cross spectrum
		cross += cfx.conj()*cfy
	freq=numpy.fft.fftfreq(nperseg)
	return cross, freq



def test_crossSpectrum():
	# from: https://stackoverflow.com/questions/51258394/compute-coherence-in-python
	x=numpy.linspace(-2500,2500,50000)
	noise=numpy.random.random(len(x))
	y=10*numpy.sin(2*numpy.pi*x)
	y2=5*numpy.sin(2*numpy.pi*x)+5+noise*50

	p11,freq=crossSpectrum(y,y)
	p22,freq=crossSpectrum(y2,y2)
	p12,freq=crossSpectrum(y,y2)

	# coherence
	coh=numpy.abs(p12)**2/p11.real/p22.real
	plot(freq[freq > 0], coh[freq > 0])
	xlabel('Normalized frequency')
	ylabel('Coherence')



def test():
    # np.random.seed(19680801)
    fs = 1000.
    t = np.linspace(0., 1., int(fs) + 1)
    x = np.cos(2. * np.pi * 100. * t) + np.random.randn(t.shape[0])
    y = np.cos(2. * np.pi * 100. * t) + np.random.randn(t.shape[0])

    f, c = Coherence(t, x, y, True, True)



if __name__ == "__main__":
    test()

