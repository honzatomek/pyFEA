
import typing
from math import pi, sin, cos, tan, asin, acos, atan2
import numpy as np


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



def distance(point1: list | np.ndarray, point2: list | np.ndarray) -> float:
    """
    Returns the distance between two n-dimensional points.
    The shorter coordinate vector gets padded with zeros.
    """
    v1, v2 = pad_with_zeros(point1, point2)
    return np.linalg.norm(v2 - v1)



def angle2v(vector1: list | np.ndarray, vector2: list | np.ndarray, out: str = "radians") -> float:
    """
    Returns the angle between two n-dimensional vectors.
    """
    v1, v2 = pad_with_zeros(vector1, vector2)
    if out == "degrees":
        return degrees(acos(max(min(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 1), -1)))
    else:
        return acos(max(min(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 1), -1))



def angle3p(point1: list | np.ndarray, point2: list | np.ndarray, point3: list | np.ndarray, out: str = "radians") -> float:
    """
    Returns the angle between three n-dimensional points with the angle being at the 1st point.
    """
    p1, p2, p3 = pad_with_zeros(point1, point2, point3)
    v1 = p2 - p1
    v2 = p3 - p1
    if out == "degrees":
        return degrees(acos(max(min(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 1), -1)))
    else:
        return acos(max(min(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 1), -1))



def normal2v(vector1: list | np.ndarray, vector2: list | np.ndarray, norm: bool = False) -> np.ndarray:
    """
    Returns a vector normal to the supplied vectors using right-hand rule.
    """
    v1, v2 = pad_with_zeros(vector1, vector2)
    if norm:
        return unit(np.cross(v1, v2))
    else:
        return np.cross(v1, v2)



def normal3p(point1: list | np.ndarray, point2: list | np.ndarray, point3: list | np.ndarray, norm: bool = False) -> np.ndarray:
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



def project_point_to_line(P: list | np.ndarray, A: list | np.ndarray, B: list | np.ndarray) -> np.ndarray:
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
    P1, P2 = closest_line_to_line(A1, A2, B1, B2)
    if P1 is None:    # parallel
        return distance_point_to_line(A1, B1, B2)
    elif P2 is None:  # intersecting
        return 0.
    else:
        return np.linalg.norm(P1, P2)


# TODO:
def reverse_cuthill_mckee():
    pass



def max_value():
    pass



def max_matrix():
    pass



def mapping():
    pass



if __name__ == "__main__":

