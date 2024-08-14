import numpy


def norm(xyz):
    xyz = numpy.array(xyz)
    return xyz / numpy.linalg.norm(xyz)
