from collections import namedtuple


V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])

def substract(v0, v1):
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def norm(v0):
    vLength = length(v0)
    if not vLength:
        return V3(0, 0, 0)

    return V3(v0.x/vLength, v0.y/vLength, v0.z/vLength)
    
def dot(v0, v1):
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

def length(v0):
    return (v0.x**2 + v0.y**2 + v0.z**2)**0.5

def cross(v1, v2):
    return V3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x,
    )