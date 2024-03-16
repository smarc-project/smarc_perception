from enum import Enum


class Side(Enum):
    """The side-scan sonar ping side (port or starboard)"""
    PORT = 0
    STARBOARD = 1


class ObjectID(Enum):
    """ObjectID for object detection"""
    NADIR = 0
    ROPE = 1
    BUOY = 2
    PIPE = 3
