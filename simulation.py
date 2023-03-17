# going to use this file for any simulation - initially just going
# to do a fun solar system simulation to get to grips

import math
import matplotlib


class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    # the goal of _repr_ is to be unambiguous
    def __repr__(self):
        return f"Vector ({self.x}, {self.y}, {self.z})"

    # the goal of _str_ is to be readable
    def __str__(self):
        return f"{self.x}i + {self.y}j + {self.z}k"

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        elif item == 2:
            return self.z
        else:
            raise IndexError("There are only 3 elements in this array! xoxo")

    def __add__(self, other):
        return Vector(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __sub__(self, other):
        return Vector(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )

    def __mul__(self, other):
        if isinstance(other, Vector):  # if vector do DOT
            return Vector(
                self.x * other.x,
                self.y * other.y,
                self.z * other.z
            )
        elif isinstance(other, (int, float)):  # if int/float do SCALAR
            return Vector(
                self.x * other,
                self.y * other,
                self.z * other
            )
        else:
            raise TypeError("that's the wrong type babe, ints, floats and vectors only!")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vector(
                self.x / other,
                self.y / other,
                self.z / other,
            )
        else:
            raise TypeError("that's the wrong type babe! ints please hun x")

    def get_magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        magnitude = self.get_magnitude()
        return Vector(
            self.x / magnitude,
            self.y / magnitude,
            self.z / magnitude,
        )


