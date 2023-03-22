import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
from simulation import Vector
import math
import numpy as np


def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


class CameraObject:
    def __init__(self, length):
        self.centre_x = 0
        self.centre_y = 0
        self.length = length
        self.vector = [0, 0, 0]
        self.vector_begin = [0, 0, - length]

    def updateVector(self, x_angle, y_angle, z_pos):
        end_vec = [self.centre_x, self.centre_y, z_pos]
        x_vec = self.length * np.tan(x_angle)
        y_vec = self.length * np.tan(y_angle)
        # this math is basically limited by the fact that the algorithm currently using always
        # assumes that the camera is at the centre of the pipe
        start_vec = [self.centre_x - x_vec, self.centre_y - y_vec, z_pos - self.length]
        self.vector = end_vec
        self.vector_begin = start_vec


class PipeEnvironment:
    def __init__(self, centre_x, centre_y, radius, height_z, camera):
        self.centre_x = centre_x  # this will be for the centre of the pipe
        self.centre_y = centre_y  # this is for the centre of the pipe
        self.radius = radius  # this is the radius of the pipe
        self.height_z = height_z  # this is the height of the pipe
        self.camera = camera  # this is the camera object

        # the plot:
        self.fig = plt.figure()  # create a figure for the whole pipe view
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect(aspect=(1, 1, 2))
        # self.fig2 = plt.figure()                                 # create another figure for the polar view
        # self.ax2 = self.fig2.add_subplot(projection='polar')

    def PlotPipeSurface(self):
        Xc, Yc, Zc = data_for_cylinder_along_z(self.centre_x, self.centre_y, self.radius, self.height_z)
        self.ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

    def UpdatePlot(self, x_angle, y_angle, z_pos):
        self.camera.updateVector(x_angle, y_angle, z_pos)
        self.ax.quiver(self.camera.vector_begin[0], self.camera.vector_begin[1], self.camera.vector_begin[2],
                       -self.camera.vector_begin[0], -self.camera.vector_begin[1], self.camera.vector[2])

