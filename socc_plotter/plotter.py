from typing import Callable, Tuple

import numpy as np
import pyqtgraph.opengl as gl
import vedo
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout
from pyqtgraph.Vector import Vector

from .window2d import Window2D
from .window3d import Window3D

Y_OFFSET = -25


class Plotter:

    screen_width: int
    screen_height: int
    screen_origin: Tuple[int, int]
    _callback: Callable

    def __init__(self, callback: Callable):
        """
        Semantic Occupancy Plotter
        """

        # TODO: preemptive check to ensure opencv-python-headless is installed

        self.set_callback(callback)

        self.app = QtWidgets.QApplication([])
        self.desktop = QtWidgets.QApplication.desktop()
        assert self.desktop is not None, "Error getting display"
        screenRect = self.desktop.screenGeometry()
        self.screen_width = screenRect.width()
        self.screen_height = screenRect.height()

        screen = QtWidgets.QDesktopWidget().screenGeometry()
        origin = screen.topLeft()
        self.screen_origin = (
            int(origin.x()),
            int(origin.y()),
        )

        self.window_2D = Window2D()
        self.window_3D = Window3D()

        ##################################################
        # window_2D
        ##################################################
        self.window_2D.setWindowTitle("UI 2D")
        self.image_label = QLabel()

        img = np.zeros(
            (self.screen_width // 2, self.screen_height, 3), dtype=np.uint8
        )
        # Turn up red channel to full scale
        img[..., 0] = 255
        qImg = QPixmap(
            QImage(img.data, img.shape[0], img.shape[1], QImage.Format_RGB888)
        )
        self.image_label.setPixmap(qImg)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.image_label)

        # snap window_2D to left half of screen
        self.window_2D.setGeometry(
            self.screen_origin[0],
            self.screen_origin[1],
            int(self.screen_width / 2),
            self.screen_height,
        )

        self.window_2D.setLayout(self.vbox)
        self.window_2D.setWindowFlags(Qt.FramelessWindowHint)

        ##################################################

        ##################################################
        # window_3D
        ##################################################
        self.window_3D = gl.GLViewWidget()

        self.window_3D.resize(800, 600)
        self.window_3D.opts["center"] = Vector(0, Y_OFFSET, 0)
        self.window_3D.opts["distance"] = 50
        self.window_3D.opts["rotation"] = QtGui.QQuaternion(1, 0, 0, 0)
        self.window_3D.opts["fov"] = 30
        self.window_3D.opts["elevation"] = 15
        self.window_3D.opts["azimuth"] = -90
        self.window_3D.setWindowTitle("UI 3D")
        # self.window_3D.setWindowFlags(Qt.FramelessWindowHint)

        self.window_3D.setGeometry(
            self.screen_origin[0] + self.screen_width // 2,
            self.screen_origin[1],
            self.screen_width // 2,
            self.screen_height,
        )

        self.grid_item = gl.GLGridItem(
            size=QtGui.QVector3D(40, 60, 1),
        )

        self.mesh_region = gl.GLMeshItem(
            pos=np.array([0, Y_OFFSET, 0], dtype=np.float32).reshape(1, 3),
            vertexes=np.array([]),
            faces=np.array([]),
            faceColors=np.array([]),
            drawEdges=False,
            edgeColor=(0, 0, 0, 1),
        )
        self.mesh_region.translate(0, Y_OFFSET, 0)

        self.graph_region = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0, 1, 0, 0.5),
            size=0.10,
            pxMode=False,
        )
        self.graph_region.rotate(-90, 1, 0, 0)
        self.graph_region.translate(0, Y_OFFSET, 0)

        car = vedo.load("media/car.obj")
        car_faces = np.array(car.faces())
        car_vertices = np.array(car.points())
        car_colors = np.array(
            [[0.5, 0.5, 0.5, 1] for i in range(len(car_faces))]
        )
        car_vertices = 0.025 * car_vertices * 2.5

        self.car_mesh_region = gl.GLMeshItem(
            pos=np.array([0, Y_OFFSET, 0], dtype=np.float32).reshape(1, 3),
            vertexes=car_vertices,
            faces=car_faces,
            faceColors=car_colors,
            drawEdges=False,
            edgeColor=(0, 0, 0, 1),
        )
        self.car_mesh_region.rotate(90, 1, 0, 0)
        self.car_mesh_region.rotate(180, 0, 0, 1)
        self.car_mesh_region.translate(0, Y_OFFSET, 1.0)

        self.window_3D.addItem(self.car_mesh_region)
        # self.window_3D.addItem(self.mesh_region)
        # self.window_3D.addItem(self.occupancy_mesh_region)
        self.window_3D.addItem(self.graph_region)
        self.window_3D.addItem(self.grid_item)
        self.window_3D.setCameraPosition(
            pos=QtGui.QVector3D(0, 0, 0),
        )
        ##################################################
        self.graph_context = dict(
            mesh_region=self.mesh_region, graph_region=self.graph_region
        )

    def set_callback(self, callback: Callable) -> None:
        self._callback = callback

    def update_graph(
        self,
    ) -> None:
        try:
            self._callback(self.graph_context)
        except KeyboardInterrupt:
            exit(0)

    def start(self) -> None:
        """
        Blocking call to start UI thread
        """
        self.window_3D.show()
        self.window_2D.show()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update_graph)
        timer.start(1)

        QtGui.QGuiApplication.instance().exec_()  # type: ignore


def main():
    import time

    def callback(graph_context):
        time.sleep(1)
        print("in callback", graph_context)
        # mesh_region = graph_context["mesh_region"]
        graph_region = graph_context["graph_region"]

        points = np.array([[1, 0, 0]])
        colors = np.array([[1, 1, 1]])

        graph_region.setData(pos=points, color=colors)

    plotter = Plotter(
        callback=callback,
    )
    plotter.start()


if __name__ == "__main__":
    main()
