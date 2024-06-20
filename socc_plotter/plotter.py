from typing import Callable, Tuple

import cv2
import numpy as np
import pyqtgraph.opengl as gl
import qimage2ndarray
import vedo
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from PyQt5.Qt import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout
from pyqtgraph.Vector import Vector

from .trajectory_utils import create_meshes, trajectory_to_3D
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
        self.window_3D.opts["distance"] = 40
        self.window_3D.opts["rotation"] = QtGui.QQuaternion(1, 0, 0, 0)
        self.window_3D.opts["fov"] = 30
        self.window_3D.opts["elevation"] = 10
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
        self.grid_item.translate(0, 0, -1.0)

        self.mesh_region = gl.GLMeshItem(
            pos=np.array([0, Y_OFFSET, 0], dtype=np.float32).reshape(1, 3),
            vertexes=np.array([]),
            faces=np.array([]),
            faceColors=np.array([]),
            drawEdges=False,
            edgeColor=(0, 0, 0, 1),
        )
        # self.mesh_region.translate(0, Y_OFFSET, 0)
        self.mesh_region.rotate(180, 1, 0, 0)

        self.graph_region = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0, 1, 0, 0.5),
            size=0.1,
            # size=0.05,
            pxMode=False,
        )
        # self.graph_region.rotate(90, 1, 0, 0)
        self.graph_region.rotate(180, 1, 0, 0)
        # self.graph_region.rotate(180, 0, 0, 1)
        # self.graph_region.translate(0, Y_OFFSET, 0)
        self.graph_region.translate(0, 0, 2)

        car = vedo.load("media/car.obj")
        # car_faces = np.array(car.faces())
        # car_vertices = np.array(car.points())
        car_faces = car.faces()
        car_vertices = car.points()
        car_colors = np.array(
            [[0.5, 0.5, 0.5, 1] for i in range(len(car_faces))]
        )
        car_vertices = 0.025 * car_vertices * 2.0

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
        # self.car_mesh_region.translate(0, Y_OFFSET, 1.0)

        # self.window_3D.addItem(self.occupancy_mesh_region)
        self.window_3D.addItem(self.graph_region)
        self.window_3D.addItem(self.grid_item)

        self.window_3D.addItem(self.mesh_region)
        self.window_3D.addItem(self.car_mesh_region)
        self.window_3D.setCameraPosition(
            pos=QtGui.QVector3D(0, -Y_OFFSET, 0),
        )
        ##################################################

    def set_3D_trajectory(self, trajectory: np.ndarray, wheel_base: float):
        trajectory_3D = trajectory_to_3D(trajectory)
        mesh_data = create_meshes(
            trajectory_3D, wheel_base, color=(0.0, 1.0, 0.0, 1.0)
        )
        self.mesh_region.setMeshData(
            vertexes=np.array(mesh_data["vertexes"], dtype=np.float32),
            faces=np.array(mesh_data["faces"], dtype=np.uint32),
            faceColors=np.array(mesh_data["faceColors"], dtype=np.float32),
        )

    def set_callback(self, callback: Callable) -> None:
        self._callback = callback

    def update_graph(
        self,
    ) -> None:
        try:
            self._callback(self)
        except KeyboardInterrupt:
            exit(0)
        except Exception:
            import traceback

            traceback.print_exc()
            exit(1)

    def set_2D_visual(self, img: np.ndarray):
        img = cv2.resize(img, (self.screen_width // 2, self.screen_height))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qImg = QPixmap(qimage2ndarray.array2qimage(img_rgb))
        self.image_label.setPixmap(qImg)

    def set_3D_visual(self, points: np.ndarray, colors: np.ndarray) -> None:
        colors_with_alpha = np.ones((colors.shape[0], 4))

        # Copy the RGB values
        colors_with_alpha[:, :3] = colors

        blanks = (
            np.isclose(colors_with_alpha[:, 0], 0)
            & np.isclose(colors_with_alpha[:, 1], 0)
            & np.isclose(colors_with_alpha[:, 2], 0)
            & np.isclose(colors_with_alpha[:, 3], 1)
        )
        colors_with_alpha[blanks] = (0.5, 0.5, 0.5, 0.2)

        self.graph_region.setData(
            pos=points,
            color=colors_with_alpha,
        )

    def sleep(self, seconds: float) -> None:
        msecs = int(seconds * 1000)
        QtTest.QTest.qWait(msecs)  # type: ignore

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

    def callback(plot: Plotter):
        time.sleep(1)
        print("in callback")
        graph_region = plot.graph_region

        points = np.array([[1, 0, 0]])
        colors = np.array([[1, 1, 1]])

        graph_region.setData(pos=points, color=colors)

    plotter = Plotter(
        callback=callback,
    )
    plotter.start()


if __name__ == "__main__":
    main()
