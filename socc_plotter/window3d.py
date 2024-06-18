import pyqtgraph.opengl as gl
from PyQt5.Qt import Qt


class Window3D(gl.GLViewWidget):
    def __init__(self):
        super().__init__()

    def keyPressEvent(self, event):
        # TODO: Make this work
        if event.key() == Qt.Key_Space:
            print("Space key pressed")
        elif event.key() == Qt.Key_Escape:
            exit(0)
