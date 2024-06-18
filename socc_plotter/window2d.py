from PyQt5.Qt import Qt
from PyQt5.QtWidgets import QWidget


class Window2D(QWidget):
    def __init__(self):
        super().__init__()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            print("Space key pressed")
        elif event.key() == Qt.Key_Escape:
            exit(0)
