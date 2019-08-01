from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QPushButton

class Button(QPushButton):

    pressed = pyqtSignal()
    released = pyqtSignal()

    def mousePressEvent(self, event):
        self.pressed.emit()
        QPushButton.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        self.released.emit()
        QPushButton.mouseReleaseEvent(self, event)
