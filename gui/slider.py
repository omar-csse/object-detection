from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QSlider

class Slider(QSlider):

    pressed = pyqtSignal()
    released = pyqtSignal()

    def mousePressEvent(self, event):
        self.pressed.emit()
        QSlider.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        self.released.emit()
        QSlider.mouseReleaseEvent(self, event)