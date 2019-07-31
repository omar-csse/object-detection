import sys
from PyQt5.QtWidgets import QApplication

from app import App


def main():
    app = QApplication(sys.argv)
    objectDetectionApp = App()
    objectDetectionApp.run()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()