import sys
import os
from PyQt5.QtWidgets import QApplication

from gui.app import App


class Main(object):

    def __init__(self):
        pass

    def clearScreen(self):
        os.system('cls')  # For Windows
        os.system('clear')  # For Linux/OS X

    def main(self):
        app = QApplication(sys.argv)
        self.clearScreen()
        objectDetectionApp = App()
        objectDetectionApp.run()
        sys.exit(app.exec_())


if __name__ == "__main__":
    Main().main()