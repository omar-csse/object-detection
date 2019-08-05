import sys
import os
from PyQt5.QtWidgets import QApplication

from app import App

def clearScreen():
    os.system('cls')  # For Windows
    os.system('clear')  # For Linux/OS X

def main():
    app = QApplication(sys.argv)
    clearScreen()
    objectDetectionApp = App()
    objectDetectionApp.run()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()