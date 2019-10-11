import csv
import os
import sys
import pathlib
import logging
import traceback
from operator import add, sub
import cv2
import numpy as np
import pandas as pd

from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QFileInfo, QDir, QUrl, QTime
from PyQt5.QtWidgets import QWidget, QLabel, QMainWindow, QPushButton, QFileDialog
from PyQt5.QtWidgets import QHBoxLayout, QGridLayout, QComboBox, QAction
from PyQt5.QtWidgets import QMessageBox, QAbstractItemView, QTableWidgetItem, QTableWidget, QTableView
from PyQt5.QtGui import QPixmap

from gui.slider import Slider
from gui.button import Button
from gui.playback import PlayBack
from gui.saveVideo import SaveVideo
# from yolov3.yolo3 import YOLOv3
# from inceptionv4.inceptionv4 import InceptionV4


class App(QMainWindow):

    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        self.totalDolphins = 0
        self.totalSurfers = 0
        self.totalSharks = 0
        self.importedCSV = None
        self.importedVideo = None
        self.importedCSVPath = None
        self.importedVideoPath = None
        self.statistics = []
        self.isPlaying = False
        self.playbackThreadCreated = False
        self.saveVideoThreadCreated = False
        self.yolov3ThreadCreated = False
        self.inceptionv4ThreadCreated = False

        self.labels = None
        self.classes = []
        self.expLabels = None
        self.detectedFrames = []
        self.detectedStats = []

        self.initGUI()
        self.menu()

    def initGUI(self):
        
        self.statusBar().showMessage('Status: Ready')
        self.setWindowTitle("Object Detection App")
        self.setFocusPolicy(Qt.StrongFocus)

        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        self.setFixedSize(800, 650)

        self.buttonsWidget1 = QWidget()
        self.buttonsWidget1Layout = QHBoxLayout(self.buttonsWidget1)
        # Load data btn
        self.loadDataBtn = QPushButton('Load Data', self)
        self.loadDataBtn.setShortcut("Ctrl+O")
        self.loadDataBtn.clicked.connect(self.loadData)
        # import csv btn
        self.importCSVBtn = QPushButton('Import CSV', self)
        self.importCSVBtn.clicked.connect(self.importCSV)
        self.buttonsWidget1Layout.addWidget(self.loadDataBtn)
        self.buttonsWidget1Layout.addWidget(self.importCSVBtn)

        # Explorer
        self.explorerLabel = QLabel("Explorer:", self)
        self.explorerView = self.createTable("Name Size Type Modified", False, False, height=100)

        self.dlmWidget = QWidget()
        self.dlmWidgetLayout = QHBoxLayout(self.dlmWidget)
        self.dlmOptions = QComboBox()
        self.dlmOptions.addItems(["YOLOv3","Inceptionv4"])
        self.dlmLabel = QLabel("DLM:  ", self)
        self.dlmWidgetLayout.addWidget(self.dlmLabel)
        self.dlmWidgetLayout.addWidget(self.dlmOptions)

        # Process data btn
        self.processDataBtn = QPushButton('Process Data', self)
        self.processDataBtn.setShortcut("Ctrl+P")
        self.processDataBtn.clicked.connect(self.processData)

        # Statistics
        self.statLabel = QLabel("Statistics:", self)
        self.statView = self.createTable("# Object Confidence XMin YMin XMax YMax", False, False, minHeight=160)

        # Total surfers and Total dolphins
        self.surfersLabel = QLabel("Total surfers:  {}".format(self.totalSurfers), self)
        self.dolphinsLabel = QLabel("Total dolphins:  {}".format(self.totalDolphins), self)
        self.sharksLabel = QLabel("Total sharks:  {}".format(self.totalSharks), self)
        self.dolphinsLabel.setAlignment(Qt.AlignCenter)
        self.surfersLabel.setAlignment(Qt.AlignCenter)
        self.sharksLabel.setAlignment(Qt.AlignCenter)

        self.buttonsWidget2 = QWidget()
        self.buttonsWidget2Layout = QHBoxLayout(self.buttonsWidget2)
        # Load data btn
        self.saveVideoBtn = QPushButton('Save Video', self)
        self.saveVideoBtn.setShortcut("Ctrl+S")
        self.saveVideoBtn.clicked.connect(self.saveVideo)
        # import csv btn
        self.exportCSVBtn = QPushButton('Export CSV', self)
        self.exportCSVBtn.setShortcut("Ctrl+E")
        self.exportCSVBtn.clicked.connect(self.exportCSV)
        self.buttonsWidget2Layout.addWidget(self.saveVideoBtn)
        self.buttonsWidget2Layout.addWidget(self.exportCSVBtn)

        self.spaceLabel = QLabel("                       ", self)
        self.inputLabel = QLabel("Input Video/Image", self)
        self.playerLabel = QLabel("Player", self)
        
        # Video
        self.videoWidget = QVideoWidget()
        self.video = self.setupVideo(self.videoWidget)
        self.currentVideoState = self.video.state()

        # Pixmap label
        self.trainedVideoLabel = QLabel()
        self.trainedVideoLabel.setMaximumHeight(250)
        self.trainedVideoLabel.setScaledContents(True)

        self.video.positionChanged.connect(self.positionChanged)
        self.video.positionChanged.connect(self.handleLabel)
        self.video.durationChanged.connect(self.durationChanged)
        self.video.stateChanged.connect(self.resetVideo)

        self.videoBtnsWidget = QWidget()
        self.videoBtnsWidgetLayout = QHBoxLayout(self.videoBtnsWidget)
        # playe video btn
        self.playVideoBtn = Button("Play/Pause")
        self.playVideoBtn.pressed.connect(self.playVideo)
        # Stop video btn
        self.stopVideoBtn = Button("Stop")
        self.stopVideoBtn.released.connect(self.resetSlider)
        self.stopVideoBtn.pressed.connect(self.stopVideo)

        self.durationLabel = QLabel('00:00:00')
        self.videoSlider = Slider()
        self.videoSlider.setEnabled(False)
        self.videoSlider.setOrientation(Qt.Horizontal)
        self.videoSlider.setMinimumWidth(160)
        self.videoSlider.setTickInterval(1)
        # self.videoSlider.pressed.connect(self.pauseVideo)
        # self.videoSlider.released.connect(self.playVideo)
        # self.videoSlider.sliderMoved.connect(self.setPosition)
        self.videoSlider.sliderMoved.connect(self.handleLabel)

        self.videoBtnsWidgetLayout.addWidget(self.playVideoBtn)
        self.videoBtnsWidgetLayout.addWidget(self.stopVideoBtn)
        self.videoBtnsWidgetLayout.addWidget(self.videoSlider)
        self.videoBtnsWidgetLayout.addWidget(self.durationLabel)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.buttonsWidget1, 0, 0)
        mainLayout.addWidget(self.explorerLabel, 1, 0)
        mainLayout.addWidget(self.spaceLabel, 2, 0)
        mainLayout.addWidget(self.explorerView, 3, 0)
        mainLayout.addWidget(self.dlmWidget, 4, 0)
        mainLayout.addWidget(self.processDataBtn, 5, 0)
        mainLayout.addWidget(self.spaceLabel, 6, 0)
        mainLayout.addWidget(self.spaceLabel, 7, 0)
        mainLayout.addWidget(self.statLabel, 8, 0)
        mainLayout.addWidget(self.spaceLabel, 9, 0)
        mainLayout.addWidget(self.statView, 10, 0)
        mainLayout.addWidget(self.spaceLabel, 11, 0)
        mainLayout.addWidget(self.surfersLabel, 12, 0)
        mainLayout.addWidget(self.dolphinsLabel, 13, 0)
        mainLayout.addWidget(self.sharksLabel, 14, 0)
        mainLayout.addWidget(self.buttonsWidget2, 15, 0)
        mainLayout.addWidget(self.spaceLabel, 0, 1)
        mainLayout.addWidget(self.inputLabel, 0, 2)
        mainLayout.addWidget(self.videoWidget, 1, 2, 5, 1)
        mainLayout.addWidget(self.playerLabel, 8, 2)
        mainLayout.addWidget(self.spaceLabel, 9, 2)
        mainLayout.addWidget(self.trainedVideoLabel, 10, 2, 5, 1)
        mainLayout.addWidget(self.videoBtnsWidget, 15, 2)
        mainLayout.setSpacing(0)
        mainWidget.setLayout(mainLayout)

    def run(self):
        self.show()

    def menu(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        # File menu
        file_menu = menubar.addMenu(' &App')
        version_action = QAction(' &Version', self)
        quit_action = QAction(' &Quit', self)
        quit_action.setShortcut("Q")
        file_menu.addAction(version_action)
        file_menu.addAction(quit_action)
        # Help menu
        help_menu = menubar.addMenu(' &Help')
        about_action = QAction(' &About ', self)
        about_action.setShortcut("Ctrl+A")
        team_action = QAction(' &The team', self)
        team_action.setShortcut("Ctrl+T")
        help_menu.addAction(about_action)
        help_menu.addAction(team_action)

        quit_action.triggered.connect(self.close)
        version_action.triggered.connect(self.version)
        about_action.triggered.connect(self.about)
        team_action.triggered.connect(self.team)

    def version(self):
        QMessageBox.about(self, "Version", "Version: v1.0.0")

    def about(self):
        QMessageBox.about(self, "Info", "Object Detection App implemented using YOLOv3 and Inseptionv4 algorithms")

    def team(self):
        QMessageBox.about(self, "Team", "Aaron Reid\nJiajun Tian\nNathan Mallet\nOmar Alqarni")
 
    @staticmethod
    def createTable(headers, isGrid, isVisible, height=None, minHeight=None):
        tableWidget = QTableWidget()
        tableWidget.setShowGrid(isGrid)
        tableWidget.setSelectionBehavior(QTableView.SelectRows)
        tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tableWidget.setColumnCount(len(headers.split()))
        tableWidget.setHorizontalHeaderLabels(headers.split())
        tableWidget.verticalHeader().setVisible(isVisible)
        if height: tableWidget.setFixedHeight(height)
        if minHeight: tableWidget.setMinimumHeight(minHeight)
        return tableWidget

    @staticmethod
    def addItemToTable(model, data):
        currentRow = model.rowCount()
        model.insertRow(currentRow)
        for i, item in enumerate(data):
            model.setItem(currentRow , i, QTableWidgetItem(item))

    def removeAllItemsFromTable(self, tableView):
        for row in range(tableView.rowCount()):
            tableView.removeRow(row)

    def removeItemFromTable(self, fileType):
        for row in range(self.explorerView.rowCount()):
            item = self.explorerView.item(row, 2)
            if item.text() == fileType:
                self.explorerView.removeRow(row)
                break

    def addFilesToExplorer(self, fileName, fileType, scaler, extensionTag, isVideo=True):
        if fileName:
            info = QFileInfo(fileName)
            if info.baseName() == self.importedVideo  and isVideo:
                QMessageBox.critical(self, "Error", "Video already exist", buttons=QMessageBox.Ok)
            elif info.baseName() == self.importedCSV and not isVideo:
                QMessageBox.critical(self, "Error", "CSV file already exist", buttons=QMessageBox.Ok)
            else:
                if (isVideo):
                    self.removeItemFromTable("video")
                    self.importedVideoPath = QUrl.fromLocalFile(fileName)
                    self.setMedia(self.importedVideoPath)
                    self.importedVideo = info.baseName()
                    self.statusBar().showMessage('Status: Video/Image added')
                else:
                    self.removeItemFromTable("csv")
                    self.importedCSVPath = info.absoluteFilePath()
                    self.readCSV(info.absoluteFilePath(), info.baseName())
                    self.statusBar().showMessage('Status: CSV File added')

                size = str(info.size()/scaler)+extensionTag
                last_modified = info.lastModified().toString()[4:10]
                self.addItemToTable(self.explorerView, [info.baseName(), size, fileType, last_modified])   

    def openFile(self, fileType):
        options = QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie","", "{0} files (*.{0})".format(fileType), QDir.currentPath(), options)
        self.currentVideoPath = fileName
        return fileName

    def loadData(self):
        self.addFilesToExplorer(self.openFile('mp4'), 'video', 1000000, 'MB', isVideo=True)

    def importCSV(self):
        self.addFilesToExplorer(self.openFile('csv'), 'csv', 1000, 'KB', isVideo=False)

    def readCSV(self, path, baseName):
        try:
            self.importedCSVPath = path
        except:
            QMessageBox.critical(self, "Error", "Invalid CSV file", buttons=QMessageBox.Ok)

    def setMedia(self, path):
        content = QMediaContent(path)
        self.video.setMedia(content)

    @staticmethod
    def setupVideoWidget(width=600, height=400):
        videoWidget = QVideoWidget()
        videoWidget.setMinimumWidth(width)
        return videoWidget

    def processData(self):
        try:

            # call the two dlms for prediction here:
            if self.getDLM() == 0:
                self.yolov3Thread()
            elif self.getDLM() == 1:
                self.inceptionv4Thread()

            print("train in {} algorithm".format(self.getDLM()))
            self.statusBar().showMessage('Status: Processing data in {}'.format(self.getDLM()))


        except (IndexError, AttributeError, TypeError):
            QMessageBox.critical(self, "Error", "Select a file from explorer", buttons=QMessageBox.Ok)

    def inceptionv4Thread(self):
        if self.inceptionv4ThreadCreated == False:
            self.inceptionv4ThreadCreated = True
            self.inceptionv4Thread = InceptionV4(self.importedVideoPath.toString())
            self.inceptionv4Thread.doneSignal.connect(self.predictionDone)
            self.inceptionv4Thread.predictionSignal.connect(self.inceptionv4Result)
            self.inceptionv4Thread.frameSignal.connect(self.setFrame_h_w)
            self.inceptionv4Thread.start()
        elif self.inceptionv4Thread.isFinished():
            self.inceptionv4Thread = InceptionV4(self.importedVideoPath.toString())
            self.inceptionv4Thread.doneSignal.connect(self.predictionDone)
            self.inceptionv4Thread.predictionSignal.connect(self.inceptionv4Result)
            self.inceptionv4Thread.frameSignal.connect(self.setFrame_h_w)
            self.inceptionv4Thread.start()
    
    def inceptionv4Result(self, img, qimg, stats, currentFrame):
        print(stats)
        pass

    def yolov3Thread(self):
        if self.yolov3ThreadCreated == False:
            self.yolov3ThreadCreated = True
            self.yolov3Thread = YOLOv3(self.importedVideoPath.toString())
            self.yolov3Thread.doneSignal.connect(self.predictionDone)
            self.yolov3Thread.predictionSignal.connect(self.yolov3Result)
            self.yolov3Thread.frameSignal.connect(self.setFrame_h_w)
            self.yolov3Thread.start()
        elif self.yolov3Thread.isFinished():
            self.yolov3Thread = YOLOv3(self.importedVideoPath.toString())
            self.yolov3Thread.doneSignal.connect(self.predictionDone)
            self.yolov3Thread.predictionSignal.connect(self.yolov3Result)
            self.yolov3Thread.frameSignal.connect(self.setFrame_h_w)
            self.yolov3Thread.start()

    def yolov3Result(self, img, qimg, stats, currentFrame):
        pass

    def predictionDone(self):
        self.statusBar().showMessage('Status: Video Detection is done')

    def saveVideo(self):

        try: 
            videopath = os.path.dirname(os.path.realpath(__file__)) + '/../out/{}_predicted.mp4'.format(self.importedVideo)
            if self.playbackThread.isFinished():
                if self.saveVideoThreadCreated == False:
                    self.saveVideoThreadCreated = True
                    self.saveVideoThread = SaveVideo(videopath, self.detectedFrames, self.frame_w, self.frame_h)
                    self.saveVideoThread.doneSignal.connect(self.videoSaved)
                    self.saveVideoThread.start()
                    self.saveVideoThread.Pause = True
                elif self.saveVideoThread.isFinished():
                    self.saveVideoThread = SaveVideo(videopath, self.detectedFrames, self.frame_w, self.frame_h)
                    self.saveVideoThread.doneSignal.connect(self.videoSaved)
                    self.saveVideoThread.start()
            else: 
                QMessageBox.critical(self, "Error", "Video is still running", buttons=QMessageBox.Ok)
        except Exception as e: 
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", "No available frames", buttons=QMessageBox.Ok)

    def videoSaved(self):
        self.statusBar().showMessage('Status: Video saved')

    def exportCSV(self):
        print("csv file will be exported")
        if not self.statistics:
            QMessageBox.critical(self, "Error", "No statistics available", buttons=QMessageBox.Ok)
        else:
            with open("{}.csv".format(self.importedVideo), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['File', 'PredictionString'])
                for i in range(len(self.statistics)):
                    writer.writerow(self.statistics[i])
                    self.statusBar().showMessage('Status: statistics exported')

    @staticmethod
    def setupVideo(videoWidget):
        video = QMediaPlayer(None, QMediaPlayer.StreamPlayback)
        video.setVideoOutput(videoWidget)
        video.setNotifyInterval(1)
        return video

    def positionChanged(self, position):
        self.videoSlider.setValue(position)

    def setPosition(self, position):
        self.video.setPosition(position)

    def handleLabel(self):
        self.durationLabel.clear()
        mtime = QTime(0,0,0,0)
        self.time = mtime.addMSecs(self.video.position())
        self.durationLabel.setText(self.time.toString())

    def resetSlider(self):
        self.videoSlider.setValue(0)

    def durationChanged(self, duration):
        seconds = (duration/1000) % 60
        minutes = (duration/60000) % 60
        hours = (duration/3600000) % 24
        self.durationLabel.setText(QTime(hours, minutes,seconds).toString())
        self.videoSlider.setRange(0, duration)

    def playVideo(self):
        try:
            if self.importedVideoPath is not None and self.importedCSVPath is not None:
                if self.playbackThreadCreated == False:
                    self.playbackThreadCreated = True
                    self.playLoadedVideo()
                    self.playbackThread = PlayBack(self.importedVideoPath.toString(), self.importedCSVPath)
                    self.playbackThread.imageSignal.connect(self.showimg)
                    self.playbackThread.frameSignal.connect(self.setFrame_h_w)
                    self.playbackThread.start()
                elif self.playbackThread.isFinished():
                    self.playLoadedVideo()
                    self.playbackThread = PlayBack(self.importedVideoPath.toString(), self.importedCSVPath)
                    self.playbackThread.imageSignal.connect(self.showimg)
                    self.playbackThread.frameSignal.connect(self.setFrame_h_w)
                    self.playbackThread.start()
                elif not self.playbackThread.isFinished():
                    self.playLoadedVideo()
                    self.playbackThread.Pause = not self.playbackThread.Pause
            else :
                QMessageBox.critical(self, "Error", "Load a CSV file or process video to train", buttons=QMessageBox.Ok)
        except Exception as e: 
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", "Load a CSV file or process video to train", buttons=QMessageBox.Ok)

    def playLoadedVideo(self):
        self.videoWidget.resize(431, 206)
        if self.video.state() == QMediaPlayer.PlayingState: self.video.pause()
        else: self.video.play()

    def setFrame_h_w(self, frame_w, frame_h):
        self.statusBar().showMessage('Status: playing back')
        self.frame_w = frame_w
        self.frame_h = frame_h

    def showimg(self, img, qimage, stats, currentFrame):
        self.currentFrame = currentFrame
        self.detectedFrames.append(img)
        self.detectedStats.append(stats)
        self.removeAllItemsFromTable(self.statView)
        for i, row in enumerate(stats):
            self.addItemToTable(self.statView, row)

        pixmap = QPixmap(qimage)
        self.trainedVideoLabel.setPixmap(pixmap)

    def stopVideo(self):
        self.videoSlider.setValue(0)
        self.video.stop()

    def resetVideo(self):
        if self.video.state() == QMediaPlayer.StoppedState:
            self.stopVideo()
            self.resetSlider()

    def selectedData(self):
        index = self.explorerView.selectionModel().currentIndex().row()
        item_type = self.explorerView.item(index, 2).text()
        item_name = self.explorerView.item(index, 0).text()
        print(item_type)
        if item_type == "video" or item_type == "img": return self.importedVideoPath, True
        else: return self.importedCSVPath, False

    def getDLM(self):
        print("Index: {} - Algorithm: {}".format(self.dlmOptions.currentIndex(), self.dlmOptions.currentText()))
        return self.dlmOptions.currentIndex()

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.playVideo()
        elif event.key() == Qt.Key_S:
            self.playbackThread.terminate()
            self.trainedVideoLabel.clear()
            self.stopVideo()
            self.resetSlider()