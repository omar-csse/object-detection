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
import time
import datetime

from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QFileInfo, QDir, QUrl, QTime
from PyQt5.QtWidgets import QWidget, QLabel, QMainWindow, QPushButton, QFileDialog
from PyQt5.QtWidgets import QHBoxLayout, QGridLayout, QComboBox, QAction
from PyQt5.QtWidgets import QMessageBox, QAbstractItemView, QTableWidgetItem, QTableWidget, QTableView
from PyQt5.QtGui import QPixmap

from gui.slider import Slider
from gui.button import Button
from gui.exportCSV import ExportCSV
from gui.playback import PlayBack
from gui.saveVideo import SaveVideo
from yolov3.yolo3 import YOLOv3
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

        self.playbackThread = None
        self.saveVideoThread = None
        self.yolov3Thread = None
        self.inceptionv4Thread = None
        self.exportCSVThread = None

        self.labels = None
        self.classes = []
        self.expLabels = None
        self.detectedFrames = []
        self.detectedStats = []

        self.canSaveVideo = True
        self.detecting = False
        self.playingback = False
        self.minutes = 0

        self.initGUI()
        self.menu()

    def initGUI(self):
        
        self.statusBar().showMessage('Status: Ready')
        self.setWindowTitle("Object Detection App")
        self.setFocusPolicy(Qt.StrongFocus)

        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        self.setFixedSize(900, 650)

        self.buttonsWidget1 = QWidget()
        self.buttonsWidget1Layout = QHBoxLayout(self.buttonsWidget1)
        # Load data btn
        self.loadDataBtn = QPushButton('Load Data', self)
        self.loadDataBtn.released.connect(self.loadData)
        # import csv btn
        self.importCSVBtn = QPushButton('Import CSV', self)
        self.importCSVBtn.released.connect(self.importCSV)
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
        self.processDataBtn.released.connect(self.processData)

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
        self.saveVideoBtn.released.connect(self.saveVideo)
        # import csv btn
        self.exportCSVBtn = QPushButton('Export CSV', self)
        self.exportCSVBtn.released.connect(self.exportCSV)
        self.buttonsWidget2Layout.addWidget(self.saveVideoBtn)
        self.buttonsWidget2Layout.addWidget(self.exportCSVBtn)

        self.spaceLabel = QLabel("                       ", self)
        self.inputLabel = QLabel("Input Video/Image", self)
        self.playerLabel = QLabel("Player", self)
        
        # Video
        self.videoWidget = QVideoWidget()
        self.videoWidget.setMaximumSize(400, 300)
        self.video = self.setupVideo(self.videoWidget)
        self.currentVideoState = self.video.state()

        # Pixmap label
        self.trainedVideoLabel = QLabel()
        self.trainedVideoLabel.setMaximumSize(400, 225)
        self.trainedVideoLabel.setScaledContents(True)

        self.video.durationChanged.connect(self.durationChanged)
        self.video.stateChanged.connect(self.resetVideo)

        self.videoBtnsWidget = QWidget()
        self.videoBtnsWidgetLayout = QHBoxLayout(self.videoBtnsWidget)
        # playe video btn
        self.playVideoBtn = QPushButton("Play/Pause")
        self.playVideoBtn.released.connect(self.playVideo)
        # Stop video btn
        self.stopVideoBtn = QPushButton("Stop")
        self.stopVideoBtn.released.connect(self.resetSlider)
        self.stopVideoBtn.released.connect(self.stopVideo)

        self.durationLabel = QLabel('00:00')
        self.videoSlider = Slider()
        self.videoSlider.setEnabled(False)
        self.videoSlider.setOrientation(Qt.Horizontal)
        self.videoSlider.setMinimumWidth(160)
        self.videoSlider.setTickInterval(1)

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
        version_action.setShortcut("Ctrl+V")
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
        guide_action = QAction(' &Guide ', self)
        guide_action.setShortcut("Ctrl+G")
        shortcuts_action = QAction(' &Keyboard Shortcuts ', self)
        shortcuts_action.setShortcut("Ctrl+k")
        help_menu.addAction(about_action)
        help_menu.addAction(team_action)
        help_menu.addAction(guide_action)
        help_menu.addAction(shortcuts_action)

        quit_action.triggered.connect(self.close)
        version_action.triggered.connect(self.version)
        about_action.triggered.connect(self.about)
        team_action.triggered.connect(self.team)
        guide_action.triggered.connect(self.guide)
        shortcuts_action.triggered.connect(self.shortcuts)

    def version(self):
        QMessageBox.about(self, "Version", "Version: v1.0.0")

    def about(self):
        QMessageBox.about(self, "Info", "Object Detection App implemented using YOLOv3 and Inceptionv4 algorithms.")

    def team(self):
        QMessageBox.about(self, "Team", "Aaron Reid\nJiajun Tian\nNathan Mallet\nOmar Alqarni")
    
    def guide(self):
        QMessageBox.about(self, "Guide", "1. Load a Video with the Load Data button\n2. Either Import a preprocessed CSV \
for that video using the Import CSV button (then skip to step 4) or select which Deep Learning Method you want with the DLM Dropdown box\n3. \
Process the Loaded Video with the selected DLM using the Process Data button\n4. View and analyse the video using the \
Play/Pause button, Stop button, and video slider\n5. Save the processed video for later viewing using the \
Save Video button\n6. Export a CSV file containing all of the detection information for the video. This CSV File \
can be used in step 2 if you want to perform detection on the same video again without waiting for it to be processed")
 
    def shortcuts(self):
        QMessageBox.about(self, "Keyboard Shortcuts", "Here is a list of all functional shortcuts for this application:\n\
    Load a Video: Ctrl+O\n    Import a CSV file: Ctrl+I\n    Process a Loaded Video: Ctrl+P\n    Save a Processed Video: Ctrl+S\n    \
Export a Processed CSV file: Ctrl+E\n    Play or Pause a Video: P\n    Stop a Video: S\n    View the Current Version of the \
Application: Ctrl+V\n    View Information About the Application: Ctrl+A\n    View Information about the Programmers: Ctrl+T\n    \
View an end-user Guide for the Application: Ctrl+G\n    View this List of Shortcuts: Ctrl+K\n    Quit the Application: Q")

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
    
    @staticmethod
    def addItemToStats(model, data):
        currentRow = model.rowCount()
        model.insertRow(currentRow)
        for i, item in enumerate(data):
            model.setItem(currentRow , i, QTableWidgetItem(item))

    def removeAllItemsFromTable(self, tableView):
        for i in reversed(range(tableView.rowCount())):
            tableView.removeRow(i)

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

    def openFile(self, fileType, explorerTitle):
        options = QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, explorerTitle,"", "{0} files (*.{0})".format(fileType), QDir.currentPath(), options)
        self.currentVideoPath = fileName
        return fileName

    def loadData(self):
        self.addFilesToExplorer(self.openFile('mp4', 'Open Movie'), 'video', 1000000, 'MB', isVideo=True)

    def importCSV(self):
        self.addFilesToExplorer(self.openFile('csv', 'Import CSV'), 'csv', 1000, 'KB', isVideo=False)

    def readCSV(self, path, baseName):
        try:
            self.importedCSVPath = path
        except:
            QMessageBox.critical(self, "Error", "Invalid CSV file", buttons=QMessageBox.Ok)

    def setMedia(self, path):
        content = QMediaContent(path)
        self.video.setMedia(content)

    def processData(self):
        try:
            # call the two dlms for prediction here:
            if self.playingback is True:
                QMessageBox.critical(self, "Error", "Stop the video first", buttons=QMessageBox.Ok)
                self.statusBar().showMessage('Status: Video is running')
            else:
                if self.detecting is False:
                    indx, txt = self.getDLM()
                    if indx == 0:
                        self.yolov3ThreadInvoke()
                    elif indx == 1:
                        self.inceptionv4ThreadInvoke()
                    print("train in {} algorithm".format(txt))
                    self.statusBar().showMessage('Status: Processing data in {}'.format(txt))
                else:
                    QMessageBox.critical(self, "Error", "Detection in progress...", buttons=QMessageBox.Ok)
                    self.statusBar().showMessage('Status: Detection in progress')
        except (IndexError, AttributeError, TypeError):
            QMessageBox.critical(self, "Error", "Select a file from explorer", buttons=QMessageBox.Ok)
        except (NameError):
            QMessageBox.critical(self, "Error", "Current model is not available", buttons=QMessageBox.Ok)

    def inceptionv4ThreadInvoke(self):
        if self.inceptionv4Thread is None:
            self.detectionStarted()
            self.inceptionv4Thread = InceptionV4(self.importedVideoPath.toString())
            self.inceptionv4Thread.doneSignal.connect(self.inceptionv4threadDone)
            self.inceptionv4Thread.predictionSignal.connect(self.showimg)
            self.inceptionv4Thread.frameSignal.connect(self.setFrame_h_w)
            self.inceptionv4Thread.start()
        elif self.inceptionv4Thread.isFinished():
            self.detectionStarted()
            self.inceptionv4Thread = InceptionV4(self.importedVideoPath.toString())
            self.inceptionv4Thread.doneSignal.connect(self.inceptionv4threadDone)
            self.inceptionv4Thread.predictionSignal.connect(self.showimg)
            self.inceptionv4Thread.frameSignal.connect(self.setFrame_h_w)
            self.inceptionv4Thread.start()

    def yolov3ThreadInvoke(self):
        if self.yolov3Thread is None:
            self.detectionStarted()
            self.yolov3Thread = YOLOv3(self.importedVideoPath.toString())
            self.yolov3Thread.doneSignal.connect(self.yolov3threadDone)
            self.yolov3Thread.predictionSignal.connect(self.showimg)
            self.yolov3Thread.frameSignal.connect(self.setFrame_h_w)
            self.yolov3Thread.start()
        elif self.yolov3Thread.isFinished():
            self.detectionStarted()
            self.yolov3Thread = YOLOv3(self.importedVideoPath.toString())
            self.yolov3Thread.doneSignal.connect(self.yolov3threadDone)
            self.yolov3Thread.predictionSignal.connect(self.showimg)
            self.yolov3Thread.frameSignal.connect(self.setFrame_h_w)
            self.yolov3Thread.start()

    def detectionStarted(self):
        self.clearStats()
        self.detecting = True
        self.canSaveVideo = False

    def inceptionv4threadDone(self, msg, dlm):
        self.threadDone(msg, dlm)
        self.inceptionv4Thread.terminate()

    def yolov3threadDone(self, msg, dlm):
        self.threadDone(msg, dlm)
        self.yolov3Thread.terminate()

    def threadDone(self, msg, dlm=False):
        self.canSaveVideo = True
        self.playingback = False
        if dlm: self.detecting = False
        self.trainedVideoLabel.clear()
        self.removeAllItemsFromTable(self.statView)
        self.statusBar().showMessage("Status: " + msg)
        self.durationLabel.setText("00:00")
        self.videoSlider.setValue(0)
        self.videoSlider.repaint()

    def saveVideo(self):
        try: 
            videopath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'out', '{}_predicted.mp4'.format(self.importedVideo))
            if self.canSaveVideo:
                if self.saveVideoThread is None:
                    self.saveVideoThread = SaveVideo(videopath, self.detectedFrames, self.frame_w, self.frame_h)
                    self.saveVideoThread.doneSignal.connect(self.dataSaved)
                    self.saveVideoThread.errorSignal.connect(self.errorMsg)
                    self.saveVideoThread.start()
                    self.saveVideoThread.Pause = True
                elif self.saveVideoThread.isFinished():
                    self.saveVideoThread = SaveVideo(videopath, self.detectedFrames, self.frame_w, self.frame_h)
                    self.saveVideoThread.doneSignal.connect(self.dataSaved)
                    self.saveVideoThread.errorSignal.connect(self.errorMsg)
                    self.saveVideoThread.start()
            else: 
                QMessageBox.critical(self, "Error", "Video is still running", buttons=QMessageBox.Ok)
        except Exception as e: 
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", "No available frames", buttons=QMessageBox.Ok)

    def dataSaved(self, msg):
        self.statusBar().showMessage(msg)

    def exportCSV(self):
        try:
            if self.canSaveVideo:
                if self.exportCSVThread is None:
                    self.exportCSVThread = ExportCSV(self.importedVideo, self.detectedStats, self.frame_w, self.frame_h)
                    self.exportCSVThread.doneSignal.connect(self.dataSaved)
                    self.exportCSVThread.errorSignal.connect(self.errorMsg)
                    self.exportCSVThread.start()
                elif self.exportCSVThread.isFinished():
                    self.exportCSVThread = ExportCSV(self.importedVideo, self.detectedStats, self.frame_w, self.frame_h)
                    self.exportCSVThread.doneSignal.connect(self.dataSaved)
                    self.exportCSVThread.errorSignal.connect(self.errorMsg)
                    self.exportCSVThread.start()
            else:
                QMessageBox.critical(self, "Error", "No statistics available", buttons=QMessageBox.Ok)
        except Exception as e: 
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", "Unable to export statistics", buttons=QMessageBox.Ok)
            

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

    def resetSlider(self):
        self.videoSlider.setValue(0)

    def durationChanged(self, duration):
        self.videoSlider.setRange(0, duration-1000)

    def playVideo(self):
        try:
            if self.detecting is False:
                if self.importedVideoPath is not None and (self.importedCSVPath is not None or self.detectedStats):
                    if self.playbackThread is None:
                        self.canSaveVideo = False
                        self.playingback = True
                        self.playbackThread = PlayBack(self.importedVideoPath.toString(), self.importedCSVPath, self.detectedStats, self.detectedFrames)
                        self.playbackThread.imageSignal.connect(self.showimg)
                        self.playbackThread.frameSignal.connect(self.setFrame_h_w)
                        self.playbackThread.doneSignal.connect(self.threadDone)
                        self.playbackThread.errorSignal.connect(self.errorMsg)
                        self.playLoadedVideo()
                        self.clearStats()
                        self.playbackThread.start()
                    elif self.playbackThread.isFinished():
                        self.canSaveVideo = False
                        self.playingback = True
                        self.playbackThread = PlayBack(self.importedVideoPath.toString(), self.importedCSVPath, self.detectedStats, self.detectedFrames)
                        self.playbackThread.imageSignal.connect(self.showimg)
                        self.playbackThread.frameSignal.connect(self.setFrame_h_w)
                        self.playbackThread.doneSignal.connect(self.threadDone)
                        self.playbackThread.errorSignal.connect(self.errorMsg)
                        self.playLoadedVideo()
                        self.clearStats()
                        self.playbackThread.start()
                    elif not self.playbackThread.isFinished():
                        self.playLoadedVideo()
                        self.playbackThread.Pause = not self.playbackThread.Pause
                else :
                    QMessageBox.critical(self, "Error", "Load a CSV file or process video to train", buttons=QMessageBox.Ok)
            else:
                QMessageBox.critical(self, "Error", "Detection in progress...", buttons=QMessageBox.Ok)
                self.statusBar().showMessage('Status: Detection in progress')
        except Exception as e: 
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", "Load a CSV file or process video to train", buttons=QMessageBox.Ok)

    def clearStats(self):
        self.detectedFrames.clear()
        self.detectedStats.clear()
        pass

    def errorMsg(self, msg):
        self.playLoadedVideo()
        self.playbackThread = None
        QMessageBox.critical(self, "Error", msg, buttons=QMessageBox.Ok)

    def playLoadedVideo(self):
        self.videoWidget.resize(400, 300)
        if self.video.state() == QMediaPlayer.PlayingState: self.video.pause()
        else: self.video.play()

    def setFrame_h_w(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h

    def handleLabel_Slider(self, sec):
        self.durationLabel.clear()
        self.durationLabel.setText(time.strftime("%M:%S", time.gmtime(sec)))
        pass

    def showimg(self, img, qimage, stats, currentFrame, playback):

        self.totalDolphins = 0
        self.totalSharks = 0
        self.totalSurfers = 0
        
        if currentFrame % 24 == 0:
            self.handleLabel_Slider(int(currentFrame/24))
            self.videoSlider.setValue(int((currentFrame/24)*1000))

        self.currentFrame = currentFrame
        if img: self.detectedFrames.append(img)
        self.detectedStats.append(stats)
        self.removeAllItemsFromTable(self.statView)
        for i, row in enumerate(stats):

            if row[0] == 'dolphin': self.totalDolphins += 1
            elif row[0] == 'shark': self.totalSharks += 1
            elif row[0] == 'surfer': self.totalSurfers += 1
            try:
                if playback is False:
                    row = [str(i+1), row[0], str(int(row[1]*100))+"%", row[2], row[3], row[4], row[5]]
                else: row = [str(i+1)] + row
            except Exception as e: 
                row = [str(i+1)] + row
            self.addItemToStats(self.statView, row)
        
        self.setObjLabel(self.dolphinsLabel, "dolphins", self.totalDolphins)
        self.setObjLabel(self.sharksLabel, "sharks", self.totalSharks)
        self.setObjLabel(self.surfersLabel, "surfers", self.totalSurfers)
        
        pixmap = QPixmap(qimage)
        self.trainedVideoLabel.setPixmap(pixmap)
    
    def setObjLabel(self, label, object_type, num):
        label.setText("Total {}:  {}".format(object_type, num))

    # def nextFrame(self):
    #     a

    # def prevFrame(self):
    #     a

    def stopVideo(self):
        try:
            if self.detecting is False:
                self.video.stop()
                if self.playbackThread is not None: self.playbackThread.stop()
                self.trainedVideoLabel.clear()
                self.removeAllItemsFromTable(self.statView)
            else:
                QMessageBox.critical(self, "Error", "Detection in progress...", buttons=QMessageBox.Ok)
                self.statusBar().showMessage('Status: Detection in progress')
        except Exception as e: 
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", "Load a CSV file or process video to train", buttons=QMessageBox.Ok)

    def resetVideo(self):
        if self.video.state() == QMediaPlayer.StoppedState:
            self.stopVideo()
            self.resetSlider()

    def getDLM(self):
        return self.dlmOptions.currentIndex(), self.dlmOptions.currentText()

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.playVideo()
        elif event.key() == Qt.Key_S:
            self.stopVideo()
            self.resetSlider()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_O: self.loadData()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_I: self.importCSV()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_P: self.processData()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S: self.saveVideo()
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_E: self.exportCSV()