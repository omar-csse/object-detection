import csv
import os
import pathlib
from operator import add, sub

from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QFileInfo, QDir, QUrl, QTime
from PyQt5.QtWidgets import QWidget, QLabel, QMainWindow, QPushButton, QFileDialog
from PyQt5.QtWidgets import QHBoxLayout, QGridLayout, QComboBox, QAction
from PyQt5.QtWidgets import QMessageBox, QAbstractItemView, QTableWidgetItem, QTableWidget, QTableView

from gui.slider import Slider
from gui.button import Button
from gui.frame import Frame

class App(QMainWindow):

    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        self.totalDolphins = 0
        self.totalSurfers = 0
        self.importedCSV = {}
        self.statistics = []
        self.isPlaying = False
        self.importedVideos = {}
        self.framesPath = os.path.dirname(os.path.realpath(__file__)) + '/frames'
        self.captureThreadCreated = False
        self.initGUI()
        self.menu()

    def initGUI(self):

        if not os.path.exists(self.framesPath): 
            os.mkdir(self.framesPath)

        self.statusBar().showMessage('Status: Ready')
        self.setWindowTitle("Object Detection App")
        self.setFocusPolicy(Qt.StrongFocus)

        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)

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
        self.statView = self.createTable("# Object Confidence", False, False, minHeight=160)

        # Total surfers and Total dolphins
        self.surfersLabel = QLabel("Total surfers:  {}".format(self.totalSurfers), self)
        self.dolphinsLabel = QLabel("Total dolphins:  {}".format(self.totalDolphins), self)
        self.dolphinsLabel.setAlignment(Qt.AlignCenter)
        self.surfersLabel.setAlignment(Qt.AlignCenter)

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
        self.trainedVideoWidget = QVideoWidget()
        self.video = self.setupVideo(self.videoWidget)
        self.currentVideoState = self.video.state()
        self.trainedVideo = self.setupVideo(self.trainedVideoWidget)

        self.video.positionChanged.connect(self.positionChanged)
        self.video.positionChanged.connect(self.handleLabel)
        self.video.durationChanged.connect(self.durationChanged)

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
        self.videoSlider.setOrientation(Qt.Horizontal)
        self.videoSlider.setMinimumWidth(160)
        self.videoSlider.setTickInterval(1)
        self.videoSlider.pressed.connect(self.pauseVideo)
        self.videoSlider.released.connect(self.sliderChanged)
        self.videoSlider.sliderMoved.connect(self.setPosition)
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
        mainLayout.addWidget(self.spaceLabel, 12, 0)
        mainLayout.addWidget(self.surfersLabel, 13, 0)
        mainLayout.addWidget(self.dolphinsLabel, 14, 0)
        mainLayout.addWidget(self.buttonsWidget2, 15, 0)
        mainLayout.addWidget(self.spaceLabel, 0, 1)
        mainLayout.addWidget(self.inputLabel, 0, 2)
        mainLayout.addWidget(self.videoWidget, 1, 2, 5, 1)
        mainLayout.addWidget(self.playerLabel, 8, 2)
        mainLayout.addWidget(self.spaceLabel, 9, 2)
        mainLayout.addWidget(self.trainedVideoWidget, 10, 2, 5, 1)
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

    def addFilesToExplorer(self, files, fileType, scaler, extensionTag, isVideo=True):
        print(files)
        for i in range(len(files)):
            if files[i]:
                info = QFileInfo(files[i])
                if info.baseName() in self.importedVideos or info.baseName() in self.importedCSV:
                    QMessageBox.critical(self, "Error", "File already exist", buttons=QMessageBox.Ok)
                else:
                    if (isVideo):
                        self.importedVideos[info.baseName()] = QUrl.fromLocalFile(files[i])
                    else: self.readCSV(info.filePath(), info.baseName())
                    size = str(info.size()/scaler)+extensionTag
                    last_modified = info.lastModified().toString()[4:10]
                    self.addItemToTable(self.explorerView, [info.baseName(), size, fileType, last_modified])
                self.statusBar().showMessage('Status: Ready')      

    def openFile(self, fileType):
        options = QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie","", "{0} files (*.{0})".format(fileType), QDir.currentPath(), options)
        self.currentVideoPath = fileName
        return [fileName]

    def loadData(self):
        self.statusBar().showMessage('Status: Loading Video/Image')
        self.addFilesToExplorer(self.openFile('mp4'), 'Video', 1000000, 'MB', isVideo=True)

    def importCSV(self):
        self.statusBar().showMessage('Status: Importing CSV File')
        self.addFilesToExplorer(self.openFile('csv'), 'csv', 1000, 'KB', isVideo=False)

    def readCSV(self, path, baseName):
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            self.importedCSV[baseName] = list(reader)

    def setMedia(self, path):
        content = QMediaContent(path)
        self.video.setMedia(content)
        self.trainedVideo.setMedia(content)

    @staticmethod
    def setupVideoWidget(width=600, height=400):
        videoWidget = QVideoWidget()
        videoWidget.setMinimumWidth(width)
        return videoWidget

    def processData(self):
        try:
            path, isVideo = self.selectedData()
            if isVideo: self.setMedia(path)
            print("train in {} algorithm".format(self.getDLM()))
            self.statusBar().showMessage('Status: Processing data in {}'.format(self.getDLM()))
        except (IndexError, AttributeError, TypeError):
            QMessageBox.critical(self, "Error", "Select a file from explorer", buttons=QMessageBox.Ok)

    def saveVideo(self):
        print("video will be saved")
        self.statusBar().showMessage('Status: Video saved')

    def exportCSV(self):
        print("csv file will be exported")
        if not self.statistics:
            QMessageBox.critical(self, "Error", "Process the data first", buttons=QMessageBox.Ok)
        else:
            with open("statistics.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
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
        self.trainedVideo.setPosition(position)

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
        self.videoWidget.resize(431, 206)
        self.trainedVideoWidget.resize(431, 224)
        if self.video.state() == QMediaPlayer.PlayingState:
            self.video.pause()
            self.trainedVideo.pause()
        else:
            self.video.play()
            self.trainedVideo.play()

    def stopVideo(self):
        self.video.stop()
        self.trainedVideo.stop()

    def sliderChanged(self):
        if self.currentVideoState == QMediaPlayer.PausedState:
            self.video.pause()
            self.trainedVideo.pause()
        else:
            self.video.play()
            self.trainedVideo.play()

    def changeVideoFrame(self, sign):
        self.trainedVideo.setPosition(sign(self.trainedVideo.position(), 100*60))
        self.video.setPosition(sign(self.video.position(), 100*60))

    def selectedData(self):
        index = self.explorerView.selectionModel().currentIndex().row()
        item_type = self.explorerView.item(index, 2).text()
        item_name = self.explorerView.item(index, 0).text()
        print(item_type)
        if item_type == "Video" or item_type == "Img": return self.importedVideos[item_name], True
        else: return self.importedCSV[item_name], False

    def pauseVideo(self):
        self.currentVideoState = self.video.state()
        self.video.pause()
        self.trainedVideo.pause()

    def getDLM(self):
        return "Index: {} - Algorithm: {}".format(self.dlmOptions.currentIndex(), self.dlmOptions.currentText())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.changeVideoFrame(add)
        elif event.key() == Qt.Key_Left:
            self.changeVideoFrame(sub)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.playVideo()
        elif event.key() == Qt.Key_S:
            self.stopVideo()
            self.resetSlider()
        elif event.key() == Qt.Key_F:
            if self.video.state() == QMediaPlayer.PlayingState or self.video.state() == QMediaPlayer.PausedState:
                if (self.captureThreadCreated == False):
                    self.captureThreadCreated = True
                    self.captureThread = Frame(self.currentVideoPath, self.video.position())
                    self.captureThread.start()
                else:
                    if (self.captureThread.isFinished()):
                        self.captureThread = Frame(self.currentVideoPath, self.video.position())
                        self.captureThread.start()