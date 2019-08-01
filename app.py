from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QFileInfo, QDir, QUrl, QThread, QTime
from PyQt5.QtWidgets import QWidget, QLabel, QMainWindow, QPushButton, QFileDialog
from PyQt5.QtWidgets import QHBoxLayout, QGridLayout, QComboBox, QAction, QSlider
from PyQt5.QtWidgets import QMessageBox, QAbstractItemView, QTableWidgetItem, QTableWidget, QTableView


class App(QMainWindow):

    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        self.totalDolphins = 0
        self.totalSurfers = 0
        self.videosPath = []
        self.initGUI()
        self.menuBar()

    def initGUI(self):
        self.setWindowTitle("Object Detection App")

        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)

        self.buttonsWidget1 = QWidget()
        self.buttonsWidget1Layout = QHBoxLayout(self.buttonsWidget1)
        # Load data btn
        self.loadDataBtn = QPushButton('Load Data', self)
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
        self.saveVideoBtn.clicked.connect(self.saveVideo)
        # import csv btn
        self.exportCSVBtn = QPushButton('Export CSV', self)
        self.exportCSVBtn.clicked.connect(self.exportCSV)
        self.buttonsWidget2Layout.addWidget(self.saveVideoBtn)
        self.buttonsWidget2Layout.addWidget(self.exportCSVBtn)

        self.spaceLabel = QLabel("                       ", self)
        self.inputLabel = QLabel("Input Video/Image", self)
        self.playerLabel = QLabel("Player", self)

        # Video
        self.videoWidget = self.setupVideoWidget(width=400, height=350)
        self.trainedVideoWidget = self.setupVideoWidget(width=400, height=350)
        self.video = self.setupVideo(self.videoWidget)
        self.trainedVideo = self.setupVideo(self.trainedVideoWidget)

        self.videoBtnsWidget = QWidget()
        self.videoBtnsWidgetLayout = QHBoxLayout(self.videoBtnsWidget)
        # playe video btn
        self.playVideoBtn = QPushButton('Play', self)
        self.playVideoBtn.clicked.connect(self.playVideo)
        # pause video btn
        self.pauseVideoBtn = QPushButton('Pause', self)
        self.pauseVideoBtn.clicked.connect(self.pauseVideo)
        # video slider and duration
        self.durationLabel = QLabel("00:00", self)
        self.videoSlider = QSlider(Qt.Horizontal)
        self.videoBtnsWidgetLayout.addWidget(self.playVideoBtn)
        self.videoBtnsWidgetLayout.addWidget(self.pauseVideoBtn)
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
        # File menu
        file_menu = menubar.addMenu(' &App')
        version_action = QAction(' &Version', self)
        quit_action = QAction(' &Quit', self)
        file_menu.addAction(quit_action)
        file_menu.addAction(version_action)
        # Help menu
        help_menu = menubar.addMenu(' &Help')
        about_action = QAction(' &About ', self)
        team_action = QAction(' &The team', self)
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

    def createTable(self, headers, isGrid, isVisible, height=None, minHeight=None):
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

    def addItemToTable(self, model, data):
        currentRow = model.rowCount()
        model.insertRow(currentRow)
        for i, item in enumerate(data):
            model.setItem(currentRow , i, QTableWidgetItem(item))

    def addFilesToExplorer(self, files, fileType, scaler, extensionTag):
        for i in range(len(files)):
            if files[i]:
                info = QFileInfo(files[i])
                self.videosPath.append(QUrl.fromLocalFile(files[i]))
                size = str(info.size()/scaler)+extensionTag
                last_modified = info.lastModified().toString()[4:10]
                self.addItemToTable(self.explorerView, [info.baseName(), size, fileType, last_modified])

    def openFile(self, fileType):
        options = QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "","", "{0} files (*.{0})".format(fileType), QDir.currentPath(), options=options)
        return files

    def loadData(self):
        self.addFilesToExplorer(self.openFile('mp4'), 'Video', 1000000, 'MB')

    def importCSV(self):
        self.addFilesToExplorer(self.openFile('csv'), 'csv', 1000, 'KB')

    def setMedia(self, path):
        content = QMediaContent(path)
        self.video.setMedia(content)
        self.trainedVideo.setMedia(content)

    def processData(self):
        if self.selectedVideo(): 
            path = self.selectedVideo()
            self.setMedia(path)
            print("train in {} algorithm".format(self.getDLM()))

    def saveVideo(self):
        print("video will be saved")

    def exportCSV(self):
        print("csv file will be exported")

    def setupVideoWidget(self, width=600, height=400):
        videoWidget = QVideoWidget()
        videoWidget.setMinimumWidth(width)
        return videoWidget

    def setupVideo(self, videoWidget):
        video = QMediaPlayer()
        video.setVideoOutput(videoWidget)
        # period of time that the change of position is notified
        video.setNotifyInterval(1)
        video.positionChanged.connect(self.positionChanged)
        video.durationChanged.connect(self.durationChanged)
        return video

    def positionChanged(self, position):
        self.videoSlider.setValue(position)

    def durationChanged(self, duration):
        seconds = (duration/1000) % 60
        minutes = (duration/60000) % 60
        hours = (duration/3600000) % 24
        self.durationLabel.setText(QTime(hours, minutes,seconds).toString())
        self.videoSlider.setRange(0, duration)

    def playVideo(self):
        self.video.play()
        self.trainedVideo.play()

    def selectedVideo(self):
        try:
            index = self.explorerView.selectionModel().selectedRows()
            return self.videosPath[index[0].row()]
        except IndexError:
            QMessageBox.critical(self, "Error", "Select a file from explorer", buttons=QMessageBox.Ok)

    def pauseVideo(self):
        self.video.pause()
        self.trainedVideo.pause()

    def getDLM(self):
        return "Indx: {} - Algorithm: {}".format(self.dlmOptions.currentIndex(), self.dlmOptions.currentText())