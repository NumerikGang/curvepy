import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QTabWidget, QVBoxLayout, \
    QGridLayout, QSlider, QGroupBox, QLabel, QFileDialog
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5 import QtGui
import utilities as u


class MyMainApp(QMainWindow):

    def __init__(self, height: int = 600, width: int = 800):
        super().__init__()
        # set the starting position of the window
        self.left = 0
        self.top = 0
        # set dimension of window
        self.width = width
        self.height = height
        self.setWindowTitle('Curvepy')  # set title of main window
        # initialize window with given Parameters
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Add tabs to main window
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)

        self.show()


class MyPlotCanvas(FigureCanvas):
    def __init__(self, parent, title: str = ""):
        fig, self.ax = plt.subplots(figsize=(5, 4))
        super().__init__(fig)
        self.setParent(parent)
        self.title = title

        # Make dummy plot to show something
        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)

        self.ax.plot(t, s)

        self.ax.set(xlabel='x', ylabel='y',
                    title=self.title)
        self.ax.grid()

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value: str):
        self._title = value

    def plot_calculated_curve(self, data):  # used to display curved calculated by corresponding algorithm
        pass


class MyOptionWidget(QWidget):
    def __init__(self, parent,  curve_type: u.CurveTypes, plot_canvas):
        super(QWidget, self).__init__(parent)

        self.plot_canvas = plot_canvas

        self.names = ['BezierCurve', 'BezierCurveThreaded', 'BezierCurveBlossoms']

        self.sld_points_cnt = QSlider(Qt.Horizontal)

        self.btn_file_select = QPushButton('select file')
        self.btn_file_select.setToolTip('Select file containing points which can be used to calculate curve')
        self.btn_file_select.clicked.connect(self.on_click_file_select)
        self.btn_calculate_curve = QPushButton('Calculate  ' + self.names[curve_type.value])
        self.btn_calculate_curve.setToolTip('Initiate curve calculation using the algorithm')
        self.btn_calculate_curve.clicked.connect(self.on_click_calculate_curve)

        self.lbl_selected_file = QLabel('No file selected')
        self.lbl_title = QLabel('Options:  ' + self.names[curve_type.value])
        self.lbl_title.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))

        self.filename = None

        # creating Layout for options
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.lbl_title)
        self.layout.addWidget(self.btn_file_select)
        self.layout.addWidget(self.lbl_selected_file)
        self.layout.addWidget(self.create_group(self.sld_points_cnt, 'Number of Points'))
        self.layout.addWidget(self.btn_calculate_curve)

    def create_group(self, slider: QSlider, name: str):
        group_box = QGroupBox(name)

        # Create Slider with max/min cnt for Points
        slider.setMinimum(100)
        slider.setMaximum(1000)
        slider.setTickPosition(QSlider.TicksBelow)  # Show rugs at bottom of slider
        slider.setTickInterval(100)  # rug interval

        # Create Labels
        label1 = QLabel('Selected Number:')
        label2 = QLabel()
        label2.setNum(100)  # set initial number to be displayed

        slider.valueChanged.connect(label2.setNum)  # dynamically change selected Number in Label

        # Create Parent Layout
        layout = QVBoxLayout()
        layout.addWidget(slider)

        # Create Layout for labels
        lbl_layout = QGridLayout()
        lbl_layout.addWidget(label1, 2, 1)
        lbl_layout.addWidget(label2, 2, 2)

        # add label layout to master layout
        layout.addLayout(lbl_layout)

        group_box.setLayout(layout)

        return group_box

    # Events for Buttons
    @pyqtSlot()
    def on_click_file_select(self):
        # select file, so we can use it to calculate curves
        self.filename, _ = QFileDialog.getOpenFileName(self, "Choose File containing data")
        if len(self.filename) == 0:
            self.lbl_selected_file.setText("No file selected")
        else:
            self.lbl_selected_file.setText(self.filename)

    @pyqtSlot()
    def on_click_calculate_curve(self):
        print("Does Something")


class MyTableWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.tabs_cnt = 4
        self.tabs_titles = ['Welcome', 'BezierCurve', 'BezierCurveThreaded', 'BezierCurveBlossoms']

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab_list = [QWidget() for x in range(self.tabs_cnt)]

        # Add tabs
        for tab, title in zip(self.tab_list, self.tabs_titles):
            self.tabs.addTab(tab, title)

        # Create Tabs
        for i, t in zip(range(1, self.tabs_cnt), u.CurveTypes):
            print(t)
            self.create_tab(self.tab_list[i], t)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def create_tab(self, tab, tab_type: u.CurveTypes):
        tab.layout = QGridLayout()
        plot = MyPlotCanvas(tab, tab_type.name)  # Add plot to tab
        option_window = MyOptionWidget(tab, tab_type, plot)  # Add options window to tab
        tab.layout.addWidget(plot, 1, 1)
        tab.layout.addWidget(option_window, 1, 2)
        tab.setLayout(tab.layout)

    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyMainApp()
    sys.exit(app.exec_())

