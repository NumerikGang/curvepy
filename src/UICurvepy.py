import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QTabWidget, QVBoxLayout, \
    QGridLayout, QSlider, QGroupBox, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot


class App(QMainWindow):

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
    def __init__(self, parent):
        fig, self.ax = plt.subplots(figsize=(5, 4))
        super().__init__(fig)
        self.setParent(parent)

        # Mathplotlib code
        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)

        self.ax.plot(t, s)

        self.ax.set(xlabel='time (s)', ylabel='voltage (mV)',
                    title='About as simple as it gets, folks')
        self.ax.grid()


class MyOptionWidget(QWidget):
    def __init__(self, parent, name: str):
        super(QWidget, self).__init__(parent)
        self.points_cnt_slider = QSlider(Qt.Horizontal)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel(name))
        self.layout.addWidget(self.create_group(self.points_cnt_slider))
        self.layout.addWidget(QPushButton('Calculate Curve'))

    def create_group(self, slider: QSlider):
        group_box = QGroupBox('Number of Points')

        # Create Slider with max/min cnt for Points
        slider.setMinimum(100)
        slider.setMaximum(1000)
        slider.setTickPosition(QSlider.TicksBelow)  # Show rugs at bottom of slider
        slider.setTickInterval(100)  # rug interval

        # Create Labels
        label1 = QLabel('Selected Number')
        label1.setAlignment(Qt.AlignCenter)
        label2 = QLabel()
        label2.setAlignment(Qt.AlignCenter)
        label2.setNum(100)  # set initial number to be displayed

        slider.valueChanged.connect(label2.setNum)  # dynamically change selected Number in Label

        # Add labels and slider to layout
        layout = QVBoxLayout()
        layout.addWidget(slider)
        layout.addWidget(label1)
        layout.addWidget(label2)

        group_box.setLayout(layout)

        return group_box


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
        for i in range(1, self.tabs_cnt):
            self.create_tab(self.tab_list[i],)  # Later change tab name for more specific naming

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def create_tab(self, tab, tab_name: str = 'Options for curve'):
        tab.layout = QGridLayout()
        figure = MyPlotCanvas(tab)  # Add plot to tab
        figure2 = MyOptionWidget(tab, tab_name)  # Add options window to tab
        tab.layout.addWidget(figure, 1, 1)
        tab.layout.addWidget(figure2, 1, 2)
        tab.setLayout(tab.layout)

    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
