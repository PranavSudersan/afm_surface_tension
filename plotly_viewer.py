import numpy as np
import plotly.graph_objs as go
import plotly.offline
from datetime import datetime
##import pandas as pd


import os, sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QHBoxLayout
from PyQt5.QtCore import QUrl, QFileInfo, pyqtSlot
from PyQt5 import QtWebEngineWidgets


class PlotlyViewer(QWidget):
    def __init__(self, fig, exec=True, parent=None):
        # Create a QApplication instance or use the existing one if it exists
        self.app = QApplication.instance() if QApplication.instance() \
                   else QApplication(sys.argv)
        super(PlotlyViewer, self).__init__(parent)
        QtWebEngineWidgets.QWebEngineProfile.defaultProfile().downloadRequested.connect(
            self.on_downloadRequested
        )

        self.view = QtWebEngineWidgets.QWebEngineView()
##        url = "https://domain/your.csv"
##        self.view.load(QtCore.QUrl(url))

        timestamp = datetime.today().strftime('%Y%m%d%H%M%S')
        self.file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'_temp_plot-{timestamp}.html'))
        plotly.offline.plot(fig, filename=self.file_path, auto_open=False)
        self.view.load(QUrl.fromLocalFile(self.file_path))
        self.setWindowTitle("Plotly Viewer")
        
        hbox = QHBoxLayout(self)
        hbox.addWidget(self.view)
        self.show()

        if exec:
            self.app.exec_()
            #sys.exit(self.app.exec_())

    @pyqtSlot("QWebEngineDownloadItem*")
    def on_downloadRequested(self, download):
        old_path = download.url().path()  # download.path()
        suffix = QFileInfo(old_path).suffix()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save File", old_path, "*." + suffix
        )
        if path:
            download.setPath(path)
            download.accept()
            
    def closeEvent(self, event):
        os.remove(self.file_path)

def except_hook(cls, exception, traceback): #display error message/print traceback
    
    trace = 'Traceback:\n' + ''.join(tb.extract_tb(traceback).format())   
    print(cls.__name__ + ':' + str(exception) + '\n\n' + trace)
    sys.__excepthook__(cls, exception, traceback)

if __name__ == "__main__":
    fig = go.Figure()
    fig.add_scatter(x=np.random.rand(100), y=np.random.rand(100), mode='markers',
                    marker={'size': 30, 'color': np.random.rand(100), 'opacity': 0.6,
                            'colorscale': 'Viridis'})
    #app = QApplication(sys.argv)
    w = PlotlyViewer(fig)
    
##fig = go.Figure()
##fig.add_scatter(x=np.random.rand(100), y=np.random.rand(100), mode='markers',
##                marker={'size': 30, 'color': np.random.rand(100), 'opacity': 0.6,
##                        'colorscale': 'Viridis'})
##win = PlotlyViewer(fig)
##
### Read data from a csv
##z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
##
##fig = go.Figure(data=[go.Surface(z=z_data.values)])
##
##fig.update_layout(title='Mt Bruno Elevation', autosize=False,
##                  width=500, height=500,
##                  margin=dict(l=65, r=50, b=65, t=90))
##
##win = PlotlyViewer(fig)
