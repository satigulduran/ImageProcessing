# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:18:30 2020

@author: Satgu
"""


import sys
from PyQt5 import QtWidgets
from finalodevicod import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()