# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TMsimulGUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TransferMatrixModeling(object):
    def setupUi(self, TransferMatrixModeling):
        TransferMatrixModeling.setObjectName("TransferMatrixModeling")
        TransferMatrixModeling.resize(958, 674)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(TransferMatrixModeling.sizePolicy().hasHeightForWidth())
        TransferMatrixModeling.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(TransferMatrixModeling)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.scrollArea_stack = QtWidgets.QScrollArea(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea_stack.sizePolicy().hasHeightForWidth())
        self.scrollArea_stack.setSizePolicy(sizePolicy)
        self.scrollArea_stack.setWidgetResizable(True)
        self.scrollArea_stack.setObjectName("scrollArea_stack")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 902, 400))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName("verticalLayout")
        self.scrollArea_stack.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_2.addWidget(self.scrollArea_stack, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.frame_2, 1, 0, 1, 1)
        self.frame_buttons = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_buttons.sizePolicy().hasHeightForWidth())
        self.frame_buttons.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(170, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 85, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        self.frame_buttons.setPalette(palette)
        self.frame_buttons.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_buttons.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_buttons.setObjectName("frame_buttons")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_buttons)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_AddLayer = QtWidgets.QPushButton(self.frame_buttons)
        self.pushButton_AddLayer.setObjectName("pushButton_AddLayer")
        self.gridLayout.addWidget(self.pushButton_AddLayer, 0, 2, 1, 1)
        self.pushButton_DeleteLayer = QtWidgets.QPushButton(self.frame_buttons)
        self.pushButton_DeleteLayer.setObjectName("pushButton_DeleteLayer")
        self.gridLayout.addWidget(self.pushButton_DeleteLayer, 1, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.frame_buttons)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 3, 4, 1)
        self.pushButton_CheckNK = QtWidgets.QPushButton(self.frame_buttons)
        self.pushButton_CheckNK.setObjectName("pushButton_CheckNK")
        self.gridLayout.addWidget(self.pushButton_CheckNK, 1, 8, 1, 1)
        self.pushButton_LoadStack = QtWidgets.QPushButton(self.frame_buttons)
        self.pushButton_LoadStack.setObjectName("pushButton_LoadStack")
        self.gridLayout.addWidget(self.pushButton_LoadStack, 0, 8, 1, 1)
        self.pushButton_SaveStack = QtWidgets.QPushButton(self.frame_buttons)
        self.pushButton_SaveStack.setObjectName("pushButton_SaveStack")
        self.gridLayout.addWidget(self.pushButton_SaveStack, 0, 9, 1, 1)
        self.spinBox_StartWave = QtWidgets.QSpinBox(self.frame_buttons)
        self.spinBox_StartWave.setMaximum(10000)
        self.spinBox_StartWave.setProperty("value", 300)
        self.spinBox_StartWave.setObjectName("spinBox_StartWave")
        self.gridLayout.addWidget(self.spinBox_StartWave, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.frame_buttons)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.pushButton_StartSimul = QtWidgets.QPushButton(self.frame_buttons)
        self.pushButton_StartSimul.setObjectName("pushButton_StartSimul")
        self.gridLayout.addWidget(self.pushButton_StartSimul, 3, 0, 1, 2)
        self.spinBox_EndWave = QtWidgets.QSpinBox(self.frame_buttons)
        self.spinBox_EndWave.setMaximum(10000)
        self.spinBox_EndWave.setProperty("value", 1200)
        self.spinBox_EndWave.setObjectName("spinBox_EndWave")
        self.gridLayout.addWidget(self.spinBox_EndWave, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame_buttons)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.pushButton_Help = QtWidgets.QPushButton(self.frame_buttons)
        self.pushButton_Help.setObjectName("pushButton_Help")
        self.gridLayout.addWidget(self.pushButton_Help, 1, 9, 1, 1)
        self.pushButton_ReorderStack = QtWidgets.QPushButton(self.frame_buttons)
        self.pushButton_ReorderStack.setObjectName("pushButton_ReorderStack")
        self.gridLayout.addWidget(self.pushButton_ReorderStack, 3, 8, 1, 1)
        self.frame_3 = QtWidgets.QFrame(self.frame_buttons)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_4.setContentsMargins(1, 1, 1, 1)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.checkBox_1D = QtWidgets.QCheckBox(self.frame_3)
        self.checkBox_1D.setObjectName("checkBox_1D")
        self.gridLayout_4.addWidget(self.checkBox_1D, 0, 0, 1, 1)
        self.checkBox_2D = QtWidgets.QCheckBox(self.frame_3)
        self.checkBox_2D.setObjectName("checkBox_2D")
        self.gridLayout_4.addWidget(self.checkBox_2D, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.frame_3, 3, 2, 1, 1)
        self.gridLayout_3.addWidget(self.frame_buttons, 0, 0, 1, 1)
        TransferMatrixModeling.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(TransferMatrixModeling)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 958, 31))
        self.menubar.setObjectName("menubar")
        TransferMatrixModeling.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(TransferMatrixModeling)
        self.statusbar.setObjectName("statusbar")
        TransferMatrixModeling.setStatusBar(self.statusbar)

        self.retranslateUi(TransferMatrixModeling)
        QtCore.QMetaObject.connectSlotsByName(TransferMatrixModeling)

    def retranslateUi(self, TransferMatrixModeling):
        _translate = QtCore.QCoreApplication.translate
        TransferMatrixModeling.setWindowTitle(_translate("TransferMatrixModeling", "Transfer Matrix Modeling"))
        self.pushButton_AddLayer.setText(_translate("TransferMatrixModeling", "Add Layer"))
        self.pushButton_DeleteLayer.setText(_translate("TransferMatrixModeling", "Delete Layer"))
        self.label_3.setText(_translate("TransferMatrixModeling", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><img src=\":/SunPic/soleil - Copy.jpg\" /></p></body></html>"))
        self.pushButton_CheckNK.setText(_translate("TransferMatrixModeling", "Check nk"))
        self.pushButton_LoadStack.setText(_translate("TransferMatrixModeling", "Load Cell Stack"))
        self.pushButton_SaveStack.setText(_translate("TransferMatrixModeling", "Save Cell Stack"))
        self.label.setText(_translate("TransferMatrixModeling", "StartWave"))
        self.pushButton_StartSimul.setText(_translate("TransferMatrixModeling", "Start Simulation"))
        self.label_2.setText(_translate("TransferMatrixModeling", "EndWave"))
        self.pushButton_Help.setText(_translate("TransferMatrixModeling", "Help/Info/Credits"))
        self.pushButton_ReorderStack.setText(_translate("TransferMatrixModeling", "Reorder"))
        self.checkBox_1D.setText(_translate("TransferMatrixModeling", "1D"))
        self.checkBox_2D.setText(_translate("TransferMatrixModeling", "2D"))

import ressourcefileSunPic_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    TransferMatrixModeling = QtWidgets.QMainWindow()
    ui = Ui_TransferMatrixModeling()
    ui.setupUi(TransferMatrixModeling)
    TransferMatrixModeling.show()
    sys.exit(app.exec_())

