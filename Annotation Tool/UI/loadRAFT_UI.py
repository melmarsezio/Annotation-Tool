# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'loadRAFT_UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        Dialog.resize(240, 193)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(Dialog)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.pb_loadFW = QtWidgets.QPushButton(Dialog)
        self.pb_loadFW.setObjectName("pb_loadFW")
        self.horizontalLayout.addWidget(self.pb_loadFW)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.FWin = QtWidgets.QPlainTextEdit(Dialog)
        self.FWin.setEnabled(True)
        self.FWin.setMaximumSize(QtCore.QSize(268, 30))
        self.FWin.setFocusPolicy(QtCore.Qt.NoFocus)
        self.FWin.setOverwriteMode(True)
        self.FWin.setObjectName("FWin")
        self.verticalLayout.addWidget(self.FWin)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.pb_loadBW = QtWidgets.QPushButton(Dialog)
        self.pb_loadBW.setObjectName("pb_loadBW")
        self.horizontalLayout_2.addWidget(self.pb_loadBW)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.BWin = QtWidgets.QPlainTextEdit(Dialog)
        self.BWin.setEnabled(True)
        self.BWin.setMaximumSize(QtCore.QSize(268, 30))
        self.BWin.setFocusPolicy(QtCore.Qt.NoFocus)
        self.BWin.setOverwriteMode(True)
        self.BWin.setObjectName("BWin")
        self.verticalLayout.addWidget(self.BWin)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pb_confirm = QtWidgets.QPushButton(Dialog)
        self.pb_confirm.setObjectName("pb_confirm")
        self.horizontalLayout_3.addWidget(self.pb_confirm)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_3.addWidget(self.pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4.addLayout(self.verticalLayout)

        self.retranslateUi(Dialog)
        self.pushButton.clicked.connect(Dialog.close)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Load RAFT"))
        self.label.setText(_translate("Dialog", "Forward:"))
        self.pb_loadFW.setText(_translate("Dialog", "Load"))
        self.FWin.setPlainText(_translate("Dialog", "[NONE]"))
        self.label_2.setText(_translate("Dialog", "Backward:"))
        self.pb_loadBW.setText(_translate("Dialog", "Load"))
        self.BWin.setPlainText(_translate("Dialog", "[NONE]"))
        self.pb_confirm.setText(_translate("Dialog", "Confirm"))
        self.pushButton.setText(_translate("Dialog", "Cancel"))
