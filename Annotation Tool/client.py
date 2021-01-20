from __future__ import print_function

import os
import sys
import getpass
import time
import pickle
import grpc

# PyQt5
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# image operation related
import math
import cv2 as cv
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# UI related
sys.path.append('UI')
from annotationUI import Ui_MainWindow
from cocoUI import Ui_Dialog as coco_Ui_Dialog
from addUI import Ui_Dialog as add_Ui_Dialog
from optFlowUI import Ui_Dialog as optFlow_Ui_Dialog
from loadRAFT_UI import Ui_Dialog as loadRAFT_Ui_Dialog
from configUI import Ui_Dialog as config_Ui_Dialog
# dependent file
sys.path.append('core')
from fileIO import *
from undistortion import *
from coco import Coco
from config import *
import utils
from ScrollBar import ScrollBar
# grpc related
sys.path.append('GRPC')
import openpose_pb2
import openpose_pb2_grpc
import raft_pb2
import raft_pb2_grpc

###### cocoUI ######
class cocoWindow(QDialog,coco_Ui_Dialog):
    def __init__(self, mainWin, parent=None):
        super(cocoWindow, self).__init__(parent)
        self.setupUi(self)
        self.mainWin = mainWin
        self.pb_continue.clicked.connect(self.saveCoco)
        cur = time.localtime()
        self.le_contr.setText(getpass.getuser())
        self.le_year.setText(str(cur.tm_year))
        self.le_dateCr.setText(f'{cur.tm_year}/{cur.tm_mon:02d}/{cur.tm_mday:02d}')

    #save coco file
    def saveCoco(self):
        dlg = QFileDialog()
        path, _ = dlg.getSaveFileName(self, 'Select file directory', './', "COCO files (*.json)")
        if not path:
            return
        
        mainWin = self.mainWin
        coco = Coco(year=int(self.le_year.text()),
                    version=self.le_version.text(),
                    description=self.le_des.text(),
                    contributor=self.le_contr.text(),
                    url=self.le_url.text(),
                    date_created=self.le_dateCr.text())
        cat = coco.getCat(id_=0)
        for idx, frame in enumerate(mainWin.videoCap):
            if mainWin.params["keyPoints"][idx]:
                imgID = coco.addImg(width=frame.shape[0],
                                    height=frame.shape[1],
                                    file_name='frame'+str(idx)+'.jpg',
                                    license_=None,
                                    flickr_url=None,
                                    coco_url=None,
                                    date_captured=None)
                keypoints = []
                for key in cat["keypoints"]:
                    kpKey = mapCOCO2KPs[key]
                    if kpKey in mainWin.params["keyPoints"][idx]:
                        coor = utils.multi(mainWin.params["keyPoints"][idx][kpKey][::-1],frame.shape[:2])
                        keypoints.extend(coor)
                        keypoints.extend([2])
                    else:
                        keypoints.extend([0,0,0])
                coco.addAnn(keypoints=keypoints,
                            num_keypoints=len(mainWin.params["keyPoints"][idx]),
                            image_id=imgID,
                            category_id=0)
        coco.saveToFile(path)
        mainWin.printMsg(f'Save COCO file to {path}')
        print(coco)
        self.close()

    #KeyPress
    def keyPressEvent(self, event):
        if str(event.key()) == '16777220':#Qt.Key_Enter:
            self.saveCoco()

###### addKeyPointUI ######
class addWindow(QDialog, add_Ui_Dialog):
    def __init__(self, mainWin, coor, parent=None):
        super(addWindow, self).__init__(parent)
        self.setupUi(self)
        self.mainWin = mainWin
        self.coor = coor
        self.cb_pointType.clear()
        self.cb_pointType.insertItems(0,[key for key in mapKP2ID if key not in mainWin.params["keyPoints"][mainWin.params["page"]]])
        self.pb_add.clicked.connect(self.add)

    def add(self):
        mainWin = self.mainWin
        type_ = self.cb_pointType.currentText()
        mainWin.params["keyPoints"][mainWin.params["page"]][type_] = self.coor
        mainWin.params["selectKey"] = type_
        mainWin.loadKeyPointList()
        mainWin.progressBar.setValue(mainWin.progressCount())
        mainWin.updateWinTitle(False)
        mainWin.printMsg(f'Key Point "{type_}" is added to the list at ({self.coor[0]:.2f}, {self.coor[1]:.2f}).')
        self.close()

#open grpc tunnel for single frame RAFT calculation
def grpcRAFT(mainWin,prev,next_):
    prev = cv.imencode('.jpg',mainWin.videoCap[prev])[1].tobytes()
    next_ = cv.imencode('.jpg',mainWin.videoCap[next_])[1].tobytes()
    try:
        print('send request')
        with grpc.insecure_channel(mainWin.IP['RAFT']) as channel:
            stub = raft_pb2_grpc.raftDetectStub(channel)
            response = stub.Upload(raft_pb2.Send(Type='frames',
                                                    frameInfo=raft_pb2.frameRAFT(prev=prev,next_=next_,
                                                    r=4-mainWin.params["rotDeg"]//90)))
        if response.status:
            flo = np.frombuffer(response.forw,dtype='float16')
        else:
            mainWin.printMsg('RAFT calculate failed. Consult with the server.')
            return None
    except grpc._channel._InactiveRpcError as e:
        print(e)
        mainWin.printMsg('Cannot connect to the Server. Verify the server\'s IP/Port!')
        return None
    return flo

#add moved keypoints to next frame
def moveNextKPs(mainWin, flo, idx, gap, H, W):
    cur_kps = mainWin.params["keyPoints"][idx-gap]
    next_kps = mainWin.params["keyPoints"][idx]
    move = utils.calcMove(flo,cur_kps,H,W,mainWin.CT)
    for key in cur_kps:
        next_kps[key] = utils.add(cur_kps[key],move[key])

###### opticalFlowUI ######
class optFlowWindow(QDialog, optFlow_Ui_Dialog):
    def __init__(self, mainWin, parent=None):
        super(optFlowWindow, self).__init__(parent)
        self.setupUi(self)
        
        self.mainWin = mainWin
        self.sb_to.setMinimum(1)
        self.sb_to.setMaximum(mainWin.maxpage)
        self.cur = mainWin.params["page"]
        self.sb_to.setValue(self.cur+1)
        self.pb_continue.clicked.connect(self.continue_)

    def continue_(self):
        end = self.sb_to.value()-1
        if end==self.cur:
            reply = QMessageBox.critical(self, "Error", "Can not have same start and end page number!!", QMessageBox.Ok, QMessageBox.Ok)
            return
        type_ = self.cb_type.currentText()
        self.close()

        mainWin = self.mainWin
        direct = 1 if end>self.cur else -1
        W,H,D = mainWin.videoCap[self.cur].shape
        _optFlow = mainWin.RAFT_FW if direct>0 else mainWin.RAFT_BW

        regions = utils.merge(mainWin.ScrollBar.getHightLight())
        offset = int(direct<0)

        for idx in range(self.cur+direct, end+direct, direct):
            print('idx',idx,end='\r')

            if idx in regions:
                continue
            gap = utils.getGap(idx, direct, regions)
            if idx-direct in regions or _optFlow[idx-gap-offset].sum()==0:
                prev=idx-gap
                flo = grpcRAFT(mainWin, prev, idx)
                if flo is None:
                    return
                flo = flo.reshape((H,W,2))
            else:
                flo = _optFlow[idx-gap]
            # print('use',idx-gap,'calc',idx)
            moveNextKPs(mainWin, flo, idx, gap, H, W)
        mainWin.progressBar.setValue(mainWin.progressCount())
        mainWin.updateWinTitle(False)

###### loadRAFT_UI ######
class loadRAFTWindow(QDialog, loadRAFT_Ui_Dialog):
    def __init__(self, mainWin, parent=None):
        super(loadRAFTWindow, self).__init__(parent)
        self.setupUi(self)
        self.mainWin = mainWin
        self.pb_loadFW.clicked.connect(self.readFW)
        self.pb_loadBW.clicked.connect(self.readBW)
        self.pb_confirm.clicked.connect(self.confirm)

    def readFW(self):
        self.read(self.FWin)
    
    def readBW(self):
        self.read(self.BWin)

    def read(self, TextEdit):
        dlg = QFileDialog()
        path,_ = dlg.getOpenFileName(self, 'Open file', './', "optical flow file (*.avi)")
        if path:
            TextEdit.setPlainText(path)
    
    def confirm(self):
        mainWin = self.mainWin
        FWpath = self.FWin.toPlainText()
        BWpath = self.BWin.toPlainText()
        if FWpath!='[NONE]' and BWpath!='[NONE]':
            if FWpath == BWpath:
                reply = QMessageBox.critical(self, "Error", "Can not have same RAFT file for both forward and backward!!", QMessageBox.Ok, QMessageBox.Ok)
            else:
                mainWin.printMsg(f'Start loading RAFT file from {FWpath} and {BWpath}...')
                mainWin.RAFT_FW = RAFT(FWpath)
                mainWin.RAFT_BW = RAFT(BWpath)
                if mainWin.RAFT_FW and mainWin.RAFT_BW and utils.checkRAFT(mainWin.videoCap,mainWin.RAFT_FW,mainWin.RAFT_BW):
                    mainWin.printMsg(f'{FWpath} and {BWpath} are successfully loaded!!')
                else:    
                    mainWin.RAFT_FW = None
                    mainWin.RAFT_BW = None
                    mainWin.printMsg(f'Something is wrong, check if RAFT file is valid.')
                self.close()
        else:
            reply = QMessageBox.critical(self, "Error", "You have not assign forward or backward RAFT file!!", QMessageBox.Ok, QMessageBox.Ok)

###### configUI ######
class configWindow(QDialog,config_Ui_Dialog):
    def __init__(self, mainWin, parent=None):
        super(configWindow, self).__init__(parent)
        self.setupUi(self)
        self.mainWin = mainWin
        self.opIP = [self.OPa,self.OPb,self.OPc,self.OPd,self.OPe]
        self.rfIP = [self.RFa,self.RFb,self.RFc,self.RFd,self.RFe]
        self.VideoDir.setText(mainWin.params["path"])
        self.UndistDir.setText(mainWin.params["undistPath"])
        utils.fillIP(mainWin.IP['openpose'],self.opIP)
        utils.fillIP(mainWin.IP['RAFT'],self.rfIP)

        self.OPS.clicked.connect(self.OPSet)
        self.OPR.clicked.connect(self.OPReset)
        self.RFS.clicked.connect(self.RFSet)
        self.RFR.clicked.connect(self.RFReset)
        self.Video.clicked.connect(self.ChangeVideoPath)
        self.Undist.clicked.connect(mainWin.loadDistortionParameter)
    
    def OPSet(self):
        mainWin = self.mainWin
        IP_str = utils.validIP(self.opIP)
        if IP_str:
            mainWin.IP['openpose'] = IP_str
        else:
            utils.fillIP(mainWin.IP['openpose'],self.opIP)
            reply = QMessageBox.critical(self, "Error", "Invalid IP address!!", QMessageBox.Ok, QMessageBox.Ok)

    def OPReset(self):
        mainWin = self.mainWin
        mainWin.IP['openpose'] = mainWin.IP['default']['openpose']
        utils.fillIP(mainWin.IP['openpose'],self.opIP)

    def RFSet(self):
        mainWin = self.mainWin
        IP_str = utils.validIP(self.rfIP)
        if IP_str:
            mainWin.IP['RAFT'] = IP_str
        else:
            utils.fillIP(mainWin.IP['RAFT'],self.rfIP)
            reply = QMessageBox.critical(self, "Error", "Invalid IP address!!", QMessageBox.Ok, QMessageBox.Ok)

    def RFReset(self):
        mainWin = self.mainWin
        mainWin.IP['RAFT'] = mainWin.IP['default']['RAFT']
        utils.fillIP(mainWin.IP['RAFT'],self.rfIP)

    def ChangeVideoPath(self):
        mainWin = self.mainWin
        dlg = QFileDialog()
        path,_ = dlg.getOpenFileName(self, 'Open file', './', "Video files (*.mpeg *.mov *.wmv *.rmvb *.flv *.mp4 *.avi)")
        if not (path and mainWin.checkUndEnable()):
            return
        mainWin.printMsg(f'Reading video from {path}...')
        mainWin.params["path"] = path
        mainWin.reload()

###### MainUI ######
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__ (self, parent=None):
        #Init window
        super(MyMainWindow, self).__init__(parent)
        self.ScrollBar = ScrollBar() # put this before setupUi because ScrollBar setup background and foreground color
        self.setupUi(self)
        self.IntensityView.setCentralItem(self.ScrollBar)

        self.params = dict() # to save in *.annot file
        #File related variables
        self.params["path"] = None
        self.videoCap = None # Video class import from core.fileIO.Video
        self.RAFT_FW = None # RAFT class import from core.fileIO.RAFT
        self.RAFT_BW = None # RAFT class import from core.fileIO.RAFT
        self.maxpage = 0
        self.params["page"] = 0 #image page
        self.params["savePath"] = None #annotation file directory
        #status related variables
        self.fullScreen = False
        self.ctrl = False
        self.A = False
        self.CT = True
        self.params["rotDeg"] = 0
        self.params["opDeg"] = 270
        self.params["selectKey"] = None #selected key point
        self.pixmap = QPixmap()
        #algorithm related variables
        self.IP = dict() # to save all IP address
        self.IP['default'] = {'openpose':'10.10.192.40:60000','RAFT':'10.10.192.40:60001'}
        self.IP['openpose'] = '10.10.192.40:60000'
        self.IP['RAFT'] = '10.10.192.40:60001'
        self.params["undistPath"] = './core/IntrinsicParameter.yml' #Undistortion parameter file directory
        try:
            self._undist = Undistortion(self.params["undistPath"]) #Undistortion object
            self.UndistAddressWin.setPlainText(self.params["undistPath"])
        except Exception:
            self._undist = None
        self.params["keyPoints"] = [] #keypoints from Openpose Server
        #counter related variables
        self.autoSaveTimer = QTimer() #autosaver
        self.autoSaveTimer.timeout.connect(self.saveAnnot)
        self.connectionTimer = QTimer() #timeout check
        self.connectionTimer.timeout.connect(self.connectTimeout)

        #### Menu ####
        ## File (F) ##
        self.actionLoad_Video.triggered.connect(self.loadVideo)
        self.actionLoad_Images.triggered.connect(self.loadImages)
        self.actionLoad_Annot.triggered.connect(self.loadAnnot)
        #------------------#
        self.actionLoad_Parameter.triggered.connect(self.loadDistortionParameter)
        self.actionLoad_RAFT.triggered.connect(self.loadOptFlow)
        self.actionLoadIntCSV.triggered.connect(self.loadIntCSV)
        #------------------#
        self.actionSave.triggered.connect(self.saveAnnotation)
        self.actionSave_as.triggered.connect(self.saveAsAnnotation)
        self.actionOutput_Coco.triggered.connect(self.saveCoco)
        self.actionOutput_Image.triggered.connect(self.saveImages)
        #------------------#
        self.actionQuit.triggered.connect(self.close)
        ## View (V) ##
        self.actionFullScreen.triggered.connect(self.triggerFullScreen)
        self.actionMaximize.triggered.connect(self.showMaximized)
        self.actionMinimize.triggered.connect(self.showMinimized)
        ## Edit (E) ##
        self.actionAdd_KeyPoint.triggered.connect(self.keyPointAdd)
        self.actionReset_KeyPoint.triggered.connect(self.keyPointReset)
        self.actionDelete_KeyPoint.triggered.connect(self.keyPointDelete)
        self.actionClear_Message.triggered.connect(self.MsgWin.clear)
        ## Setting (S) ##
        self.actionConfig.triggered.connect(self.config)
        self.actionCT_switch.triggered.connect(self.BoundarySwitch)
        #### MainWindow ####
        ## Key Points ##
        self.keyPointList.itemSelectionChanged.connect(self.keyPointSelected)
        ## Buttons ##
        self.pb_AddPoint.clicked.connect(self.keyPointAdd)
        self.pb_Reset.clicked.connect(self.keyPointReset)
        self.pb_DeletePoint.clicked.connect(self.keyPointDelete)
        self.pb_LoadUndist.clicked.connect(self.loadDistortionParameter)
        self.pb_Openpose.clicked.connect(self.Openpose)
        self.pb_CalcOptFlow.clicked.connect(self.getOptFlow)
        self.pb_Prop.clicked.connect(self.optFlow)
        ## Image Window ##
        self.cb_autoSave.stateChanged.connect(self.setAutoSave)
        self.pb_rotateC.clicked.connect(self.rotateClock)
        self.pb_rotateAC.clicked.connect(self.rotateAntiClock)
        ## Slider ##
        self.ScrollBar.valueChanged.connect(self.slide)
        self.sb_Page.valueChanged.connect(self.spinBox)

###### FILE I/O FUNCTION ######
    #load video and convert to image
    def loadVideo(self):
        dlg = QFileDialog()
        path,_ = dlg.getOpenFileName(self, 'Open file', './', "Video files (*.mpeg *.mov *.wmv *.rmvb *.flv *.mp4 *.avi)")
        if not (path and self.checkUndEnable()):
            return
        self.MsgWin.clear()
        self.printMsg(f'Reading video from {path}...')
        self.params["path"] = path
        self.videoCap = Video(path,self.UndistEnable.checkState(),self._undist)
        self.printMsg(f'{path} loaded successfully!!')
        self.reset()
        self.update()

    #load image
    def loadImages(self):
        dlg = QFileDialog()
        imagePathSet,_ = dlg.getOpenFileNames(self, 'Open file', './', "Image files (*.jpg *.gif *.png *.ico)")
        if not (imagePathSet and self.checkUndEnable()):
            return
        path = '/'.join(imagePathSet[0].split('/')[:-1])+'/..'
        self.MsgWin.clear()
        self.printMsg(f'Loading images directly from directory {path}...')
        self.params["path"] = imagePathSet
        self.videoCap = Images(imagePathSet,self.UndistEnable.checkState(),self._undist)
        self.reset()
        self.update()

    #load *.annot file
    def loadAnnot(self):
        dlg = QFileDialog()
        path,_ = dlg.getOpenFileName(self, 'Open file', './', "Annotation files (*.annot)")
        if not path:
            return
        self.MsgWin.clear()
        self.printMsg(f'Loading annotation file from {path}...')
        self.params = pickle.load(open(path,'rb'))
        self.reload()
        if self.videoCap:
            self.printMsg(f'{path} is successfully loaded!!')
        else:
            reply = QMessageBox.warning(self, "Warning", "Parameters successfully loaded, but video failed. Check the existence of video.", QMessageBox.Ok, QMessageBox.Ok)
            self.printMsg(f'Parameters successfully loaded from {path}, but video failed. Check the existence of video.')
        self.update()

    #load RAFT result file
    def loadOptFlow(self):
        if not self.videoCap:
            reply = QMessageBox.critical(self, "Error", "Empty Image Collection!!", QMessageBox.Ok, QMessageBox.Ok)
            return
        dlg = loadRAFTWindow(self)
        dlg.show()
        dlg.exec_()
            
    #save COCO file (json)
    def saveCoco(self):
        if not self.videoCap:
            reply = QMessageBox.critical(self, "Error", "Empty Image Collection!!", QMessageBox.Ok, QMessageBox.Ok)
            return
        dlg = cocoWindow(self)
        dlg.show()
        dlg.exec_()
    
    #output frames
    def saveImages(self):
        if not self.videoCap:
            reply = QMessageBox.critical(self, "Error", "Empty Image Collection!!", QMessageBox.Ok, QMessageBox.Ok)
            return
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        path = dlg.getExistingDirectory(self, "Choose output path", "")
        if path:
            print(path)
            self.printMsg(f'Output image frames to {path}...')
            saveFrame(self.videoCap, path, rot=4-self.params["rotDeg"]//90)
            self.printMsg(f'Image frames output successfully!!')
    
    #save *.annot file
    def saveAnnot(self, auto=True):
        if self.params["savePath"]:
            pickle.dump(self.params,open(self.params["savePath"], 'wb'))
            self.updateWinTitle(True)
            if auto:
                self.printMsg(f'Autosave annotation file to {self.params["savePath"]}.')
    
    #save *.annot to default directory
    def saveAnnotation(self):
        if not self.params["keyPoints"]:
            reply = QMessageBox.critical(self, "Error", "Empty Key Points Set!!", QMessageBox.Ok, QMessageBox.Ok)
            return
        if not self.params["savePath"]:
            dlg = QFileDialog()
            path,_ = dlg.getSaveFileName(self, 'Select file directory', './', "Annotation files (*.annot)")
            if path:
                self.params["savePath"] = path
        if self.params["savePath"]:
            self.saveAnnot(auto=False)
            self.printMsg(f'Save annotation file to {self.params["savePath"]}.')

    #save *.annot to newly defined directory
    def saveAsAnnotation(self):
        if not self.params["keyPoints"]:
            reply = QMessageBox.critical(self, "Error", "Empty Key Points Set!!", QMessageBox.Ok, QMessageBox.Ok)
            return
        dlg = QFileDialog()
        path,_ = dlg.getSaveFileName(self, 'Select file directory', './', "Annotation files (*.annot)")
        if not path:
            return
        self.params["savePath"] = path
        self.saveAnnot(auto=False)
        self.printMsg(f'Save annotation file as {self.params["savePath"]}.')

###### STATUS ######
    def resetWrapper(self):
        temp = self.params["page"]
        self.RAFT_FW,self.RAFT_BW = None,None
        self.params["page"] = temp
        self.ScrollBar.setMaximum(self.maxpage)
        self.ScrollBar.setPos(self.params["page"]+1)
        self.progressBar.setMaximum(self.maxpage)
        self.sb_Page.setEnabled(True)
        self.sb_Page.setMinimum(1)
        self.sb_Page.setMaximum(self.maxpage)
        self.sb_Page.setValue(self.params["page"]+1)
        self.statusbar.showMessage(f'Min #frame: {1}\t\tMax #frame: {self.maxpage}')
        path = '/'.join(self.params["path"][0].split('/')[:-1])+'/..' if isinstance(self.params["path"],list) else self.params["path"]
        self.setWindowTitle(f"SIEMENS Annotation Tool {path}")
        self.UndistAddressWin.setPlainText(self.params["undistPath"])
        self.loadKeyPointList()
        self.progressBar.setValue(self.progressCount())

    #Reset variable
    def reset(self):
        self.maxpage = len(self.videoCap)
        self.params["page"] = 0
        self.params["rotDeg"] = 0
        self.params["opDeg"] = 270
        self.params["selectKey"] = None
        self.params["keyPoints"] = [{} for _ in range(self.maxpage)]
        self.params["savePath"] = None
        self.resetWrapper()

    #Reload *.annot file
    def reload(self):
        try:
            self._undist = Undistortion(self.params["undistPath"])
            self.UndistAddressWin.setPlainText(self.params["undistPath"])
        except Exception:
            self._undist = None
        objectType = Images if isinstance(self.params["path"],list) else Video
        self.videoCap = objectType(self.params["path"],self.UndistEnable.checkState(),self._undist)
        self.maxpage = len(self.videoCap)
        self.resetWrapper()

    #Update title (title showes file directory and save status)
    def updateWinTitle(self, state=True):
        titleList = self.windowTitle().split()
        if state:
            self.setWindowTitle(' '.join(titleList[:-1]))
        elif "[UNSAVED]*" != titleList[-1]:
            self.setWindowTitle(self.windowTitle()+" [UNSAVED]*")

    #Toggle FullScreen/NormalScreen
    def triggerFullScreen(self):
        if self.fullScreen:
            self.showNormal()
            self.actionFullScreen.setText('FullScreen (&F)')
        else:
            self.showFullScreen()
            self.actionFullScreen.setText('Restore (&F)')
        self.fullScreen = not self.fullScreen
    
    #Pop configuration dialog
    def config(self):
        win = configWindow(self)
        win.show()
        win.exec_()

    #Switch CT mode/general model
    def BoundarySwitch(self):
        self.CT = not self.CT
        if self.CT:
            self.actionCT_switch.setText('CT Boundary Disable')
        else:
            self.actionCT_switch.setText('CT Boundary Enable')

    #KeyPress
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.keyPointList.setCurrentRow(-1)
            self.params["selectKey"] = None
        elif event.key() == Qt.Key_Up:
            if self.keyPointList.currentRow() == -1:
                self.keyPointList.setCurrentRow(self.keyPointList.count()-1)
            else:
                self.keyPointList.setCurrentRow(self.keyPointList.currentRow()-1)
        elif event.key() == Qt.Key_Down:
            if self.keyPointList.currentRow() == -1:
                self.keyPointList.setCurrentRow(0)
            else:
                self.keyPointList.setCurrentRow(self.keyPointList.currentRow()+1)
        elif event.key() == Qt.Key_Left:
            self.params["page"] = max(self.params["page"]-1,0)
            self.ScrollBar.setPos(self.params["page"]+1)
        elif event.key() == Qt.Key_Right:
            self.params["page"] = min(self.params["page"]+1,self.maxpage-1)
            self.ScrollBar.setPos(self.params["page"]+1)
        elif event.key() == Qt.Key_Tab:
            if self.keyPointList.currentRow() == -1:
                self.keyPointList.setCurrentRow(0)
            else:
                self.keyPointList.setCurrentRow(self.keyPointList.currentRow()+1)
        elif event.key() == Qt.Key_Control:
            self.ctrl = True
        elif event.key() == Qt.Key_A:
            self.A = True
        elif event.key() == Qt.Key_P:
            self.optFlow()
        elif str(event.key()) == '16777220': #Qt.Key_Enter
            self.ScrollBar.addHighLight()
    
    #KeyRelease
    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.ctrl = False
        elif event.key() == Qt.Key_A:
            self.A = False

    #Add KeyPoint
    def keyPointAdd(self,coor):
        if not self.videoCap:
            reply = QMessageBox.critical(self, "Error", "Empty Image Collection!!", QMessageBox.Ok, QMessageBox.Ok)
            return
        if len(self.params["keyPoints"][self.params["page"]])>=len(mapKP2ID):
            reply = QMessageBox.critical(self, "Error", "Key Points Full!!", QMessageBox.Ok, QMessageBox.Ok)
            return

        if not coor:
            coor = (0.5,0.5)
        dlg = addWindow(self,coor)
        dlg.show()
        dlg.exec_()

    #Delete selected KeyPoint
    def keyPointDelete(self):
        if self.keyPointList.currentRow() == -1:
            reply = QMessageBox.critical(self, "Error", "No key point has been selected!! Select a point first.", QMessageBox.Ok, QMessageBox.Ok)
        elif not self.params["selectKey"] == 'Empty':
            cur = self.keyPointList.currentRow()
            self.params["keyPoints"][self.params["page"]].pop(self.params["selectKey"])
            self.updateWinTitle(False)
            self.printMsg(f'Key Point "{self.params["selectKey"]}" is deleted from the list.')
            self.loadKeyPointList()
            self.keyPointList.setCurrentRow(min(cur,self.keyPointList.count()-1))
            self.progressBar.setValue(self.progressCount())
                
    #Reset Selected KeyPoint
    def keyPointReset(self):
        if self.keyPointList.currentIndex().row() == -1:
            reply = QMessageBox.critical(self, "Error", "No key point has been selected!! Select a point first.", QMessageBox.Ok, QMessageBox.Ok)
        else:
            self.params["keyPoints"][self.params["page"]][self.params["selectKey"]] = (0.5,0.5)
            self.updateWinTitle(False)
            self.loadKeyPointList()
            self.printMsg(f'Key Point "{self.params["selectKey"]}" is reset.')

    #Load Undistortion Parameter file
    def loadDistortionParameter(self):
        if not self.UndistEnable.checkState():
            reply = QMessageBox.critical(self, "Error", "Undistortion unabled!", QMessageBox.Ok, QMessageBox.Ok)
            return
        dlg = QFileDialog()
        fname,_ = dlg.getOpenFileName(self, 'Open Undistortion Parameter file', './', "Configuration files (*.yml)")
        if not fname:
            return
        self.params["undistPath"] = fname
        self._undist = Undistortion(fname)
        self.UndistAddressWin.setPlainText(self.params["undistPath"])
        self.printMsg(f'Undistortion Parameter file loaded successfully from {fname}')

    #Load the pre-calculated Intensity Datas
    def loadIntCSV(self):
        dlg = QFileDialog()
        fname,_ = dlg.getOpenFileName(self, 'Open Intensity Data File', './', "Intensity datas (*.csv)")
        if not fname:
            return
        absDiff = np.loadtxt(open(fname,'rb'),delimiter=',')
        if len(absDiff) != self.maxpage:
            reply = QMessageBox.critical(self, "Error", "Incorrect CSV File, frames length not match!", QMessageBox.Ok, QMessageBox.Ok)
            return 
        self.ScrollBar.newPlot(absDiff)
        self.printMsg(f'Intensity Data File loaded successfully from {fname}')
        rms = utils.RMSCurve(absDiff, 0.9)
        errorRegions = utils.detectErrorRegion(rms, sThresh=150, eThresh=10, winSize=20)
        self.printMsg(f'Add error frames at {errorRegions}')
        # print(errorRegions)
        for region in errorRegions:
            self.ScrollBar.addHighLight(*region)

    #Check if Undistortion enabled without *.yml file
    def checkUndEnable(self):
        if self.UndistEnable.checkState() and not self._undist:
            reply = QMessageBox.critical(self, "Error", "Undistortion Parameter file has not been selected!!", QMessageBox.Ok, QMessageBox.Ok)
            return False
        return True

    #Openpose Server connection timeout ####UNDER DEVELOP####
    def connectTimeout(self):
        reply = QMessageBox.warning(self, "Warning", "The connection to OpenposeServer has timeout, check on server.", QMessageBox.Ok, QMessageBox.Ok)
        self.connectionTimer.stop()

    #Mouse click behavior
    def mousePressEvent(self, event):
        x_offset = (self.ImageWin.width()-self.pixmap.width())/2
        y_offset = (self.ImageWin.height()-self.pixmap.height())/2
        x = event.x()-self.ImageWin.pos().x()-x_offset
        y = event.y()-self.ImageWin.pos().y()-21-y_offset
        if 0<=x<=self.pixmap.width() and 0<=y<=self.pixmap.height():
            y,x,height,width = utils.coorCvtInv((y,x),
                                                (self.pixmap.height(),self.pixmap.width()),
                                                self.params["rotDeg"],
                                                self.params["opDeg"])
            KPs = self.params["keyPoints"][self.params["page"]]
            if self.ctrl and (self.keyPointList.currentRow()!=-1):
                KPs[self.params["selectKey"]] = (y/height,x/width)
                self.updateWinTitle(False)
                self.loadKeyPointList()
            elif self.A:
                self.keyPointAdd((y/height,x/width))
            else:
                newSelectKey = utils.selectKey(y,x,KPs,height,width)
                if newSelectKey:
                    self.params["selectKey"] = newSelectKey
                    self.loadKeyPointList()

    #AutoSave toggle
    def setAutoSave(self):
        if self.cb_autoSave.isChecked():
            self.autoSaveTimer.start(300000)#5分钟300000
            self.printMsg('Autosave ON')
        else:
            self.autoSaveTimer.stop()
            self.printMsg('Autosave OFF')

    #Image 90 rotate
    def rotateClock(self):
        self.params["rotDeg"] = (self.params["rotDeg"]+90)%360
        self.update()
    
    #Image -90 rotate
    def rotateAntiClock(self):
        self.params["rotDeg"] = (self.params["rotDeg"]-90)%360
        self.update()

###### ALGORITHM ######
    #open grpc tunnel with server for Openpose Keypoint predictions
    def grpcOP(self):
        try:
            # self.connectionTimer.start(5000)
            cursor = self.params["page"]
            with grpc.insecure_channel(self.IP['openpose']) as channel:
                img = self.videoCap[cursor]
                self.params["opDeg"] = self.params["rotDeg"]
                img = cv.cvtColor(np.rot90(img,4-self.params["rotDeg"]//90),cv.COLOR_RGB2BGR)
                stub = openpose_pb2_grpc.OpenposeDetectStub(channel)
                h,w,c = img.shape
                response = stub.Upload(openpose_pb2.Send(img=img.tobytes(),
                                                         filename='%s' % cursor,
                                                         h=h,w=w,c=c,r=self.params["rotDeg"]))
            # self.connectionTimer.stop()
            kpDict = eval(response.result.decode(encoding='utf-8'))
            return kpDict
        except grpc._channel._InactiveRpcError:
            self.printMsg('Cannot connect to the Server. Verify the server\'s IP/Port!')
            return None

    #Get Openpose Keypoints from Server
    def Openpose(self):
        if not self.videoCap:
            reply = QMessageBox.critical(self, "Error", "Empty Image Collection!!", QMessageBox.Ok, QMessageBox.Ok)
            return
        cursor = self.params["page"]
        kpDict = self.grpcOP()        
        if kpDict:
            for key in kpDict:
                self.params["keyPoints"][cursor][key] = kpDict[key]
            self.updateWinTitle(False)
            self.loadKeyPointList()
            self.printMsg('Key points received from Openpose Server!!')
            self.progressBar.setValue(self.progressCount())
        elif kpDict is not None:
            self.printMsg('Received EMPTY key point from Openpose Server.')

    #open grpc tunnel with server for RAFT File
    def grpcOptFlow(self, videoName):
        assert isinstance(self.params["path"],str) # only support video for RAFT calculation
        MAX_MESSAGE_LENGTH = 1024*1024*1024
        try:
            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.Directory)
            path = dlg.getExistingDirectory(self, "Choose RAFT save path", "")
            if not path:
                return False,None
            with grpc.insecure_channel(self.IP['RAFT'],options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                ]) as channel:
                stub = raft_pb2_grpc.raftDetectStub(channel)
                response = stub.Upload(raft_pb2.Send(Type='video',videoName=videoName))
                if response.status:
                    return response,path
                else:
                    self.printMsg(f'Returned ERROR RAFT matrices from server!!')
                    return False,None
        except grpc._channel._InactiveRpcError as e:
            print(e)
            self.printMsg('Cannot connect to the Server. Verify the server\'s IP/Port!')
            return False,None

    #Get RAFT file from server
    def getOptFlow(self):
        if not self.videoCap:
            reply = QMessageBox.critical(self, "Error", "Empty Image Collection!!", QMessageBox.Ok, QMessageBox.Ok)
            return
        videoName = (self.params["path"].split('/')[-1]).split('.')[0]
        response,path = self.grpcOptFlow(videoName)
        if response:
            forw = response.forw
            backw = response.backw
            self.RAFT_FW = writeReadRAFT(path+'/'+videoName+'_forw.avi', forw)
            self.RAFT_BW = writeReadRAFT(path+'/'+videoName+'_backw.avi', backw)
            if utils.checkRAFT(self.videoCap, self.RAFT_FW, self.RAFT_BW):
                self.printMsg(f'Successfully fetch RAFT matrices from server!!')
                return True
            else:
                self.RAFT_FW = None
                self.RAFT_BW = None
                os.remove(path+'/'+videoName+'_forw.avi')
                os.remove(path+'/'+videoName+'_backw.avi')
                self.printMsg(f'Returned ERROR RAFT matrices from server!!')
                return False
        
    #get RAFT matrix from server and move keypoints accordingly
    def optFlow(self):
        if not self.videoCap:
            reply = QMessageBox.critical(self, "Error", "Empty Image Collection!!", QMessageBox.Ok, QMessageBox.Ok)
            return
        if not self.RAFT_FW or not self.RAFT_BW:
            reply = QMessageBox.critical(self, "Error", "Empty RAFT!! Load RAFT either from localfile or request from RAFTServer first!", QMessageBox.Ok, QMessageBox.Ok)
            return
        
        dlg = optFlowWindow(self)
        dlg.show()
        dlg.exec_()

###### DISPLAY ######
    #List KeyPoints
    def loadKeyPointList(self):
        temp = self.params["selectKey"]
        self.keyPointList.clear()
        self.params["selectKey"] = temp
        if not self.params["keyPoints"]:
            return

        kps = self.params["keyPoints"][self.params["page"]]
        if kps:
            #show keypoints in ascending order of y and x
            kps = sorted(kps.items(),key=lambda item:(item[1][0],item[1][1]))
            self.keyPointList.addItems([f'{kp[0]}: ({kp[1][1]:.2f}, {kp[1][0]:.2f})' for kp in kps])
            self.keyPointList.setEnabled(True)
        else:
            self.keyPointList.addItem('Empty')
            self.keyPointList.setEnabled(False)
        
        #highlight selected key in KeyPoints
        for i in range(self.keyPointList.count()):
            if self.keyPointList.item(i).text().split(':')[0]==self.params["selectKey"]:
                self.keyPointList.setCurrentRow(i)
                break

    #Key Point selected
    def keyPointSelected(self):
        if self.keyPointList.currentItem():
            self.params["selectKey"] = self.keyPointList.currentItem().text().split(':')[0]
        else:
            self.params["selectKey"] = None

    #Msg box
    def printMsg(self, msg, type_ = 'A', widget=None):
        cur = time.localtime()
        msg = f'[{cur.tm_year}/{cur.tm_mon:02d}/{cur.tm_mday:02d} {cur.tm_hour:02d}:{cur.tm_min:02d}:{cur.tm_sec:02d}] '+msg
        if widget==None:
            widget = self.MsgWin
        if type_.lower() == 'c':
            widget.setPlainText(msg)
        elif type_.lower() == 'a':
            widget.appendPlainText(msg)
    
    #calc progress
    def progressCount(self):
        count = 0
        for kp in self.params["keyPoints"]:
            if kp:
                count+=1
        return count

    #reload frame
    def paintEvent(self,event):
        self.ctrl = self.ctrl and self.centralwidget.hasFocus()
        self.A = self.A and self.centralwidget.hasFocus()
        if self.maxpage:
            qImg = utils.cvtImg2QImage(self.videoCap[self.params["page"]],QImage)
            matrix = QTransform()
            matrix.rotate(self.params["rotDeg"])
            img = qImg.transformed(matrix).scaled(self.ImageWin.width(),
                                                  self.ImageWin.height(),
                                                  Qt.KeepAspectRatio,
                                                  Qt.FastTransformation)
            self.pixmap = QPixmap.fromImage(img)
            self.showSkeleton(self.pixmap)
            self.showKeyPoints(self.pixmap)
            self.ImageWin.setPixmap(self.pixmap)

    #show keypoints on the frame
    def showKeyPoints(self,pixmap):
        painter = QPainter()
        painter.begin(pixmap)
        painter.setPen(QPen(Qt.red,5))
        kpDict = self.params["keyPoints"][self.params["page"]]
        for key in kpDict:
            y,x = utils.coorCvt(kpDict[key],self.params["rotDeg"],self.params["opDeg"])
            if y<0 or y>1 or x<0 or x>1:
                continue
            x = x*pixmap.width()
            y = y*pixmap.height()
            if key == self.params["selectKey"]:
                painter.setPen(QPen(Qt.green,5))
            painter.drawPoint(x,y)
            painter.drawText(x+2,y,key)
            painter.setPen(QPen(Qt.red,5))
        painter.end()
    
    #show sheleton on the frame
    def showSkeleton(self,pixmap):
        painter = QPainter()
        painter.begin(pixmap)
        kpDict = self.params["keyPoints"][self.params["page"]]
        for i in range(len(skeleton)):
            joint = [mapID2KP[kp] for kp in skeleton[i]]
            if joint[0] in kpDict and joint[1] in kpDict:
                painter.setPen(QPen(Qt.GlobalColor(i+1),3))
                y1,x1 = utils.coorCvt(kpDict[joint[0]],self.params["rotDeg"],self.params["opDeg"])
                x1 = x1*pixmap.width()
                y1 = y1*pixmap.height()
                y2,x2 = utils.coorCvt(kpDict[joint[1]],self.params["rotDeg"],self.params["opDeg"])
                x2 = x2*pixmap.width()
                y2 = y2*pixmap.height()
                painter.drawLine(x1,y1,x2,y2)
        painter.end()
    
    #switch frame when slide
    def slide(self):
        self.params["page"] = self.ScrollBar.Pos()-1
        self.sb_Page.setValue(self.params["page"]+1)
        self.loadKeyPointList()
        self.update()
        self.centralwidget.setFocus()
    
    #switch frame when edit pagebox
    def spinBox(self):
        self.params["page"] = self.sb_Page.value()-1 #会触发 self.slide, 不需要做update. (spinBox基本上和slider绑定在一起)
        self.ScrollBar.setPos(self.params["page"]+1)
        self.sb_Page.setFocus()

if __name__=="__main__":
    for proxy in ['http_proxy','https_proxy']:
        if proxy in os.environ:
            del os.environ[proxy]
    
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())

