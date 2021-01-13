import cv2 as cv
import os
import numpy as np

class Video(object):
    def __init__(self, videosPath, undistEnable=None, _undist=None):
        self.cap = cv.VideoCapture(videosPath)
        self.len = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.undistEnable = undistEnable
        self._undist = _undist
        self.shape = self.shape()
        self.read = self.cap.read
        self.get = self.cap.get
    
    def __getitem__(self, indices):
        assert not isinstance(indices, tuple)
        indices = self.len+indices if indices<0 else indices
        if indices>=self.len:
            raise IndexError("list index out of range")

        self.reset(indices)
        _, frame = self.cap.read()
        if _:
            frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            if (self.undistEnable and self._undist):
                frame = self._undist.exec(frame)
            return frame
        else:
            raise Exception
    
    def __len__(self):
        return self.len

    def reset(self, index=0):
        self.cap.set(cv.CAP_PROP_POS_FRAMES,index)
    
    def shape(self):
        return (self.len,*self[0].shape)

class Images(object):
    def __init__(self, imagePathSet, undistEnable=None, _undist=None):
        self.imgPathSet = imagePathSet
        self.len = len(imagePathSet)
        self.undistEnable = undistEnable
        self._undist = _undist
        self.shape = self.shape()
    
    def __getitem__(self, indices):
        assert not isinstance(indices, tuple)
        indices = self.len+indices if indices<0 else indices

        if indices>=self.len:
            raise IndexError("list index out of range")
        
        frame = cv.cvtColor(cv.imread(self.imgPathSet[indices]),cv.COLOR_BGR2RGB)
        if (self.undistEnable and self._undist):
            frame = self._undist.exec(frame)
        return frame
    
    def __len__(self):
        return self.len
    
    def shape(self):
        return (self.len,*self[0].shape)


def pipeRAFT(RAFTframe, offset=128, magFactor=2):
    """convert type of RAFT
    adjust offset and precision.

    :param RAFTframe: RAFT frame matrix
    :rtype: numpy.ndarray
    """
    return (RAFTframe.astype(np.float16)-offset)/magFactor

class RAFT(object):
    def __init__(self, raftPath):
        self.cap = cv.VideoCapture(raftPath)
        self.len = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.shape = self.shape()
    
    def __getitem__(self, indices):
        assert not isinstance(indices, tuple)
        indices = self.len+indices if indices<0 else indices
        
        if indices>=self.len:
            raise IndexError("list index out of range")
        self.reset(indices)
        _, frame = self.cap.read()
        if _:
            return pipeRAFT(frame[...,:2])
        else:
            raise Exception
    
    def __len__(self):
        return self.len

    def reset(self, index=0):
        self.cap.set(cv.CAP_PROP_POS_FRAMES,index)
    
    def shape(self):
        return (self.len,*self[0].shape)

def writeReadRAFT(path, byteFile):
    f = open(path,'wb')
    f.write(byteFile)
    f.close()
    return RAFT(path)

def video2frame(videosPath,timeInterval=1):
    retVal = []
    vidcap = cv.VideoCapture(videosPath)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % timeInterval == 0:
            retVal.append(image)
            success, image = vidcap.read()
            count += 1
    return np.array(retVal)

def saveFrame(iterator, savePath, rot=0):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    count = 0
    for img in iterator:
        img = cv.cvtColor(np.rot90(img,rot),cv.COLOR_RGB2BGR) 
        cv.imencode('.jpg', img)[1].tofile(savePath + "\\frame%d.jpg" % count)
        count+=1

if __name__ == '__main__':
    videosPath = input()
    savePath = videosPath.split('.')[0]
    timeInterval = 1
    video2frame(videosPath, savePath, timeInterval)