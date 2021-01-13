import time
import numpy as np
import os
import shutil
import sys
import glob
sys.path.append('raft/core')
import argparse
import cv2
import torch
from torch.utils.data import IterableDataset, DataLoader

from undistortion import *
from raft import RAFT
from utils.utils import InputPadder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger(object):
    def __init__(self, logFile ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile,'a')
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

class videoIter(IterableDataset):
    def __init__(self, videoCap, pad, distortion=None):
        self.videoCap = videoCap
        self.len = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT)-1)
        print('\tvideo info:', self.len+1, int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._undist = distortion
        self.pad = pad
        success, self.next = self.videoCap.read()
        self.next = convert_image(self._undist(self.next)) if success else None
        self.counter = 0
        self.shape = self.next.shape if success else None
    
    def __iter__(self):
        return self

    def __next__(self):
        self.prev = self.next
        success, self.next = self.videoCap.read()
        if success:
            self.next = convert_image(self._undist(self.next))
        elif self.pad>0:
            self.pad -= 1
            self.next = torch.from_numpy(np.zeros(self.shape)).float().to(DEVICE)
        else:
            raise StopIteration
        self.counter += 1
        return [self.prev, self.next]

def distortionFunc(distortion):
    def dummy(x):
        return x
    if distortion:
        return Undistortion(args.distortAddr).exec
    else:
        return dummy

def convert_image(img,rot=1):
    if rot:
        img = np.rot90(img,rot).astype(np.uint8)
    img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
    return img.to(DEVICE)

def calcPad(L, batch, gpus):
    n_gpu = len(gpus)
    rem = (L-1)%batch%n_gpu
    if rem:
        pad = n_gpu-rem
    else:
        pad = 0
    return int(pad)

def unpad2CPU(padder, numpy, array):
    if numpy:
        return padder.unpad(array).permute(0,2,3,1).cpu().numpy()
    else:
        return padder.unpad(array).permute(0,2,3,1).cpu()

def modifyOF(array, _max, _min):
    for i in range(array.shape[0]):
        if array[i].max()>=_max or array[i].min()<_min:
            array[i] = np.zeros(array[i].shape)
    return array

def checkOF(dtype, multiFactor, array):
    if dtype=="int8":
        maxVal = 127.5/multiFactor
        minVal = -128.5/multiFactor
    return modifyOF(array,maxVal,minVal)
    
def resizeMulti(array, sizeFactor, multiFactor, dtype):
    assert type(dtype)==type
    w, h, c = array.shape[-3:]
    array = np.round(np.array([cv2.resize(f.astype(np.float32)*multiFactor,(int(h/sizeFactor),int(w/sizeFactor))) for f in array])).astype(dtype)
    return array

def pipeline(model, image1, image2, padder, multiFactor=1, numpy=True, last = False, pad=0):
    _, flo = model(image1, image2, iters=20, test_mode=True)
    if last:
        flo = flo[:-pad]
    flo = unpad2CPU(padder, numpy, flo)
    flo = checkOF("int8", multiFactor, flo)
    return flo

def checkDir(dirFile):
    dirFile = dirFile.split('/')
    for i in range(1,len(dirFile)+1):
        path = '/'.join(dirFile[:i])
        if not os.path.exists(path):
            os.mkdir(path)

def run():
    sys.stdout = Logger(args.log)
    checkDir('../complete')
    checkDir(args.out+'forward/')
    checkDir(args.out+'backward/')

    model = torch.nn.DataParallel(RAFT(args), device_ids=args.gpus)
    model.load_state_dict(torch.load(args.model))
    model.to(DEVICE)
    
    with torch.no_grad():
        videoList = glob.glob(os.path.join(args.videos, '*.mp4')) # get all video path
        fourcc = cv2.VideoWriter_fourcc(*'FFV1') # video encoding code
        total_job = len(videoList) # counter
        fin_job=0
        
        _undist = distortionFunc(args.distortionOFF) ## undistortion.exec() function
        multiFactor=2

        for videoPath in videoList:
            try:
                fileName = videoPath.split('/')[-1]
                print(f'processing {fileName}')
                fileName = fileName.split('.')[0]+'.avi'
                start = time.time() # timer
                videoCap = cv2.VideoCapture(videoPath) # video reader
                pad = calcPad(videoCap.get(cv2.CAP_PROP_FRAME_COUNT),args.batch,args.gpus) # calculate padding frame number

                videoIterator = videoIter(videoCap, pad, _undist)
                padder = InputPadder(videoIterator.shape) # frame padder (X8)
                video_loader = DataLoader(dataset=videoIterator,batch_size=args.batch)
                rem = videoCap.get(cv2.CAP_PROP_FRAME_COUNT)+pad # remaining counter
                
                forw = cv2.VideoWriter(args.out+'forward/'+fileName,cv2.CAP_FFMPEG, fourcc,20.0,(*videoIterator.shape[:-3:-1],)) # forward video writer
                back = cv2.VideoWriter(args.out+'backward/'+fileName,cv2.CAP_FFMPEG, fourcc,20.0,(*videoIterator.shape[:-3:-1],)) # backward video writer

                for i_batch, data_blob in enumerate(video_loader):
                    image1, image2 = data_blob
                    image1, image2 = padder.pad(image1, image2)
                    # pipeline of calculating RAFT result
                    forwUp = pipeline(model, image1, image2, padder, multiFactor, last=rem<=args.batch, pad=pad)
                    backUp = pipeline(model, image2, image1, padder, multiFactor, last=rem<=args.batch, pad=pad)
                    # write each frame in batch
                    for frame in forwUp:
                        frame = np.round(np.concatenate([frame, np.zeros((*frame.shape[:2],1))],2)*2+128).astype(np.uint8)
                        forw.write(frame)
                    for frame in backUp:
                        frame = np.round(np.concatenate([frame, np.zeros((*frame.shape[:2],1))],2)*2+128).astype(np.uint8)
                        back.write(frame)
                    # update remaining counter
                    rem-=args.batch

                forw.release()
                back.release()

                shutil.move(videoPath, '../complete/'+fileName)
                fin_job+=1
                print(f'\tcost time: {time.time()-start:.2f}s, progress: {fin_job/total_job*100:.2f}%')
            except RuntimeError as e:
                print(e)
                print(f'\t{fileName} failed, move to next video')
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='raft/raft-small.pth')
    parser.add_argument('--distortionOFF', action='store_false')
    parser.add_argument('--distortAddr', default='./IntrinsicParameter.yml')
    parser.add_argument('--videos', default='/home/intern/videos/srrs/front/')
    parser.add_argument('--out', default='../RAFTavi/')
    parser.add_argument('--log', default='./RAFT.log')
    parser.add_argument('--small', action='store_false')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1,2,3])
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()

    run()
    