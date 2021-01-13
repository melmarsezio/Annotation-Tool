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
        self.shape = self.next.shape
    
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

def run():
    sys.stdout = Logger(args.log) 
    if not os.path.exists('../complete'):
        os.mkdir('../complete')
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    model = torch.nn.DataParallel(RAFT(args), device_ids=args.gpus)
    model.load_state_dict(torch.load(args.model))
    model.to(DEVICE)
    
    with torch.no_grad():
        videoList = glob.glob(os.path.join(args.videos, '*.mp4'))
        total_job = len(videoList)
        fin_job=0
        ## undistortion.exec() function
        _undist = distortionFunc(args.distortionOFF)
        sizeFactor=4
        multiFactor=2
        
        for videoPath in videoList:
            try:
                fileName = videoPath.split('/')[-1]
                print(f'processing {fileName}')
                start = time.time()
                videoCap = cv2.VideoCapture(videoPath)
                pad = calcPad(videoCap.get(cv2.CAP_PROP_FRAME_COUNT),args.batch,args.gpus)

                videoIterator = videoIter(videoCap, pad, _undist)
                padder = InputPadder(videoIterator.shape)
                video_loader = DataLoader(dataset=videoIterator,batch_size=args.batch)

                flo_forw,flo_back = [],[]
                for i_batch, data_blob in enumerate(video_loader):
                    image1, image2 = data_blob
                    image1, image2 = padder.pad(image1, image2)

                    forwLow, forwUp = model(image1, image2, iters=20, test_mode=True)
                    backLow, backUp = model(image2, image1, iters=20, test_mode=True)

                    forwUp = unpad2CPU(padder, True, forwUp)
                    backUp = unpad2CPU(padder, True, backUp)
                    forwUp = checkOF("int8", multiFactor, forwUp)
                    backUp = checkOF("int8", multiFactor, backUp)
                    flo_forw.append(forwUp)
                    flo_back.append(backUp)

                print(f'\tcalculation time: {time.time()-start:.2f}s')

                # flo_forw = np.concatenate(flo_forw,0).astype(np.float16)
                # flo_back = np.concatenate(flo_back,0).astype(np.float16)
                # if pad:
                #     flo_forw = flo_forw[:-pad]
                #     flo_back = flo_back[:-pad]

                # flo_forw = resizeMulti(flo_forw, sizeFactor=sizeFactor, multiFactor=multiFactor, dtype=np.int8)
                # flo_back = resizeMulti(flo_back, sizeFactor=sizeFactor, multiFactor=multiFactor, dtype=np.int8)
                flo_forw = np.round(np.concatenate(flo_forw,0).astype(np.float16)+128).astype(np.uint8)
                flo_back = np.round(np.concatenate(flo_back,0).astype(np.float16)+128).astype(np.uint8)
                if pad:
                    flo_forw = flo_forw[:-pad]
                    flo_back = flo_back[:-pad]

                flo = np.stack([flo_forw, flo_back])
                print(f'\tfinal output dims:', *flo.shape)

                np.save('flo',flo)
                # np.save(args.out + fileName,flo)
                # shutil.move(videoPath, '../complete/'+fileName)
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
    parser.add_argument('--out', default='../RAFTnpy/')
    parser.add_argument('--log', default='./RAFT.log')
    parser.add_argument('--small', action='store_false')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1,2,3])
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()

    run()
    