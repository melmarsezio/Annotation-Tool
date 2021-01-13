from concurrent import futures
import time
import logging
import numpy as np
import socket
import os
import sys
sys.path.append('raft/core')

import argparse
import cv2
import torch

from raft import RAFT
from utils.utils import InputPadder

import grpc
import raft_pb2
import raft_pb2_grpc

DEVICE = 'cuda'
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def convert_image(img,rot=0):
    if rot:
        img = np.rot90(img,rot)
    img = img.astype(np.uint8)
    img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

class RaftDetect(raft_pb2_grpc.raftDetectServicer):
    def __init__(self):
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))
        self.model = self.model.module
        self.model.to(DEVICE)

    def Upload(self, request, context):
        print(f'received, message type: {request.Type}..')
        start = time.time()
        if request.Type=='frames':
            msg = request.frameInfo
            prev = cv2.imdecode(np.frombuffer(msg.prev, np.uint8), cv2.IMREAD_COLOR)
            next_ = cv2.imdecode(np.frombuffer(msg.next_, np.uint8), cv2.IMREAD_COLOR)
            rot = msg.r
            with torch.no_grad():
                image1 = convert_image(prev,rot)
                image2 = convert_image(next_,rot)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                flow_up = padder.unpad(flow_up)
                flow_up = flow_up[0].permute(1,2,0).cpu().numpy().astype(np.float16)
            print(f'    time:{time.time()-start:.2f}s')
            return raft_pb2.raftResult(status=True,forw=flow_up.tobytes(),dummy=0)
        elif request.Type=='video':
            videoName = request.videoName.split('.')[0]+'.avi'
            forwPath = args.dir+'forward/'
            backPath = args.dir+'backward/'
            if os.path.isfile(forwPath+videoName) and os.path.isfile(backPath+videoName):
                status=True
                fwRAFT = open(forwPath+videoName,'rb').read()
                bwRAFT = open(backPath+videoName,'rb').read()
                print('    fetch',videoName)
            else:
                status=False
                fwRAFT=np.array([]).tobytes()
                bwRAFT=np.array([]).tobytes()
                print('   ',videoName,'not exist!!!')
            print(f'    time:{time.time()-start:.2f}s')
            return raft_pb2.raftResult(status=status,forw=fwRAFT,backw=bwRAFT)
        
def serve():
    port = '[::]:'+args.port
    
    MAX_MESSAGE_LENGTH = 1024*1024*1024
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
                ])
    raft_pb2_grpc.add_raftDetectServicer_to_server(RaftDetect(), server)
    server.add_insecure_port(port)
    print(f'=====The server is deployed at {port}=====')
    print('=====Server all setup!!=====')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        print('=====Server Shutting Down!!=====')
        server.stop(0)

if __name__ == '__main__':
    for proxy in ['http_proxy','https_proxy']:
        if proxy in os.environ:
            del os.environ[proxy]

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='raft/raft-small.pth')
    parser.add_argument('--small', action='store_false')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    parser.add_argument('--dir', default='/home/intern/RAFTavi/')
    parser.add_argument('--gpus', type=list, nargs='+', default=[0,1,2,3])
    parser.add_argument('--port', default='50000')
    args = parser.parse_args()

    logging.basicConfig()
    serve()