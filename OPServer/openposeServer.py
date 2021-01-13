from concurrent import futures
import time
import logging
import numpy as np
import argparse

import os
import sys

from openpose.model_wrapper import ModelWrapper

import grpc
import openpose_pb2
import openpose_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
model_path = "./openpose/trained_models/"

class OpenposeDetect(openpose_pb2_grpc.OpenposeDetectServicer):
    def __init__(self):
        self.model_wrapper = ModelWrapper(model_path)
        self.model_wrapper.process_image(np.zeros((3,3,3)))

    def Upload(self, request, context):
        print('received openpose request..')
        file_data = request.img
        file_name = request.filename
        h = request.h
        w = request.w
        c = request.c
        r = request.r
        file_data = np.frombuffer(file_data,dtype='uint8').reshape(h,w,c)
        #处理openpose
        skeleton = self.model_wrapper.process_image(file_data)
        if skeleton:
            skeleton = skeleton[0]
            print(skeleton.keypoints.keys())
            return openpose_pb2.OpenposeResult(result=bytes(str(skeleton.keypoints),encoding='utf-8'))
        else:
            print('Empty keypoints')
            return openpose_pb2.OpenposeResult(result=bytes(str(dict()),encoding='utf-8'))

def serve():
    port = '[::]:'+args.port
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    openpose_pb2_grpc.add_OpenposeDetectServicer_to_server(OpenposeDetect(), server)
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
    parser.add_argument('--port', default='50051')
    args = parser.parse_args()

    logging.basicConfig()
    serve()