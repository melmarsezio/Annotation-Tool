#!/usr/bin/env python
# -*- coding:utf-8 -*- 
'''
* @Author: Kai CHEN@z0041zvt
* @CreateDate: 2020-07-23 13:58:17
* @LastEditors: Kai CHEN@z0041zvt
* @LastEditTime: 2020-07-23 14:08:31
* @Email: kai-chen@siemens-healthineers.com
* @Description: 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2 as cv
import pathlib
import numpy as np

class Undistortion(object):
    def __init__(self, path, enableNewMtx=True):
        self.__enableNewMtx = enableNewMtx
        self.__mtx, self.__dist = None, None
        self.__parseParams(path)
        self._genUndistMap()
    
    def __parseParams(self, path):
        fs = cv.FileStorage(path, cv.FileStorage_READ)
        __p = pathlib.Path(path)
        if not __p.exists():
            raise IOError("Can't find intrinsic parameter file!")
        assert __p.suffixes[0] == ".yml"
        
        self.__mtx = fs.getNode("Camera Matrix").mat()
        self.__dist = fs.getNode("Distortion").mat()
        assert isinstance(self.__mtx, np.ndarray) and isinstance(self.__dist, np.ndarray)

        self._resolution = tuple(fs.getNode("Resolution").mat())
        fs.release()
    
    def _genUndistMap(self):

        if self.__enableNewMtx:
            _newcameramtx, self._roi = cv.getOptimalNewCameraMatrix(
                self.__mtx, self.__dist, self._resolution, 1.0, self._resolution)
        else:
            _newcameramtx = self.__mtx
            self._roi = (0, 0, *self.__resolution)
        
        self._mapx, self._mapy = cv.initUndistortRectifyMap(
            self.__mtx, self.__dist, None, _newcameramtx, self._resolution, 5)

    def exec(self, image, crop=True):
        
        dst = cv.remap(image, self._mapx, self._mapy, cv.INTER_AREA)
        if not crop:
            return dst
        else:
            x, y, w, h = self._roi
            dst = dst[y:y+h, x:x+w]
            return dst

if __name__ == "__main__":
    #example
    _path = pathlib.Path('./IntrinsicParameter.yml')
    _undist = Undistortion(str(_path))
    image = cv.imread('raw.mp4#t=0.jpg')

    uImage = _undist.exec(image, True)

    cv.imshow('undistortion', uImage)
    cv.imshow('origin', image)
    cv.waitKey()