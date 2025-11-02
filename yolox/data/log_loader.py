"""
Create: 2021.09.17
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import numpy as np
import struct
import cv2

class LogLoader(object):
    def __init__(self, file_path, yuv = False):
        self.width = 1920
        self.height = 1080
        self.yuv = yuv
        
        if self.yuv:
            self.channel = 2
        else:
            self.channel = 3

        self.img_size = self.width * self.height * self.channel
        self.f = open(file_path, 'rb')
        self.num_of_tl = 0
        self.tick = 0
        self.save_flag = 0
        self.tl_img_coordi_x = np.zeros((5), dtype = np.float)
        self.tl_img_coordi_y = np.zeros((5), dtype = np.float)
        self.cx = 0
        self.cy = 0
        self.fx = 0
        self.fy = 0
        self.radial_distort = np.zeros((3), dtype = np.float)
        self.tangential_distort = np.zeros((2), dtype = np.float)
        self.dx = 0
        self.dy = 0
        self.step = 4
        self.fps = 30
        self.frame_num = 0
        self.crop_offset_x = 0
        self.crop_offset_y = 0
        self.roi_width = 0
        self.roi_height = 0

    def __getitem__(self, idx):
        try:
            data = self.f.read(self.img_size)
        except:
            self.f.close()

            raise ValueError('Cannot Read Bin Data')

        im = np.frombuffer(data, dtype = np.uint8)

        if im.shape[0] != self.img_size:
            self.f.close()

            return None, None

        im = im.reshape(self.height, self.width, self.channel)

        if self.yuv:
            im = cv2.cvtColor(im, cv2.COLOR_YUV2BGR_UYVY)

        self.frame_num += 1

        return im, None
