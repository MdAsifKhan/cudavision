#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'December 2018'

import cv2
import os
import datetime

from utils.arg_pars import opt
from utils.logging_setup import logger


for video_name in os.listdir(opt.video_folder):
    logger.debug(video_name)
    save_video_folder = os.path.join(opt.save_folder, video_name)
    if not os.path.exists(save_video_folder):
        os.mkdir(save_video_folder)
    cap = cv2.VideoCapture(os.path.join(opt.video_folder, video_name))
    ret, frame = cap.read()
    assert ret

    logger.debug(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_n = 1
    try:
        while True:
            ret, frame = cap.read()
            assert ret
            cv2.imwrite(os.path.join(save_video_folder, '%d.png' % frame_n), frame)
            frame_n += 1
            if frame_n % 1000 == 0:
                logger.debug('%s %s' % (str(datetime.datetime.now()), frame_n))
    except AssertionError:
        pass



