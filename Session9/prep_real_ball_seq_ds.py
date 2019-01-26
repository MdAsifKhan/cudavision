#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'January 2019'

import os
import re
import numpy as np

from arguments import opt

save_path = opt.seq_real_balls
# save_path = os.path.join(os.path.abspath(dataset_root), 'npy')
if not os.path.exists(save_path):
    os.mkdir(save_path)


class Ball:
    def __init__(self, filename):
        self.filename = filename
        self.x = []
        self.y = []
        self.frames = []
        self.valid = True

    def next_frame(self, x, y, frame):
        if len(self.x) == 0 or frame - self.frames[-1] < 4:
            self.x.append(x)
            self.y.append(y)
            self.frames.append(frame)
            return True
        if len(self.x) < 20:
            self.valid = False
        return False

    def centers(self):
        coord = []
        for fidx, frame in enumerate(self.frames[:-1]):
            coord.append([frame, self.x[fidx], self.y[fidx]])
            for fidx_fake in range(self.frames[fidx + 1] - frame - 1):
                coord.append([fidx_fake + frame, -1, -1])
            # coord.append([[-1, -1]] * (self.frames[fidx + 1] - frame))
        coord.append([self.frames[-1], self.x[-1], self.y[-1]])
        coord = np.array(coord).reshape((-1, 3))
        # coord[:, 0] = coord[:, 0] * opt.map_size_x / 640
        # coord[:, 1] = coord[:, 1] * opt.map_size_y / 480
        coord = np.hstack(([self.filename] * coord.shape[0], coord))
        return coord


balls = []
for filename in os.listdir('SoccerDataSeq'):
    if not filename.endswith('txt'):
        continue
    with open(os.path.abspath(filename), 'r') as f:
        ball = Ball(filename)
        for line_idx, line in enumerate(f):
            if not line.startswith('label::ball'):
                continue
            line = line.split('|')
            frame = int(re.search(r'frame(\d*).jpg', line[1]).group(1))
            y,x = list(map(lambda x: int(x), line[-4:-2]))
            if not ball.next_frame(x, y, frame):
                if ball.valid:
                    balls.append(ball)
                ball = Ball()


for ball_idx, ball in enumerate(balls):
    np.savetxt(os.path.join(save_path, 'ball%d.txt' % ball_idx), ball.centers(), fmt='%d')




