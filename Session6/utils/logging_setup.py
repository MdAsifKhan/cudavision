#!/usr/bin/env python

"""Logger parameters for the entire process.
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2018'


import logging
import datetime
import sys
import re
from os.path import join

from utils.arg_pars import opt


logger = logging.getLogger('basic')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

filename = sys.argv[0]
search = re.search(r'\/*(\w*).py', filename)
filename = search.group(1)


def path_logger():
    global logger
    path_logging = join(opt.dataset_root, 'logs', '%s%s(%s)' %
                        (opt.log_str, filename,
                         str(datetime.datetime.now())))
    fh = logging.FileHandler(path_logging, mode='w')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelno)s - %(filename)s - '
                                  '%(funcName)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))

    return logger