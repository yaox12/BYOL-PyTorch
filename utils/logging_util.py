# -*- coding: utf-8 -*-
import logging
import sys

def get_std_logging():
    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s',
        level=logging.INFO
    )
    return logging
