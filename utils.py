import numpy as np
import pandas as pd

import os
import random
import pickle
import re
import time
import datetime
import copy

import multiprocessing
from multiprocessing import Pool, Queue


def cosine_similarity(df):
    mat = df.values
    norms = np.sum(mat**2, axis=0)**0.5
    norms = norms.reshape(1,-1)
    cossim = (mat.T @ mat) / (norms.T @ norms) 
    return pd.DataFrame(index=df.columns, columns=df.columns, data=cossim)


def getLeastUsedGpuId():
    import pynvml
    pynvml.nvmlInit()
    
    gpu_count = pynvml.nvmlDeviceGetCount()
    def getGpuUsage(index):
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index))
        usage = meminfo.used / meminfo.total
        return usage
    usage = [getGpuUsage(i) for i in range(gpu_count)]
    least_used_gpu_id = usage.index(min(usage))
    
    pynvml.nvmlShutdown()
    return least_used_gpu_id
    

def getMyTime():
    return time.strftime('%m/%d %H:%M:%S', time.localtime(time.time()))

def logPrint(*args, **kwargs):
    print(getMyTime(), '|', *args, **kwargs)




STYLE = {
        'fore':
        {   # 前景色
            'black'    : 30,   #  黑色
            'red'      : 31,   #  红色
            'green'    : 32,   #  绿色
            'yellow'   : 33,   #  黄色
            'blue'     : 34,   #  蓝色
            'purple'   : 35,   #  紫红色
            'cyan'     : 36,   #  青蓝色
            'white'    : 37,   #  白色
        },

        'back' :
        {   # 背景
            'black'     : 40,  #  黑色
            'red'       : 41,  #  红色
            'green'     : 42,  #  绿色
            'yellow'    : 43,  #  黄色
            'blue'      : 44,  #  蓝色
            'purple'    : 45,  #  紫红色
            'cyan'      : 46,  #  青蓝色
            'white'     : 47,  #  白色
        },

        'mode' :
        {   # 显示模式
            'mormal'    : 0,   #  终端默认设置
            'bold'      : 1,   #  高亮显示
            'underline' : 4,   #  使用下划线
            'blink'     : 5,   #  闪烁
            'invert'    : 7,   #  反白显示
            'hide'      : 8,   #  不可见
        },

        'default' :
        {
            'end' : 0,
        },
}

def UseStyle(string, mode = '', fore = '', back = ''):
    mode  = '%s' % STYLE['mode'][mode] if mode in STYLE['mode'] else ''
    fore  = '%s' % STYLE['fore'][fore] if fore in STYLE['fore'] else ''
    back  = '%s' % STYLE['back'][back] if back in STYLE['back'] else ''
    style = ';'.join([s for s in [mode, fore, back] if s])
    style = '\033[%sm' % style if style else ''
    end   = '\033[%sm' % STYLE['default']['end'] if style else ''
    return '%s%s%s' % (style, string, end)