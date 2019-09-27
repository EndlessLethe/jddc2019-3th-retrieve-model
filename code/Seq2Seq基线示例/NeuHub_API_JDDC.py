# -*- coding: utf-8 -*-
"""
大赛容器只提供内网访问
内网token:79914a773ce44d72bd3d8a6e75dfca36
具体使用细则和说明可见大赛‘API资源池’

该程序依赖python3
"""

import urllib
from urllib import request
from urllib.parse import quote
import string as st
import json
import jieba

def parse(text):
    return list(jieba.cut(text))


if __name__ == '__main__':
    text = '请您稍等，我们马上为你安排发货#EOS[数字]'
    seg = parse(text)
    print(seg)
