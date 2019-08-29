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


def parse(text):
    content = {"token":"79914a773ce44d72bd3d8a6e75dfca36","text":text}
    urlcontent = urllib.parse.urlencode(content)
    url = 'http://jdialog-lexeme-stage.jd.com/lexeme?{}'.format(urlcontent)

    address = quote(url,safe=st.printable)
    response = request.urlopen(address).read().decode('utf-8')
    response = json.loads(json.dumps(response))
    response = json.loads(response)
    
#    #json结果示例，json串包含分词、词性、命名实体识别等，参赛者可根据各自需要自行选择
#    print(response)
    
    #分词
    length = len(response['tokenizedText'])
    seg = []
    for i in range(length):
        seg.append(response['tokenizedText'][i]['word'])
    return seg


if __name__ == '__main__':
    text = '请您稍等，我们马上为你安排发货#EOS[数字]'
    seg = parse(text)
    print(seg)
