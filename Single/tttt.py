#coding=utf-8
#Author miracle.why@qq.com


import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO


def wordindex(word:str):
    """
    获取单词后缀和名字
    :param word:
    :return:
    """
    import re
    pattnum = re.compile("\(([\d]+)\)$")
    res = re.findall(pattnum, word)

    if res:
        return (re.sub(pattnum,"",word), res[0])
    else:
        return False

print(wordindex("gta(2)(1)"))

