#coding=utf-8
#Author miracle.why@qq.com
import math
import re

import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import itertools

# 判断这个数字n的平方和的字符串可以拆分为连续的子字符串并且和等于n
def is_punishment_number(n):
    # 计算s的连续子串相加和

    s_str = str(n*n)
    s_substrings = []

    for i in range(len(s_str)):
        for j in range(i, len(s_str)):
            s_substring = s_str[i:j + 1]
            if int(s_substring) <= n:  # 剪枝，小于n的才可以是答案
                s_substrings.append(s_substring)
    print(s_substrings)
    # 检查s的连续子串组合是否合法
    s_substrings.sort()
    for s_substring in s_substrings:
        if int(s_substring) != n:
            return False




print(is_punishment_number(36))