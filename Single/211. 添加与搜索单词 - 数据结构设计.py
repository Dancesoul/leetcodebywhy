# -*- coding:utf-8 -*-
# @FileName  :211. 添加与搜索单词 - 数据结构设计.py
# @Time      :2021/10/19 0019 10:04
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
# Description:
import re


class WordDictionary:

    d = []
    def __init__(self):
        pass
    def addWord(self, word: str) -> None:
        self.d.append(word)

    def search(self, word: str) -> bool:
        if not self.d:
            return False
        i = 0
        j = len(self.d) -1
        patt = re.compile(word+"$")
        while i<=j:
            p1 = re.findall(patt,self.d[i])
            p2 = re.findall(patt,self.d[j])
            if p1 or p2:
                return True
            i+=1
            j-=1
        return False


w = WordDictionary()

print(w.search("a"))