# -*- coding:utf-8 -*-
# @FileName  :哈希表，字典树 面试题16.02 单词频率.py
# @Time      :2023/10/18 0018 14:31
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
""" 
    Description:   
"""
import collections
from typing import *


class Trie:
    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False
        self.nums = 0    # 以该节点结尾的次数

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            ind = ord(ch) - ord('a')
            if not node.children[ind]:
                node.children[ind] = Trie()
            node = node.children[ind]
        node.isEnd =True
        node.nums += 1

    def search(self, word: str) -> int:
        node = self
        for ch in word:
            ind = ord(ch) - ord('a')
            if not node.children[ind]:
                return 0
            node = node.children[ind]
        if node.isEnd:
            return node.nums
        else:
            return 0


class WordsFrequency:

    def __init__(self, book: List[str]):
        self.trie = Trie()
        for word in book:
            self.trie.insert(word)

    def get(self, word: str) -> int:
        return self.trie.search(word)
