# -*- coding:utf-8 -*-
# @FileName  :面试题 03.06. 动物收容所.py
# @Time      :2023/8/9 0009 13:51
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
""" 
    Description:   
"""
from typing import *


class AnimalShelf:

    def __init__(self):
        self.animal = []

    def enqueue(self, animal: List[int]) -> None:
        self.animal.append(animal[0])

    def dequeueAny(self) -> List[int]:
        return self.animal.pop(0)

    def dequeueDog(self) -> List[int]:
        for i,ani in enumerate(self.animal):
            if ani[1] == 1:
                return self.animal.pop(i)
        return [-1, -1]

    def dequeueCat(self) -> List[int]:
        for i,ani in enumerate(self.animal):
            if ani[1] == 0:
                return self.animal.pop(i)
        return [-1, -1]

# Your AnimalShelf object will be instantiated and called as such:
# obj = AnimalShelf()
# obj.enqueue(animal)
# param_2 = obj.dequeueAny()
# param_3 = obj.dequeueDog()
# param_4 = obj.dequeueCat()
