# -*- coding:utf-8 -*-
# @FileName  :剑指 Offer II 041. 滑动窗口的平均值.py
# @Time      :2023/1/30 0030 15:24
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
# Description:
class MovingAverage:

    def __init__(self, size: int):
        """
        Initialize your data structure here.
        """
        self.window = []
        self.full = size

    def next(self, val: int) -> float:
        self.window.append(val)
        if len(self.window) > self.full:
            self.window.pop(0)
        return sum(self.window)/len(self.window)


# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)