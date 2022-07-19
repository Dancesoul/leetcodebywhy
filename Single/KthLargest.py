# -*- coding:utf-8 -*-
# @FileName  :KthLargest.py
# @Time      :2021/11/30 0030 15:59
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
# Description:
from typing import *
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.stack = sorted(nums,reverse=True)[:k]

    def add(self, val: int) -> int:
        if val > self.stack[-1]:
            self.stack.pop(-1)
            self.stack.append(val)
            self.stack.sort(reverse=True)
        return self.stack[-1]



# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)