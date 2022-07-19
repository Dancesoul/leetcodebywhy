#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ScriptName: 面试题 02.01. 移除重复节点
Project: yuncaiTest
Author: Miracle.why.
Create Date: 2021-02-06 10:42:48
Description:
"""
__author__ = 'miracle.why@qq.com'


class TripleInOne:

    def __init__(self, stackSize: int):
        self.stackSize = stackSize
        self.stack=  [[],[],[]]

    def push(self, stackNum: int, value: int) -> None:
        if len(self.stack[stackNum]) == self.stackSize:
            return
        self.stack[stackNum].append(value)

    def pop(self, stackNum: int) -> int:
        if self.isEmpty(stackNum):
            return -1
        return self.stack[stackNum].pop()

    def peek(self, stackNum: int) -> int:
        if self.isEmpty(stackNum):
            return -1
        return self.stack[stackNum][0]

    def isEmpty(self, stackNum: int) -> bool:
        return len(self.stack[stackNum]) is 0


# Your TripleInOne object will be instantiated and called as such:
obj = TripleInOne(1)
print(obj.stack)
obj.push(0,1)
obj.push(0,2)

print(obj.stack)