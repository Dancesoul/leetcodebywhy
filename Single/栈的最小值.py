# -*- coding:utf-8 -*-
# @FileName  :栈的最小值.py
# @Time      :2021/10/9 0009 15:50
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
# Description:

class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.mimstack = []

    def push(self, x: int) -> None:
        self.stack.insert(0,x)
        if not self.mimstack:
            self.mimstack.insert(0,x)
        else:
            minnum = self.mimstack[0]
            if x <= minnum:
                self.mimstack.insert(0,x)
            else:
                self.mimstack.insert(0,minnum)


    def pop(self) -> None:
        self.mimstack.pop(0)
        self.stack.pop(0)

    def top(self) -> int:
        return self.stack[0]

    def getMin(self) -> int:
        return self.mimstack[0]


# Your MinStack object will be instantiated and called as such:
minStack = MinStack()
print(minStack.push(-2))
print(minStack.push(0))
print(minStack.push(-3))
print(minStack.getMin())
print(minStack.pop())
print(minStack.top())
print(minStack.getMin())
