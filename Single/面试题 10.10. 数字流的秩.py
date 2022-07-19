# -*- coding:utf-8 -*-
# @FileName  :面试题 10.10. 数字流的秩.py
# @Time      :2021/10/12 0012 16:11
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
# Description:
class StreamRank:

    stream = []
    def __init__(self):
        pass

    def track(self, x: int) -> None:
        self.stream.append(x)
        self.stream.sort()

    def getRankOfNumber(self, x: int) -> int:
        if x not in self.stream:
            self.stream.append(x)
            return self.stream.index(x)
        else:
            return self.stream.index(x)+1


# Your StreamRank object will be instantiated and called as such:
# obj = StreamRank()
# obj.track(x)
# param_2 = obj.getRankOfNumber(x)