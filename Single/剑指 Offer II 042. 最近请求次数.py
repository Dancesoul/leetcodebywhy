# -*- coding:utf-8 -*-
# @FileName  :剑指 Offer II 042. 最近请求次数.py
# @Time      :2023/1/28 0028 15:23
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
# Description:
class RecentCounter:

    def __init__(self):
        self.res = []

    def ping(self, t: int) -> int:
        self.res.append(t)
        while self.res[0] < t -3000:
            self.res.pop(0)
        return len(self.res)


# Your RecentCounter object will be instantiated and called as such:
# obj = RecentCounter()
# param_1 = obj.ping(t)
if __name__ == '__main__':
    r = RecentCounter()
    print(r.ping(1))
    print(r.ping(100))
    print(r.ping(3001))
    print(r.ping(3002))
    print(r.ping(100000))
