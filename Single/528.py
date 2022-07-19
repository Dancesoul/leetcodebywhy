#coding=utf-8
#Author miracle.why@qq.com
from typing import *
class Solution:

	def __init__(self, w: List[int]):
		self.wl=[]
		self.w=w
		lens=sum(w)
		for i in w:
			self.wl+=[i for _ in range(i) ]

	def pickIndex(self) -> int:
		import random
		return self.w.index(random.choice(self.wl))

if __name__ == '__main__':
	s=Solution([1])
	print(s.pickIndex())