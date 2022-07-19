#coding=utf-8
#Author miracle.why@qq.com
from typing import *
class NumArray:

	def __init__(self, nums: List[int]):
		self.sums=[i for i in range(len(nums)+1)]
		for num in range(0,len(nums)+1):
			self.sums[num+1]=nums[num]+self.sums[num]
	def sumRange(self, i: int, j: int) -> int:
			return self.sums[j+1]-self.sums[i]

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)