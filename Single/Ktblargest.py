#coding=utf-8
#Author miracle.why@qq.com
import heapq
class KthLargest:
	def __init__(self, k, nums):
		self.pool = nums
		heapq.heapify(self.pool)
		self.k = k
		while len(self.pool) > k:
			heapq.heappop(self.pool)

	def add(self, val):
		if len(self.pool) < self.k:
			heapq.heappush(self.pool, val)
		elif val > self.pool[0]:
			heapq.heapreplace(self.pool, val)
		return self.pool[0]
# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)

k=KthLargest(3, [4,5,8,2])
print(k.add(3))
print(k.add(5))
print(k.add(10))

print(k.add(9))