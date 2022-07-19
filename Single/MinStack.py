#coding=utf-8
#Author miracle.why@qq.com 
class MinStack(object):

	def __init__(self):
		"""
		initialize your data structure here.
		"""
		self.stack=[]

	def push(self, x):
		"""
		:type x: int
		:rtype: None
		"""
		self.stack.append(x)

	def pop(self):
		"""
		:rtype: None
		"""
		self.stack.pop()

	def top(self):
		"""
		:rtype: int
		"""
		return self.stack[len(self.stack)-1]

	def getMin(self):
		"""
		:rtype: int
		"""
		return min(self.stack)

