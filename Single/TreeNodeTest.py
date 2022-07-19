#coding=utf-8
#Author miracle.why@qq.com
import binarytree
class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None

class Solution:
	def sumNumbers(self, root: TreeNode) -> int:

		def helper(root:TreeNode,pre:str):
			if root is None:
				return int(pre)
			return helper(root.left,pre+str(root.val))+helper(root.right,pre+str(root.val))
		return helper(root,"0")



if __name__ == '__main__':
	t=TreeNode([1,2,3])
	s=Solution()
	print(s.sumNumbers(t))