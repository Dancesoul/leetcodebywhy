# -*- coding:utf-8 -*-
# @FileName  :LCP 44. 开幕式焰火.py
# @Time      :2023/1/5 0005 18:43
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
# Description:

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def numColor(self, root: TreeNode) -> int:
        memo = []
        def helper(root):
            if root.val not in memo and root.val is not None:
                memo.append(root.val)
            if root.left:
                helper(root.left)
            if root.right:
                helper(root.right)
        helper(root)
        return len(memo)
