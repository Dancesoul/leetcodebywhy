# -*- coding:utf-8 -*-
# @FileName  :剑指 Offer II 052. 展平二叉搜索树.py
# @Time      :2023/1/14 0014 14:30
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
# Description:
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def __init__(self):
        self.myroot = TreeNode(-1)

    def increasingBST(self, root: TreeNode) -> TreeNode:
        roottemp = TreeNode(-1)
        self.myroot = roottemp

        def helper(curroot: TreeNode):
            if curroot is None:
                return
            helper(curroot.left)
            self.myroot.right = curroot
            self.myroot = self.myroot.right
            curroot.left = None
            helper(curroot.right)

        helper(root)
        return roottemp.right

