# -*- coding:utf-8 -*-
# @FileName  :剑指 Offer II 056. 二叉搜索树中两个节点之和.py
# @Time      :2023/1/12 0012 17:45
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
# Description:
"""
二叉搜索树性质
设x是二叉搜索树中的一个结点。如果y是x左子树中的一个结点，那么y.key≤x.key。如果y是x右子树中的一个结点，那么y.key≥x.key。
在二叉搜索树中：
    1.若任意结点的左子树不空，则左子树上所有结点的值均不大于它的根结点的值。
    2.若任意结点的右子树不空，则右子树上所有结点的值均不小于它的根结点的值。
    3.任意结点的左、右子树也分别为二叉搜索树。
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def findTarget(self, root: TreeNode, k: int) -> bool:
        """
        剑指 Offer II 056. 二叉搜索树中两个节点之和
        利用二叉搜索树性质， 找 x =  k- val
            1  x > val  就进入右叉树
                如果右叉树是空的 那么这个x 无法构成答案
            2 x < val 就进入左叉树
                如果左子树是空的， 那么这个x 无法构成答案
            3 x = val 那么这个x 无法构成答案  因为二叉搜索树没有相同的

            进入下一个循环查找 如果x 是None 就是新的循环
            全部结束没有找到 那么返回false

        :param root:
        :param k:
        :return:
        """
        memo = []
        def helper(root:TreeNode,k):
            if root is None:
                return False
            if k - root.val in memo:
                return True
            memo.append(root.val)
            return helper(root.left,k) or helper(root.right,k)



if __name__ == '__main__':
    s = Solution()
    root = TreeNode([8,6,10,5,7,9,11])
    print(s.findTarget(root = root, k = 12))