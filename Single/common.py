#coding=utf-8
#Author miracle.why@qq.com 
#存放公共的类
#ListNode  链表
#TreeNode  树

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None