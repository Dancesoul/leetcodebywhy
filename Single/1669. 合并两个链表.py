# -*- coding:utf-8 -*-
# @FileName  :1669. 合并两个链表.py
# @Time      :2023/1/30 0030 10:18
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
# Description:
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def __init__(self):
        self.last = None
        self.anode = None
        self.bnode = None

    def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:

        def helper(node1: ListNode, i: int):
            if i + 1 == a:
                self.anode = node1
            if i - 1 == b:
                self.bnode = node1
                return
            helper(node1.next, i + 1)

        def helper2(node2: ListNode):
            if node2 is None:
                return
            self.last = node2
            helper2(node2.next)

        helper(list1, 0)
        helper2(list2)
        self.anode.next = list2
        self.last.next = self.bnode
        return list1
