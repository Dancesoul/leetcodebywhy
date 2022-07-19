# -*- coding:utf-8 -*-
# @FileName  :n个顾客打折.py
# @Time      :2021/10/9 0009 16:24
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
# Description:
from typing import *
class Cashier:

    i = 1
    def __init__(self, n: int, discount: int, products: List[int], prices: List[int]):
        self.n = n
        self.discount = discount
        self.products = {}
        l = len(products)
        for ll in range(l):
            self.products[products[ll]] = prices[ll]


    def getBill(self, product: List[int], amount: List[int]) -> float:
        lens = len(product)
        sum_price = 0
        for p in range(lens):
            if product[p] not in self.products:
                continue
            sum_price += self.products[product[p]] * amount[p]
        if self.i == self.n:
            sum_price = sum_price - sum_price * (self.discount / 100)
            self.i =0
        self.i+=1
        return sum_price
cashier = Cashier(3,50,[1,2,3,4,5,6,7],[100,200,300,400,300,200,100])
print(cashier.getBill([1,2],[1,2]))
print(cashier.getBill([3,7],[10,10]))
print(cashier.getBill([1,2,3,4,5,6,7],[1,1,1,1,1,1,1]))
print(cashier.getBill([4],[10]))
print(cashier.getBill([7,3],[10,10]))
print(cashier.getBill([7,5,3,1,6,4,2],[10,10,10,9,9,9,7]))
print(cashier.getBill([2,3,5],[5,3,2]))