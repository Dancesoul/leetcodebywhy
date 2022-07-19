#coding=utf-8
#Author miracle.why@qq.com 
import random

'''
扑克54张 1-K 各四张  王两张
三张底
'''
l=[]
p1=[]
p2=[]
p3=[]
di=[]
for i in range(1,14):
    l.append(i)
pokelist=l+l+l+l
pokelist.append("smalljoke")
pokelist.append("bigjoke")
random.shuffle(pokelist)

p1=pokelist[0:17]
p2=pokelist[17:17+17]
p3=pokelist[17+17:17+17+17]
di=pokelist[-3:]
#print(len(p1),len(p2),len(p3),di)
print(p1,p2,p3,di)