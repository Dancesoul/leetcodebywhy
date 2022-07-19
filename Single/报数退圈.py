#coding=utf-8
#Author miracle.why@qq.com 

#有n个人围成一圈，顺序排号。从第一个人开始报数（从1到3报数），凡报到3的人退出圈子，问最后留下的人是原来第几号

import random

num=random.randint(3,100)
startlist=[]  #原始人
for v in range(1,num):
    startlist.append(v)

roundlist=startlist.copy()
peoplenum=num
outnum = 1  # 报数
while(peoplenum>1):
    i = 0  # 列表索引  当前人的位置
    while(i<len(roundlist)):
        if outnum==3:
            roundlist[i]="out"
            outnum=1

        else:
            outnum+=1
        i+=1
    print(roundlist)
    while('out' in roundlist):
        roundlist.remove('out')
    peoplenum=len(roundlist)
print(roundlist)