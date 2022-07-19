#coding=utf-8
#Author 815016453@qq.com 
import random
import re
import string


randomstr=random.sample(string.ascii_letters+string.digits,60)
randomstr="".join(randomstr)
print(randomstr)
patten=re.compile("[0-9]+")
res=re.findall(patten,randomstr)
#print(res)
lenres=[]
for lenth in res:
    lenres.append(len(lenth))
#print(lenres)
print(max(lenres))
