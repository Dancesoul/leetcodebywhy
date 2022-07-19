#coding=utf-8
#Author miracle.why@qq.com 
class Solution(object):
    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        maxnum=max(nums)
        print(maxnum)
        maxbool=True
        for v in nums:
            if v==maxnum:
                continue
            if 2*v<maxnum or 2*v == maxnum:
                continue
            else:
                maxbool=False
                break
        if maxbool==True:
            return nums.index(maxnum)
        else:
            return -1




