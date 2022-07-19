#coding=utf-8
#Author miracle.why@qq.com
import random
class Solution(object):
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums=nums

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        nums=self.nums
        return nums
    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """
        nums=self.nums[:]
        res=[]
        while(nums):
            lucky=random.choice(nums)
            res.append(lucky)
            nums.remove(lucky)
        return res
# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()
t=Solution([1,2,3])
print(t.shuffle())
print(t.reset())