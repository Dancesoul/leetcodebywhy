#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ScriptName: Solutions3
Project: yuncaiTest
Author: Miracle.why.
Create Date: 2021-02-19 10:23:50
Description:
"""
__author__ = 'miracle.why@qq.com'

import string
from typing import *


class Solution:
    def tribonacci(self, n: int) -> int:
        memo = {}
        def helper(n):
            if n in memo:
                return memo[n]

            if n == 0 or n==1:
                memo[n] = n

            elif n == 2:
                memo[n] = 1

            else:
                memo[n]= helper(n-1)+helper((n-2))+helper(n-3)
            return memo[n]

        return helper(n)


    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        # 766. 托普利茨矩阵
        m = len(matrix)
        n = len(matrix[0])
        if m ==1 :
            return True

        exel = matrix[0]
        i=1
        while(i<m):   # 行
            j=1
            while(j<n):  # 列
                if matrix[i][j] != exel[j-1]:
                    return False
                j+=1
            exel = matrix[i]
            i+=1
        return True


    def calculate(self, s: str) -> int:
        # 224. 基本计算器
        res, num, sign = 0, 0, 1
        stack = []
        for c in s:
            if c.isdigit():
                num = 10 * num + int(c)
            elif c == "+" or c == "-":
                res += sign * num
                num = 0
                sign = 1 if c == "+" else -1
            elif c == "(":
                stack.append(res)
                stack.append(sign)
                res = 0
                sign = 1
            elif c == ")":
                res += sign * num
                num = 0
                res *= stack.pop()
                res += stack.pop()
        res += sign * num
        return res

    def getSumAbsoluteDifferences(self, nums: List[int]) -> List[int]:
        # 1685. 有序数组中差绝对值之和
        lens = len(nums)
        res = [0] * lens
        s = sum(nums)
        n = 0
        for i in range(lens):
            res[i] = s - nums[i]*lens + (i * nums[i]-n)*2
            n+=nums[i]
        return res


    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
        # 1178. 猜字谜
        import collections
        frequency = collections.Counter()

        for word in words:
            mask = 0
            for ch in word:
                mask |= (1 << (ord(ch) - ord("a")))
            if str(bin(mask)).count("1") <= 7:
                frequency[mask] += 1

        ans = list()
        for puzzle in puzzles:
            total = 0

            # 枚举子集方法一
            # for choose in range(1 << 6):
            #     mask = 0
            #     for i in range(6):
            #         if choose & (1 << i):
            #             mask |= (1 << (ord(puzzle[i + 1]) - ord("a")))
            #     mask |= (1 << (ord(puzzle[0]) - ord("a")))
            #     if mask in frequency:
            #         total += frequency[mask]

            # 枚举子集方法二
            mask = 0
            for i in range(1, 7):
                mask |= (1 << (ord(puzzle[i]) - ord("a")))

            subset = mask
            while subset:
                s = subset | (1 << (ord(puzzle[0]) - ord("a")))
                if s in frequency:
                    total += frequency[s]
                subset = (subset - 1) & mask

            # 在枚举子集的过程中，要么会漏掉全集 mask，要么会漏掉空集
            # 这里会漏掉空集，因此需要额外判断空集
            if (1 << (ord(puzzle[0]) - ord("a"))) in frequency:
                total += frequency[1 << (ord(puzzle[0]) - ord("a"))]

            ans.append(total)

        return ans

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        # 90. 子集 II
        lens = len(nums)
        import itertools
        res = [[]]
        for i in range(1,lens+1):
            for v in itertools.combinations(nums,i):
                if list(v) not in res:
                    res.append(list(v))
        return res

    def hammingWeight(self, n: int) -> int:
        # 191. 位1的个数
        i,j=0,31
        res = 0

        while(i<j):
            if n & (1<<i):
                res+=1
            if n & (1<<j):
                res+=1
            i+=1
            j-=1
        return res


    def findIntegers(self, num: int) -> int:
        # 600. 不含连续1的非负整数
        res = 0
        for i in range(num):
            if bin(i).find("11") != -1:
                res+=1
        return res

    def hammingDistance(self, x: int, y: int) -> int:
        # 461. 汉明距离
        return bin(x^y).count("1")


    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        # 852. 山脉数组的峰顶索引
        lens = len(arr)
        i = 0
        j = lens-1
        while(i < j):
            if arr[i] < arr[i+1]:
                i+=1
            else:
                return i
            if arr[j] < arr[j-1]:
                j-=1
            else:
                return j
        return i

    def minPairSum(self, nums: List[int]) -> int:
        # 1877. 数组中最大数对和的最小值
        nums.sort()
        n = len(nums)
        num = 0
        for i in range(n // 2):
            num = max(num,nums[i]+nums[n-1-i])
        return num

    def trap(self, height: List[int]) -> int:
        # 42. 接雨水
        lens = len(height)
        left, right = [0] * (lens + 1), [0] * (lens + 1)
        ans = 0
        for i in range(1, len(height) + 1):   # 求出从1开始每个位置的左侧最大值
            left[i] = max(left[i - 1], height[i - 1])

        for i in range(len(height) - 1, 0, -1):   # 求出从最后一个（lens-1）开始 右侧最大值
            right[i] = max(right[i + 1], height[i])

        for i in range(len(height)):  # 遍历整个数组， 求当前位置能集的水 = 最小的左右最大值 - 当前高度
            ans += max(0, min(left[i + 1], right[i]) - height[i])
        return ans

    def maxLengthBetweenEqualCharacters(self, s: str) -> int:
        # 1624. 两个相同字符之间的最长子字符串
        lens = len(s)
        if lens == len(set(s)):
            return -1

        i=0
        maxnum = 0
        while i<lens:
            j=i+1
            while j<lens:
                if s[i] == s[j]:
                    maxnum = max(maxnum,j-i-1)
                j+=1
            i+=1
        return maxnum

    def kthFactor(self, n: int, k: int) -> int:
        # 1492. n 的第 k 个因子
        factor = [1,n]
        i = 2
        j = n
        while i <= j :
            num1 = n % i   # 本次检查是不是因数
            j = n // i # j应该到的位置
            if num1 == 0:
                factor.append(i)
                factor.append(j)
            i+=1
        factor = list(set(factor))
        factor.sort()
        print(factor)
        if k > len(factor):
            return -1
        else:
            return factor[k-1]

    def closestDivisors(self, num: int) -> List[int]:
        # 1362. 最接近的因数
        import math
        num1 = num + 1
        num2 = num + 2
        mid = math.sqrt(num1)  # 开根 如果直接得到一个整数 那么直接返回, 如果有小数，那么从mid处开始使用快慢指针，向两边扫描
        mid2 = math.sqrt(num2)  # 第二个数开根
        print(mid,mid2)

    def getStrongest(self, arr: List[int], k: int) -> List[int]:
        # 1471 1471. 数组中的 k 个最强值
        arr = sorted(arr)
        lens = len(arr)
        mid = arr[int((lens - 1) // 2)]

        for i in range(lens):
            arr[i] = [abs(arr[i] - mid),arr[i]]

        arr = sorted(arr,reverse=True,key=lambda x:(x[0],x[1]))
        res= []
        for j in range(lens):
            if j > k-1:
                break
            res.append(arr[j][1])
        return res

    def strToInt(self, str: str) -> int:
        # 剑指 Offer 67. 把字符串转换成整数
        import string
        max_int = 2**31 - 1
        min_int = -2**31
        res = str.split()
        if res:
            if res[0][0] in string.digits or res[0][0] in ["-","+"]:
                symbol = ""
                temp = res[0]
                ans = ""
                if temp[0] in ["-","+"]:
                    symbol = temp[0]
                    temp = temp[1:]

                for i in temp:
                    if i not in string.digits:
                        break
                    else:
                        ans += i
                if ans != "":
                    ans = int(symbol+ans)
                    if ans >= max_int:
                        return max_int
                    elif ans <= min_int:
                        return min_int
                    else:
                        return ans
        return 0

    def isIsomorphic(self, s: str, t: str) -> bool:
        # 205. 同构字符串
        lens = len(s)
        lib = {"l":{},"r":{}}

        for i in range(lens):
            a = s[i]
            b = t[i]
            if a not in lib["l"]:
                lib["l"][a] = b
            if b not in lib["r"]:
                lib["r"][b] = a
            if lib["l"][a] == b and lib["r"][b] == a:
                continue
            else:
                return False
        return True

    def partitionDisjoint(self, nums: List[int]) -> int:
        # 915. 分割数组 左边的最大都小于等于右边的最小 那么左边全都小于等于右边
        if len(nums) == 2:
            return 1
        minindex = nums.index(min(nums))
        maxindex = nums.index(max(nums[minindex+1:]))
        i = minindex
        j = maxindex
        # 结果肯定在最小值和最大值之间， 因为总是有结果  所有最小一定在最大的的左边,即 最小值左边都应该在左边，最大值右边的，都应该在右边
        maxnum = max(nums[:i+1])
        minnum = min(nums[i+1:])
        r = sorted(nums[i+1:])
        while i<j:
            if maxnum <=minnum:
                return i+1
            i += 1
            r.remove(nums[i])
            minnum = r[0]
            maxnum = max(maxnum,nums[i])

        return i

    def interpret(self, command: str) -> str:
        # 1678. 设计 Goal 解析器
        return command.replace("()","o").replace("(al)","al")

    def nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
        # 1201. 丑数 III  有问题
        res= []
        l = sorted([a,b,c])
        a = l[0]
        b = l[1]
        c = l[2]
        ab = (b-a)//a # a-b之间a的数量
        bc = (c-b)//b # b-c之间b的数量
        x =1  # a的数量
        y =1  # b的数量
        z =1  # c的数量
        if n <=3:
            return l[n-1]

        while x+y+z <= n:
            abi = 0
            while abi <=ab:
                res.append(a*x)
                abi+=1
                x+=1
            bci = 0
            while bci <=bc:
                res.append(b*y)
                bci+=1
                y+=1
            res.append(c*z)
            z+=1
        return res[n-1]

    def findLongestSubarray(self, array: List[str]) -> List[str]:
        # 面试题 17.05.  字母与数字
        i = 0
        j = 0
        ind = 0
        lens = len(array)
        res = 0
        while ind < lens:
            if array[ind] in string.ascii_letters:
                i+=1
            else:
                j+=1
            if i == j:
                res=ind
            ind+=1
        if res:
            return array[:res+1]
        else:
            return []

    def findComplement(self, num: int) -> int:
        # 476. 数字的补数

        b = str(bin(num)[2:])
        res = ""
        for bb in b:
            res = res + str(1-int(bb))
        return int(res,base=2)

    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        # 剑指 Offer 04. 二维数组中的查找
        r = len(matrix) -1
        j = 0
        while r >=0 and j <len(matrix[0]):
            temp = matrix[r][j]
            if temp == target:
                return True
            elif temp > target:
                r -= 1
            else:
                j += 1
        return False

    def plusOne(self, digits: List[int]) -> List[int]:
        # 66. 加一
        j = len(digits) - 1
        quan = 0
        while j >= 0 :
            if quan == 1:
                temp = digits[j] + 1
                quan = 0
            else:
                temp = digits[j] + 1

            if temp == 10:
                quan = 1
                digits[j] = 0
                j -=1
            else:
                digits[j] = temp
                return digits
        if quan == 1:
            digits.insert(0,1)
        return digits

    def minDominoRotations(self, tops: List[int], bottoms: List[int]) -> int:
        # 1007. 行相等的最少多米诺旋转
        from collections import Counter
        all = tops + bottoms
        lens = len(tops)
        c = Counter(all)
        value = c.most_common(1)[0][0] #最多的元素
        v_lens = c.most_common(1)[0][1]  #最多元素的个数 小于一半则无法成功
        if v_lens<lens:
            return -1
        i = 0
        top_v = []
        bottoms_v = []
        while i < lens:
            if tops[i] == value:
                top_v.append(i)
            if bottoms[i]  == value:
                bottoms_v.append(i)
            if bottoms[i] !=value and tops[i] !=value:
                return -1
            i+=1
        t_lens = len(top_v)
        b_lens = len(bottoms_v)
        return lens - max(t_lens,b_lens)

    def multiply(self, A: int, B: int) -> int:
        # 面试题 08.05. 递归乘法
        if A >= B:
            base = A
            x = B
        else:
            base = B
            x = A
        if x == 1:
            return base
        else:
            if x%2 ==0:
                return self.multiply(base,x//2) + self.multiply(base,x//2)
            else:
                return self.multiply(base,x//2) + self.multiply(base,x//2)+self.multiply(base,1)

    def majorityElement(self, nums: List[int]) -> List[int]:
        # 229. 求众数 II
        lens = len(nums)
        c = Counter(nums)
        res = []
        for key,ind in c.most_common():
            if ind > lens/3 :
                res.append(key)
            else:
                break
        return res

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        #  496. 下一个更大元素 I
        nums1 = {i:-1 for i in nums1}
        i = 0
        lens = len(nums2)
        while i < lens:
            temp = nums2[i]
            if temp in nums1:
                j = i+1
                while j < lens:
                    if nums2[j] > temp:
                        nums1[temp] = nums2[j]
                        break
                    j+=1
            i+=1
        return list(nums1.values())

    def missingNumber(self, nums: List[int]) -> int:
        # 268. 丢失的数字
        lens = len(nums)
        return (set(range(lens+1)) - set(nums)).pop()

    def findNthDigit(self, n: int) -> int:
        # 400. 第 N 位数字
        d, count = 1, 9
        while n > d * count:
            n -= d * count
            d += 1
            count *= 10
        index = n - 1
        start = 10 ** (d - 1)
        num = start + index // d
        digitIndex = index % d
        return num // 10 ** (d - digitIndex - 1) % 10


    def maxPower(self, s: str) -> int:
        # 1446. 连续字符
        lens = len(s)
        i = 1
        maxnum = 1
        last = s[0]
        now = 1
        while i < lens:
            if last != s[i]:
                maxnum = max(maxnum,now)
                last = s[i]
                now = 1
            else:
                now+=1
            i+=1
        return max(maxnum,now)

    def subsets(self, nums: List[int]) -> List[List[int]]:
        # 78. 子集
        import itertools
        return  list(itertools.product("".join(nums),repeat=2))

    def minimumDifference(self, nums: List[int], k: int) -> int:
        # 1984. 学生分数的最小差值
        if k==1:
            return 0
        nums.sort()
        i = 0
        lens = len(nums)
        minnum = float("inf")
        while i+k-1<lens:
            minnum = min(minnum,nums[k-1 + i]-nums[i])
            i+=1

        return minnum

    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """





if __name__ == '__main__':

    s = Solution()
    print(s.minimumDifference(nums = [9,5,1,6], k = 2))