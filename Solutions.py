# coding=utf-8
# Author miracle.why@qq.com
import collections
import string
from collections import Counter
from typing import *

import numpy
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        maxnum = max(nums)
        print(maxnum)
        maxbool = True
        for v in nums:
            if v == maxnum:
                continue
            if 2 * v < maxnum or 2 * v == maxnum:
                continue
            else:
                maxbool = False
                break
        if maxbool == True:
            return nums.index(maxnum)
        else:
            return -1

    def shortestCompletingWord(self, licensePlate, words):
        """
        :type licensePlate: str
        :type words: List[str]
        :rtype: str
        """
        res = []
        for word in words:
            for i in word:
                if i in licensePlate:
                    if word.count(i) == licensePlate.count(i):
                        continue
                    else:
                        break
                else:
                    break
            licensePlate = licensePlate.lower()
            afterset = set(licensePlate)
            for i in afterset:
                licensePlate.count(i)

    def repeatedNTimes(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        n1 = len(A) / 2
        n2 = len(set(A)) - 1
        if n1 == n2:
            n = n1
            for i in A:
                if A.count(i) == n:
                    return i
                else:
                    continue
            return -1
        else:
            return -1

    def numRabbits(self, answers):
        """
        :type answers: List[int]
        :rtype: int
        """
        answers.sort()
        result = 0
        i = 0
        while i < len(answers):
            count = answers[i:].count(answers[i])
            result += (answers[i] + 1)
            i += min(count, answers[i] + 1)
        return result

    def threenum(self, nums):
        """
        :type nums: List[int]
        """
        res = []
        for a in nums:
            if a == 0:
                continue
            for b in nums:
                for c in nums:
                    res.append(a * 100 + b * 10 + c)
        return set(res)

    def canMeasureWater(self, x, y, z):
        """
        :type x: int
        :type y: int
        :type z: int
        :rtype: bool
        """
        if z == 0:
            return True
        if x + y < z:
            return False
        if x > y:
            smaller = y
        else:
            smaller = x
        gong = None
        for i in range(1, smaller + 1):
            if ((x % i == 0) and (y % i == 0)):
                gong = i
        if gong != None:
            if z % gong == 0:
                return True
            else:
                return False
        else:
            return False

    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        在给定的网格中，每个单元格可以有以下三个值之一：

值 0 代表空单元格；
值 1 代表新鲜橘子；
值 2 代表腐烂的橘子。
每分钟，任何与腐烂的橘子（在 4 个正方向上）相邻的新鲜橘子都会腐烂。

返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1。
        """
        # 先找烂橘子 也就是2
        # 腐烂是 相邻 并且是同列同行
        badnum = 0
        goodnum = 0
        for bigvalue in grid:
            badnum = badnum + bigvalue.count(2)
            goodnum = goodnum + bigvalue.count(1)
        if goodnum == 0:
            return 0

        if badnum == 0:
            return -1

        minute = 0
        while (goodnum != 0):
            polute = False
            for y, bigvalue in enumerate(grid):
                for x, smallvalue in enumerate(bigvalue):
                    if smallvalue == 2:
                        # 开始污染
                        if x != 0:
                            if grid[y][x - 1] == 1:
                                grid[y][x - 1] = 3
                                goodnum = goodnum - 1
                                polute = True
                        if x != len(bigvalue) - 1:
                            if grid[y][x + 1] == 1:
                                grid[y][x + 1] = 3
                                goodnum = goodnum - 1
                                polute = True
                        if y != 0:
                            if grid[y - 1][x] == 1:
                                grid[y - 1][x] = 3
                                goodnum = goodnum - 1
                                polute = True
                        if y != len(grid) - 1:
                            if grid[y + 1][x] == 1:
                                grid[y + 1][x] = 3
                                goodnum = goodnum - 1
                                polute = True

            if polute == False:
                return -1
            minute += 1
            for y, bigvalue in enumerate(grid):
                for x, smallvalue in enumerate(bigvalue):
                    if smallvalue == 3:
                        grid[y][x] = 2

        return minute

    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = ""
        resdict = {}
        sets = set(s)
        print(sets)
        for key in sets:
            resdict[key] = s.count(key)

        num = 0
        while (num <= len(sets)):
            for key in resdict.keys():
                if resdict[key] == max(list(resdict.values())) and resdict[key] != 0:
                    for i in range(resdict[key]):
                        res = res + key
                    resdict[key] = 0
            num += 1
        return res

    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        """
        给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的一个字母异位词。

示例 1:

输入: s = "anagram", t = "nagaram"
输出: true
        """
        from collections import Counter
        return Counter(s) == Counter(t)

    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        # 如果涉及到对顺序无要求的题目，可以考虑dict
        dict_p = {}  # 构建dict
        for i in p:
            dict_p[i] = dict_p.get(i, 0) + 1  # 将每个字符作为key，对应出现次数作为value，添加至dict
        dict_s = {}
        list_n = []
        len_p = len(p)  # 后面要重复用到，所以先算出，提高效率
        for i, val in enumerate(s):  # 将每个键值展开进行循环
            dict_s[val] = dict_s.get(val, 0) + 1  # 也是字符 作为key，次数作为value
            if dict_s == dict_p:  # 如果添加完以后刚好完全相同，此时刚好循环了前len_p个，i = len_ - 1，所以返回index为0
                list_n.append(i - len_p + 1)
            if i - len_p + 1 >= 0:  # 当循环次数超过len_p后，就对dict进行操作：
                dict_s[s[i - len_p + 1]] = dict_s.get(s[i - len_p + 1]) - 1  # 每往后循环一次index，就减去前面掉出去的那个字符一次
                if dict_s[s[i - len_p + 1]] == 0:  # 如果在for循环一开始添加进的字符不在目标字符串中，删掉
                    del dict_s[s[i - len_p + 1]]
        return list_n
        '''
        from collections import Counter
        res = []
        l_s = len(s)
        l_num = len(p)
        count_p = Counter(p)

        for i, v in enumerate(s):
            if v not in p:
                continue
            if l_s - i < len(p):
                continue
            if Counter(s[i:i + l_num]) == count_p:
                res.append(i)
        return res
        '''

    def rotatedDigits(self, N: int) -> int:
        import re
        res = 0

        for num in range(1, N + 1):
            patt1 = re.compile('^(?:0|8|1)+$')
            patt = re.compile('^(?:0|8|1|5|2|6|9)+$')
            num = str(num)
            if re.findall(patt, num):
                if re.findall(patt1, num):
                    continue
                else:
                    res += 1
        return res

    def canThreePartsEqualSum(self, A) -> bool:
        sums = sum(A)
        if sums % 3 != 0:
            return False

        avg = sums / 3
        max_l = len(A)
        i = 0
        j = max_l - 1
        first = last = 0
        while i < j:
            if first != avg:
                first += A[i]
                i += 1

            if last != avg:
                last += A[j]
                j -= 1

            if last == avg and first == avg:
                return True
        return False

    def isgys(self, a, b) -> bool:

        if a > b:
            min = b
        else:
            min = a
        for i in range(2, min + 1):
            if a % i == 0 and b % i == 0:
                return True
        return False

    def lastStoneWeightII(self, stones):
        """
        :type stones: List[int]
        :rtype: int
        """
        if len(stones) < 2:
            return stones[0]
        temp = sorted(stones)
        temp.reverse()
        while (True):
            if len(temp) == 1:
                return temp[0]
            y = temp[0]
            temp.remove(temp[0])
            x = temp[0]
            temp.remove(temp[0])
            if y != x:
                temp.append(y - x)
                temp = sorted(temp)
                temp.reverse()
            if len(temp) == 0:
                return 0

    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        import heapq
        return heapq.nlargest(k, nums)[k - 1]

    def validSquare(self, p1, p2, p3, p4):
        """
        :type p1: List[int]
        :type p2: List[int]
        :type p3: List[int]
        :type p4: List[int]
        :rtype: bool
        p1-p2的距离p1p2等于p1p2`2= (p1x-p2x)`2+(p1y-p2y)`2
        勾股定理确定直角
        """
        import math
        p1p2 = math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))
        p2p3 = math.sqrt(math.pow((p2[0] - p3[0]), 2) + math.pow((p2[1] - p3[1]), 2))
        p3p4 = math.sqrt(math.pow((p3[0] - p4[0]), 2) + math.pow((p3[1] - p4[1]), 2))
        p4p1 = math.sqrt(math.pow((p4[0] - p1[0]), 2) + math.pow((p4[1] - p1[1]), 2))
        p1p3 = math.sqrt(math.pow((p1[0] - p3[0]), 2) + math.pow((p1[1] - p3[1]), 2))
        p2p4 = math.sqrt(math.pow((p2[0] - p4[0]), 2) + math.pow((p2[1] - p4[1]), 2))
        distance = [p1p2, p2p3, p3p4, p4p1, p1p3, p2p4]
        setdistance = set(distance)
        if len(setdistance) == 2:
            setdistance = sorted(setdistance)
            s1 = round(math.pow(setdistance[0], 2) * 2, 5)
            s2 = round(math.pow(setdistance[1], 2), 5)
            if s1 == s2:
                return True
            else:
                return False
        else:
            return False

    def spiralMatrixIII(self, R, C, r0, c0):
        """
        :type R: int
        :type C: int
        :type r0: int
        :type c0: int
        :rtype: List[List[int]]
        """
        maxsize = R * C  # 总的有效格子数
        walkspace = 0  # 踩的格子数
        thisround = 1  # 当等于1时，说明走R。等于2时，走C。 等于2时，要换加减号。
        walkround = 1  # 目前的圈 走几格
        walklist = []  # 结果集
        jisuan = True  # 控制加减
        while (True):
            walksize = 0  # 这一圈走了第几格
            while (walkspace < maxsize):

                if r0 < R and c0 < C and r0 >= 0 and c0 >= 0:  # 边界外

                    walklist.append([r0, c0])
                    walkspace += 1
                if walkspace == maxsize:
                    return walklist

                if thisround == 1:
                    if jisuan:
                        c0 += 1
                        walksize += 1
                    else:
                        c0 -= 1
                        walksize += 1
                else:
                    if jisuan:
                        r0 += 1
                        walksize += 1
                    else:
                        r0 -= 1
                        walksize += 1

                if walksize == walkround:
                    if thisround == 1:
                        thisround = 2
                    else:
                        thisround = 1
                        jisuan = not jisuan
                        walkround += 1
                    break


    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        sets=list(set(s))
        lens=len(s)
        max=len(sets)
        maxword = ""
        res=0
        i=0
        while(i<len(s)):
            if s[i] not in maxword:
                maxword=maxword+s[i]
                if len(maxword)==max:
                    return max
                elif len(maxword)>res:
                    res=len(maxword)
                i+=1
            else:
                if len(maxword)>res:
                    res=len(maxword)
                start=s.index(s[i])
                s=s[start+1:]
                if len(s)<=res:
                    return res
                maxword=""
                i=0
        return res

    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        import numpy as np
        return np.median(nums1 + nums2)



    def longestPalindrome(self, s):
        size = len(s)
        if size <= 1:
            return s
        # 二维 dp 问题
        # 状态：dp[i,j]: s[i:j] 包括 i，j ，表示的字符串是不是回文串
        dp = [[False for _ in range(size)] for _ in range(size)]

        longest_l = 1
        res = s[0]

        for i in range(1, size):
            for j in range(i):
                # 状态转移方程：如果头尾字符相等并且中间也是回文
                # 或者中间的长度小于等于 1
                if s[j] == s[i] and (j >= i - 2 or dp[j + 1][i - 1]):
                    dp[j][i] = True
                    if i - j + 1 > longest_l:
                        longest_l = i - j + 1
                        res = s[j:i + 1]
        return res

    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """


        memo = dict()

        def match(s, p):
            print(memo)
            if (s, p) in memo:
                return memo[(s, p)]
            if not p:
                return not s
            first_match = bool(s) and p[0] in [s[0], '.']
            if len(p) >= 2 and p[1] == '*':
                re = match(s, p[2:]) or (first_match and match(s[1:], p))
            else:
                re = first_match and match(s[1:], p[1:])
            memo[(s, p)] = re
            return re

        return match(s, p)

    def lemonadeChange(self, bills):
        # 考虑3种情况
        d = {5: 0, 10: 0, 20: 0}  # 用一个字典存储剩余零钱
        for i in bills:
            if i == 5:  # 付5美元，零钱中5美元数量+1
                d[5] += 1
            if i == 10:  # 付10美元只能通过5美元找零，检查5美元数量
                if d[5] == 0:
                    return False
                d[10] += 1
                d[5] -= 1
            if i == 20:  # 付20美元，有两种方式找零，10+5或者5+5+5
                # 通过分析发现，5美元找零消耗量最大，所以找零优先考虑10+5
                if d[10] >= 1 and d[5] >= 1:
                    d[10] -= 1
                    d[5] -= 1
                elif d[5] >= 3:
                    d[5] -= 3
                else:
                    return False
                d[20] += 1
        return True

    def minDeletionSize(self, A):
        """
        :type A: List[str]
        :rtype: int
        """
        minnum=0
        if len(A[0])==1:
            return 0
        for i in range(len(A[0])):
            istr=""
            for j in range(len(A)):
                istr=istr+A[j][i]
            sortedstr=sorted(istr)
            if sortedstr!=list(istr):
                minnum+=1
            else:
                continue

        return minnum

    def arrayNesting(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for i in range(len(nums)):
            if nums[i] == -1:
                continue
            temp = 1
            path_index = i
            while nums[path_index] != i:
                nums[path_index], path_index = -1, nums[path_index]
                temp += 1
            nums[path_index] = -1
            res = max(temp, res)
        return res

    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        len1 = len(num1)
        len2 = len(num2)
        num1=list(num1)
        num1.reverse()
        num2=list(num2)
        num2.reverse()

        res=""
        quan=0
        if len1<len2:
            temp=num1[:]
            num1=num2
            num2=temp
            len1 = len(num1)
            len2 = len(num2)
        for ind in range(len1):
            if ind>(len2-1):
                sum=int(num1[ind])+0+quan

            else:
                sum=int(num1[ind])+int(num2[ind])+quan
            if sum > 9:
                quan = 1
                res = res + str(sum%10)
            else:
                quan=0
                res = res + str(sum)
            if ind==len1-1 and quan==1:
                res=res+"1"
        res=list(res)
        res.reverse()
        res="".join(res)
        return res

    def duplicateZeros(self, arr):
        """
        :type arr: List[int]
        :rtype: None Do not return anything, modify arr in-place instead.
        """
        temp=arr[:]
        findnum=0
        for k,val in enumerate(temp):
            if val ==0:
                arr.pop()
                arr.insert(k+1+findnum,0)
                findnum+=1
            if k==len(temp)-1-findnum:
                break
        print(arr)

    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        import math
        if n<10:
            if n==1 or n==7:
                return True
            else:
                return False
        res=0
        for num in str(n):
            res=res+int(math.pow(int(num),2))
        return self.isHappy(res)

    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if target in nums:
            return nums.index(target)
        else:
            nums.insert(len(nums),target)
            nums.sort()
            return nums.index(target)

    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n==1:
            return "1"
        else:
            round=1
            res = "1"
            while(True):
                if round==n:
                    break
                val=res[0]
                num=0
                temp = ""
                for k,i in enumerate(res):
                    if i==val:
                        num+=1
                    else:
                        temp=temp+str(num)+str(val)
                        num=1
                        val=i
                    if k==len(res)-1:
                        temp = temp + str(num) + str(val)
                res=temp
                round+=1
            return res

    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        intervals.sort()
        n = len(intervals)
        if n == 0:
            return []
        res = [intervals[0]]
        for i in range(1, n):
            if intervals[i][0] <= res[-1][1]:
                res[-1][1] = max(res[-1][1], intervals[i][1])
            else:
                res.append(intervals[i])
        return res

    def uncommonFromSentences(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: List[str]
        """
        res=[]
        from collections import Counter
        alist=A.split()
        blist=B.split()
        c=Counter(alist+blist)

        for i in c:
            if  c[i]==1:
                res.append(i)

        return res

    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        maxx = len(matrix)
        if maxx == 0:
            return matrix
        if maxx == 1:
            return matrix[0]
        maxy = len(matrix[0])
        if maxy == 0:
            return matrix
        num = 1  # 控制遍历个数
        roundnum = 0  # 控制圈数。 上下左右全走了 圈数+1
        x = y = 0  # x,y坐标
        result = [matrix[x][y]]  # 原点

        xmove = False
        ymove = True  # 控制方向 False 是不写，Ture是写
        xwrite = False
        ywrite = True  # 控制加减  Ture是加  Fslse 是减
        while (num < (maxx * maxy)):

            if x == roundnum and y == maxy - roundnum - 1 and ywrite == True:  # 右上角转弯
                xmove = True
                xwrite = True
                ymove = False
                ywrite = False
            elif x == maxx - roundnum - 1 and y == maxy - roundnum - 1 and xwrite == True:  # 右下角转弯
                xmove = False
                xwrite = False
                ymove = True
                ywrite = False
            elif x == maxx - roundnum - 1 and y == roundnum and ywrite == False:  # 左下角转弯
                xmove = True
                xwrite = False
                ymove = False
                ywrite = False
            if xmove == True:
                if xwrite == True:
                    x += 1
                else:
                    x -= 1
            if ymove == True:
                if ywrite == True:
                    y += 1
                else:
                    y -= 1

            if matrix[x][y] not in result:
                result.append(matrix[x][y])
                num += 1
            else:  # 说明处于左上角，应该右转
                xmove = False
                xwrite = False
                ymove = True
                ywrite = True
                roundnum += 1
                x += 1

        return result

    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        x=list(str(x))
        res=""
        fuhao=""
        if x[0]=="-":
            fuhao="-"
            x.pop(0)
            x.reverse()
            res="".join(x)
        else:
            x.reverse()
            res = res.join(x)
        if len(res)>10:
            return 0
        elif len(res)==10:
            if int(res)>2147483648:
                return 0
        return fuhao+str(int(res))
    '''
    暴力遍历法
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        maxarea=0
        for ikey,ival in enumerate(height):
            for jkey,jval in enumerate(height):
                maxarea=max(maxarea,min(ival,jval)*(max(ikey,jkey)-min(ikey,jkey)))
        return maxarea
    '''

    def maxArea(self, height):
        """
        双指针方法
        :type height: List[int]
        :rtype: int
        """
        maxarea=0
        i=0
        j=len(height)-1
        while(i<j):
            maxarea=max(maxarea,min(height[i],height[j])*(j-i))
            if max(height[i],height[j])==height[i]:
                j-=1
            else:
                i+=1
        return maxarea
    def writeRoman(self,num):
        num=0

    '''
    def intToRoman(self, num):
        """
        I             1
V             5
X             10
L             50
C             100
D             500
M             1000

        :type num: int
        :rtype: str
        """

        nums = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        romans = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

        index = 0
        res = ''
        while index < 13:
            # 注意：这里是等于号，表示尽量使用大的"面值"
            while num >= nums[index]:
                res += romans[index]
                num -= nums[index]
            index += 1
        return res
    '''
    def intToRoman(self, num):

        lib = {1: "I", 4: "IV", 5: "V", 9: "IX", 10: "X", 40: "XL", 50: "L", 90: "XC", 100: "C", 400: "CD", 500: "D",
               900: "CM", 1000: "M"}
        res = ""
        temp = list(lib.keys())
        temp.sort(reverse=True)
        for ind in temp:
            while num >= ind:
                res = res + lib[ind]
                num -= ind
        return res

    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        x=str(x)
        lenx=len(x)
        ind=0
        while(ind<int(lenx/2)):
            if x[ind]!=x[lenx-ind-1]:
                return False
            ind+=1
        return True

    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        lib={'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, 'L': 50, 'XC': 90, 'C': 100, 'CD': 400, 'D': 500, 'CM': 900, 'M': 1000}
        ind=0
        res=0
        while(ind<len(s)):
            if s[ind] in lib:
                if  ind<len(s)-1:
                    if s[ind]+s[ind+1] in lib:
                        res=res+lib[s[ind]+s[ind+1]]
                        ind+=2
                        continue
                res=res+lib[s[ind]]
                ind+=1
            else:
                return "-1"
        return res

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) ==0:
            return ""
        if len(strs)==1:
            return strs[0]
        minlen=len(min(strs))
        minstr=min(strs)
        for i in strs:
            if len(i)<minlen:
                minlen=len(i)
                minstr=i
        res=""
        for ind,val in enumerate(minstr):
            for j in strs:
                if j[ind]!=val:
                    return res
            res += val
        return res

    def threeSum(self, nums):
        #15. 三数之和
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res=[]
        nums.sort()
        for key,val in enumerate(nums):
            if val>0:    #如果第一个值没有小于0 最小的都大于0，那么其余相加不可能等于0 直接返回
                break
            if key > 0 and nums[key] == nums[key - 1]:   #用来处理连续相同的值
                continue
            i=key+1       #双指针 i指向头  j指向尾  因为排了序，i就代表小  j代表大
            j=len(nums)-1
            while(i<j):     #双指针逐渐靠近
                sumnum=val+nums[i]+nums[j]
                if sumnum<0:    #如果小于0 那么需要减少小的 i往大的数走
                    i+=1
                    while i < j and nums[i] == nums[i - 1]:   #处理连续相同的值
                        i += 1
                elif sumnum>0:    #如果值大于0  那么需要减小大值
                    j-=1
                    while i < j and nums[j] == nums[j + 1]:  #处理连续相同的值
                        j -= 1
                else:   #等于0 添加到答案列表中
                    res.append([val, nums[i], nums[j]])
                    i+=1
                    j-=1
                    while i < j and nums[i] == nums[i - 1]: #处理连续相同的值
                       i += 1
                    while i < j and nums[j] == nums[j + 1]:  #处理连续相同的值
                       j -= 1

        return res

    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        cha=float('inf')
        res=cha
        for ind,val in enumerate(nums):
            i=ind+1
            j=len(nums)-1
            while(i<j):
                sumnum=val+nums[i]+nums[j]
                if abs(sumnum-target)<cha:
                    cha=abs(sumnum-target)
                    res=sumnum
                if sumnum>target:
                    j-=1
                elif sumnum<target:
                    i+=1
                else:
                    return res
        return res

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        lib={'2': ['a', 'b', 'c'],
             '3': ['d', 'e', 'f'],
             '4': ['g', 'h', 'i'],
             '5': ['j', 'k', 'l'],
             '6': ['m', 'n', 'o'],
             '7': ['p', 'q', 'r','s'],
             '8': ['t', 'u','v'],
             '9': ['w', 'x','y','z']}

        def backtrack(combination, next_digits):
            if len(next_digits) == 0:
                output.append(combination)
            else:
                for letter in lib[next_digits[0]]:
                    backtrack(combination + letter, next_digits[1:])

        output = []
        if digits:
            backtrack("", digits)
        return output

    def numJewelsInStones(self, J, S):
        """
        :type J: str
        :type S: str
        :rtype: int
        """
        import re
        sumnum=0
        for patt in J:
            sumnum+=len(re.findall(patt,S))
        return sumnum

    def toGoatLatin(self, S):
        """
        :type S: str
        :rtype: str
        """
        lib=['a','A','E','e','I','i', 'o','O','u','U']
        strlist=S.split()
        num=0
        for ind,val in enumerate(strlist):
            if val[0] in lib:
                val+="ma"
            else:
                val=val+val[0]
                val=val.replace(val[0],"",1)
                val+="ma"

            i=0
            while(i<num+1):
                val+="a"
                i+=1
            strlist[ind]=val
            num+=1
        return " ".join(strlist)

    def numFriendRequests(self, ages):
        """
        未完成
        :type ages: List[int]
        :rtype: int
        """
        from collections import Counter
        c=Counter(ages)

        agelist=list(c.keys())
        if len(agelist)==1:
            return sum(c.values())
        agelist.sort()
        agelist.reverse()
        if c.get(agelist[0])>1:
            res = c.get(agelist[0])
        else:
            res=0
        for ind1,p1 in enumerate(agelist):
            for p2 in agelist[ind1+1:]:
                if p2 <= 0.5 * p1 + 7  or (p2 > 100 and p1 < 100):
                    continue
                else:
                    res+=c.get(p1)*c.get(p2)
        return res

    def largeGroupPositions(self, S):
        """
        :type S: str
        :rtype: List[List[int]]
        """
        res=[]
        i=0
        while(i<len(S)):
            temp = 1
            if i==len(S)-1:
                break
            if S[i]==S[i+1]:
                while(i+temp<len(S)):
                    if S[i+temp]==S[i]:
                        temp+=1
                    else:
                        break
                if temp>=3:
                    res.append([i,i+temp-1])
            i=i+temp
        return res

    def findPeakElement(self, nums):
        """
        复杂度 logN  二分法
        :type nums: List[int]
        :rtype: int
        """

        return nums.index(max(nums))


    def maximumGap(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)<3:
            return 0
        nums.sort()
        i=0
        maxnum=0
        while(i<len(nums)-1):
            cha=nums[i+1]-nums[i]
            if cha>maxnum:
                maxnum=cha
            i+=1
        return maxnum

    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        #补齐
        v1=version1.split(".")
        v2=version2.split(".")
        v1len=len(v1)
        v2len=len(v2)
        if v1len>=v2len:
            i=v2len
            while(i<v1len):
                v2.append("0")
                i+=1
        else:
            i=v1len
            while(i<v2len):
                v1.append("0")
                i+=1
        for ind in range(len(v1)):
            if int(v1[ind])>int(v2[ind]):
                return 1
            elif int(v1[ind])<int(v2[ind]):
                return -1
            else:
                continue
        return 0

    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """


        lib = {}
        for ind, num in enumerate(numbers):
            lib[num] = ind

        for i, n in enumerate(numbers):
            j = lib.get(target - n)
            if j != i and j != None:
                return [i, j]


    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        if len(str)==0:
            return 0
        import re
        import math
        int_max=int(math.pow(2,31)-1)
        int_min=int(math.pow(-2,31))
        res=[]
        if re.findall('\s',str[0]):
            kongbai=re.findall('[\S]{1}',str)
            if kongbai:
                ind=str.index(kongbai[0])
                if ind<len(str)-1:
                    if re.findall('(?:\+{1}|-{1})',str[ind]):
                        if re.findall('[^0-9]', str[ind+1]):
                            return 0
                        res=re.findall('-?\+?[0-9]+',str[ind:])
                    elif re.findall('[0-9]',str[ind]):
                        res = re.findall('-?\+?[0-9]+', str[ind:])
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0
        elif re.findall('(?:\+{1}|-{1})',str[0]):
            if len(str)<2:
                return 0
            if re.findall('[^0-9]', str[1]):
                return 0
            res = re.findall('-?\+?[0-9]+', str)
        elif re.findall('[0-9]+',str[0]):
            res = re.findall('-?\+?[0-9]+', str)
        else:
            return 0


        if res:
            if len(re.findall('(?:\+{1}|-{1})',res[0]))>1:
                return 0
            res = int(res[0])
            if res > int_max:
                return int_max
            elif res < int_min:
                return int_min
            return res
        else:
            return 0

    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        lib={1:['q','w','e','r','t','y','u','i','o','p'],
             2:['a','s','d','f','g','h','j','k','l'],
             3:['z','x','c','v','b','n','m']}


        res=[]
        for word in words:
            temp=word.lower()
            writebool=True
            if temp[0] in lib[1]:
                wordind = 0
                while(wordind<len(temp)):
                    if temp[wordind] not in lib[1]:
                        writebool=False
                        break
                    wordind += 1
                if writebool:
                    res.append(word)
            elif temp[0] in lib[2]:
                wordind = 0
                while (wordind < len(temp)):
                    if temp[wordind] not in lib[2]:
                        writebool = False
                        break
                    wordind += 1
                if writebool:
                    res.append(word)

            else:
                wordind = 0
                while (wordind < len(temp)):
                    if temp[wordind] not in lib[3]:
                        writebool = False
                        break
                    wordind += 1
                if writebool:
                    res.append(word)

        return res

    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        from collections import Counter
        lib={"{":"}","[":"]","(":")"}

        j=len(s)
        if j<2 or j%2!=0:
            return False
        c=Counter(s)
        for l in lib:
            if c.get(l)!=c.get(lib[l]):
                return False
        temp=[]
        for ind,val in enumerate(s):
            if len(temp)<1:
                temp.append(val)
                continue
            if len(temp)>len(s[ind:j]):
                return False
            if lib[temp[-1]]==val:
                temp.pop()
            else:
                temp.append(val)
        if temp:
            return False
        else:
            return True

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i=0
        while(i<len(nums)):
            if i==0:
                i+=1
                continue
            if nums[i] in nums[0:i]:
                nums.pop(i)
            else:
                i+=1
        return nums

    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        i = 0
        while (i < len(nums)):
            if nums[i] == val:
                nums.pop(i)
            else:
                i += 1
        return nums

    def strStr(self, haystack, needle):
        #28
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if len(needle)==0:
            return 0
        if needle in haystack:
            return haystack.index(needle)
        else:
            return -1

    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        temp = nums[0]
        max_ = temp
        for i in range(1, len(nums)):
            if temp > 0:
                temp += nums[i]
            else:
                temp = nums[i]
            max_ = max(max_, temp)
        return max_

    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums) < 2:
            return True

        last=len(nums)-1
        ind=last-1
        while(ind>=0):
            if nums[ind]+ind>=last:
                last=ind
            ind-=1
        return last==0

    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        s=s.split()
        if len(s)>0:
            return len(s[-1])
        else:
            return 0

    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        if not root:
            return 0
        children=[root.left,root.right]
        if not children:
            return 1
        min_depth=0
        for c in children:
            if c:
                min_depth=min(self.minDepth(c),min_depth)
        return min_depth+1

    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        from collections import Counter
        c=Counter(nums)
        for num in c:
            if c.get(num)==1:
                return num

    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        n=len(nums)
        nums=set(nums)
        res=[]
        for i in range(1,n+1):
            if i not in nums:
                res.append(i)

        return res

    def isSymmetric(self, root:TreeNode):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.matchchild(root, root)
    def matchchild(self,l:TreeNode,r:TreeNode):
        if l==None and r ==None:
            return True
        if l==None or r==None:
            return False
        if l.val!=r.val:
            return False
        return self.matchchild(l.right,r.left) and self.matchchild(l.left,r.right)

    def fairCandySwap(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        """
        suma=sum(A)
        sumb=sum(B)
        setB=set(B)

        for aa in A:
            if aa+(sumb-suma)/2 in setB:
                return [aa,aa+(sumb-suma)/2]

    def sortArrayByParity(self, A):
        res = []
        for a1 in A:
            if a1 % 2 == 0:
                res.insert(0,a1)
            else:
                res.append(a1)
        return res

    def dayOfYear(self, date):
        """
		:type date: str
		:rtype: int
		"""
        nums=[31,28,31,30,31,30,31,31,30,31,30,31]
        data=date.split("-")
        y=int(data[0])
        m=int(data[1])
        d=int(data[2])
        run=False
        if y%100==0:
            if y%400==0:
                run=True
        else:
            if y%4==0:
                run=True
        sumnum=d
        if m>1:
            sumnum=sum(nums[:m-1])+sumnum
        if run and m>2:
            sumnum+=1
        return sumnum

    def isPalindrome(self, s):
        """
		:type s: str
		:rtype: bool
		"""
        if len(s)==1 or len(s)==0:
            return True
        import re
        s = s.lower()
        s = re.sub('([^0-9a-z]+)', "", s)

        start=0
        end=len(s)-1
        while(start<end):
            if s[start]!=s[end]:
                return False
            start+=1
            end-=1
        return True

    def countCharacters(self, words, chars):
        """
		:type words: List[str]
		:type chars: str
		:rtype: int
		"""
        res=0
        for word in words:
            acter=True
            temp = list(chars)
            for char in word:
                if char in temp:
                    temp.remove(char)
                else:
                    acter=False
                    break
            if acter:
                res+=len(word)
        return res

    def relativeSortArray(self, arr1, arr2):
        """
		:type arr1: List[int]
		:type arr2: List[int]
		:rtype: List[int]
		"""
        if len(arr1)==0 or len(arr1)==1:
            return arr1
        from collections import Counter
        c=Counter(arr1)
        res=[]
        notin=list(set(arr1) - set(arr2))
        for key in arr2:
            num=c.get(key)
            i=0
            while(i<num):
                res.append(key)
                i+=1
        res2 = []
        if notin:
            for key1 in notin:
                num = c.get(key1)
                i = 0
                while (i < num):
                    res2.append(key1)
                    i += 1
            res2.sort()
        return res+res2

    def defangIPaddr(self, address):
        """
		:type address: str
		:rtype: str
		"""
        return address.replace('.','[.]')

    def distributeCandies(self, candies, num_people):
        """
		:type candies: int
		:type num_people: int
		:rtype: List[int]
		"""
        res=[0 for i in range(num_people)]
        ind=0
        fen=0
        while(candies):
            fen+=1
            if candies<=fen:
                res[ind] = res[ind] + candies
                break
            else:
                res[ind]=res[ind]+fen
            if ind==num_people-1:
                ind=0
            else:
                ind+=1
            candies-=fen
        return res

    def findOcurrences(self, text, first, second):
        """
		:type text: str
		:type first: str
		:type second: str
		:rtype: List[str]
		"""
        text=text.split(" ")
        res=[]
        while(True):
            if first in text:
                if len(text)<3:
                    break
                ind=text.index(first)
                if ind < len(text) - 2:
                    if text[ind+1]==second:
                        res.append(text[ind+2])
                        text.pop(ind)
                    else:
                        text.pop(ind)
                else:
                    break

            else:
                break
        return res

    def gcdOfStrings(self, str1: str, str2: str) -> str:
        if (len(str2) > len(str1)):
            return self.gcdOfStrings(str2, str1)
        s = str1.replace(str2, "")
        if s == "":
            return str2
        elif s == str1:
            return ""
        else:
            return self.gcdOfStrings(str2, s)


    def validPalindrome(self, s):
        """
		:type s: str
		:rtype: bool
		"""
        lens=len(s)
        if lens==0 or lens==1:
            return True

        if s==s[::-1]:
            return True
        i=0
        j=lens-1
        while(i<j):
            if s[i]==s[j]:
                i+=1
                j-=1
            else:
                if s[i+1:j]==s[j:i+1:-1] or s[i:j-1]==s[j-1:i:-1]:
                    return True
                else:
                    return False

        return False

    def isBoomerang(self, points):
        """
		:type points: List[List[int]]
		:rtype: bool
		"""
        x1=points[0][0]
        y1=points[0][1]
        x2 = points[1][0]
        y2 = points[1][1]
        x3 = points[2][0]
        y3 = points[2][1]

        if points[1]==points[2] or points[1]==points[0] or points[0]==points[2]:   #如果有相同的点 直接false
            return False
        if x1 == 0 and x2 == 0 and x3 == 0:         #如果x轴全是0  那么必定是一条直线
            return False
        if y1 == 0 and y2 == 0 and y3 == 0:          #如果y轴全是0  必定一条直线
            return False

        #点斜式  相等的斜率，表示一条直线
        if (y1-y2)*(x2-x3)==(y2-y3)*(x1-x2):
            return False
        else:
            return True

    def numMovesStones(self, a, b, c):
        """
		:type a: int
		:type b: int
		:type c: int
		:rtype: List[int]
		"""
        if abs(a-b)==1 or abs(b-c)==1:
            return [0,0]
        l_move=0
        r_move=0
        min_move=0  # 最小的如果两边不连续，永远是2
        #c-b的走法  r_move
        if abs(c-b) !=1:
            r_move=abs(c-b)-1
            min_move+=1

        if abs(b-a)!=1:
            l_move=abs(b-a)-1
            min_move+=1

        return [min_move,r_move+l_move]

    def plusOne(self, digits):
        """
		:type digits: List[int]
		:rtype: List[int]
		"""
        '''
        temp=[str(val) for val in digits]
        return list(str(int("".join(temp))+1))
        '''
        digits.reverse()
        ind=0
        lens=len(digits)
        while(ind<lens):
            val=digits[ind]+1
            if val==10:
                digits[ind]=0
                ind+=1
                if ind==lens:
                    digits.append(1)
            else:
                digits[ind]=val
                break
        digits.reverse()
        return digits

    def dietPlanPerformance(self, calories, k, lower, upper):
        """
		:type calories: List[int]
		:type k: int
		:type lower: int
		:type upper: int
		:rtype: int
		"""
        res = 0
        s = sum(calories[0: k])
        if s > upper:
            res += 1
        elif s < lower:
            res -= 1
        for i in range(1, len(calories) - k + 1):
            s = s - calories[i - 1] + calories[i + k - 1]
            if s > upper:
                res += 1
            elif s < lower:
                res -= 1
        return res

    def invalidTransactions(self, transactions):
        """
		:type transactions: List[str]
		:rtype: List[str]
		"""
        res = []
        temp=[val.split(',') for val in transactions]
        for i,trans1 in enumerate(temp):
            if int(trans1[2])>1000:
                res.append(transactions[i])
                continue
            for j,trans2 in enumerate(temp):
                if i==j:
                    continue
                if trans2[0]==trans1[0] and abs(int(trans2[1])-int(trans1[1]))<=60 and trans2[3]!=trans1[3]:
                    res.append(transactions[i])
                    break
        return res

    def dayOfTheWeek(self, day, month, year):
        """
		:type day: int
		:type month: int
		:type year: int
		:rtype: str
		"""
        dnum=[ "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday","Sunday"]
        import datetime
        return dnum[datetime.datetime(year,month,day).weekday()]

    def distanceBetweenBusStops(self, distance, start, destination):
        """
		:type distance: List[int]
		:type start: int
		:type destination: int
		:rtype: int
		"""
        if start==destination:
            return 0
        if start>destination:
            t=destination
            destination=start
            start=t
        tatol=sum(distance)
        a=sum(distance[start:destination])
        b=tatol-a
        return min(a,b)

    def strongPasswordChecker(self, s):
        """  未完成  密码强度检测
		:type s: str
		:rtype: int
		"""
        import re
        lens=len(s)
        runnum=0
        len6bool=True
        len20bool=True
        lower=True
        deight=True
        hight=True
        lianxu3=True

        if lens<6:
            len6bool=False
        elif lens>20:
            len20bool=False

        if not re.findall('[a-z]',s):
            lower=False

        if not re.findall('[0-9]',s):
            deight=False

        if not re.findall('[A-Z]',s):
            hight=False
        ind=0
        lianxu3num=0
        while(ind+1<lens):
            lianxunum=1
            j=ind+1
            while(j<lens):
                if s[ind]==s[j]:
                    lianxunum+=1
                    j+=1
                else:
                    ind=j
                    break
                if lianxunum%3==0:
                    lianxu3num+=1
                    lianxu3=False
            ind=j

        if len6bool and len20bool and lower and deight and hight and lianxu3:
            return 0
        else:
            if not len6bool:   #第一类 小于6的
                if not lianxu3:   #有三连的处理
                    while (lianxu3num > 0):  # 顺带处理 大小写数字 修改三连变其中一种 因为小于6 优先插入
                        if not lower:
                            lower = True
                        elif not hight:
                            hight = True
                        elif not deight:
                            deight = True
                        else:
                            break
                        lianxu3num -= 1
                        runnum += 1
                        lens+=1
                    #处理完连续 继续处理其他问题
                if not lower:
                    runnum += 1
                    lens += 1
                if not hight:
                    runnum += 1
                    lens += 1
                if not deight:
                    runnum += 1
                    lens += 1

                if lens<6:
                    runnum+=6-lens
                    lens=6
                elif lens>20:
                    runnum+=lens-20
                    lens=20
            elif not len20bool:  #对超过20的处理
                if not lianxu3:  # 有三连的处理
                    while (lianxu3num > 0 ):  # 顺带处理 大小写数字 修改三连变其中一种 因为大于20 优先替换 不改变长度
                        if not lower:
                            lower = True
                        elif not hight:
                            hight = True
                        elif not deight:
                            deight = True
                        else:
                            break
                        lianxu3num -= 1
                        runnum += 1
                        # 可以顺带处理的，处理后还是有连续的,
                if not lower:
                    runnum += 1
                    if lianxu3num>0:
                        lianxu3num-=1
                if not hight:
                    runnum += 1
                    if lianxu3num>0:
                        lianxu3num-=1
                if not deight:
                    runnum += 1
                    if lianxu3num>0:
                        lianxu3num-=1
                if lianxu3num>=lens-20:
                    runnum+=lianxu3num   #处理完 还有连续的 还得处理连续的
                else:
                    runnum+=lens-20
            else:    #处理6-20之间的  尽量使用替换 不改变长度
                if not lianxu3:  # 有三连的处理
                    while (lianxu3num > 0):  # 顺带处理 大小写数字 修改三连变其中一种 因为大于20 优先替换 不改变长度
                        if not lower:
                            lower = True
                        elif not hight:
                            hight = True
                        elif not deight:
                            deight = True
                        else:
                            break
                        lianxu3num -= 1
                        runnum += 1  # 可以顺带处理的，处理后还是有连续的,
                if not lower:
                    runnum += 1
                    if lianxu3num > 0:
                        lianxu3num -= 1
                if not hight:
                    runnum += 1
                    if lianxu3num > 0:
                        lianxu3num -= 1
                if not deight:
                    runnum += 1
                    if lianxu3num > 0:
                        lianxu3num -= 1
                #处理完 有可能还有多余三连
                runnum += lianxu3num

        return runnum
    def hammingWeight(self, n):
        """
		:type n: int
		:rtype: int
		"""
        return bin(n).count('1')

    def nextGreaterElement(self, nums1, nums2):
        """
		:type nums1: List[int]
		:type nums2: List[int]
		:rtype: List[int]
		"""
        res=[]
        for num in nums1:
            ind=nums2.index(num)
            if ind==len(nums2)-1:
                res.append(-1)
                continue
            j=ind+1
            findbool=False
            while(j<len(nums2)):
                num2=nums2[j]
                if num2>num:
                    res.append(num2)
                    findbool=True
                    break
                else:
                    j+=1
                    continue
            if not findbool:
                res.append(-1)
        return res

    def fizzBuzz(self, n):
        """
		:type n: int
		:rtype: List[str]
		"""
        res=[]
        for num in range(1,n+1):
            if num%3==0 and num%5==0:
                res.append('FizzBuzz')
            elif num%3==0:
                res.append('Fizz')
            elif num%5==0:
                res.append('Buzz')
            else:
                res.append(str(num))

        return res

    def thirdMax(self, nums):
        """
		:type nums: List[int]
		:rtype: int
		"""
        nums=sorted(set(nums),reverse=True)
        if len(nums)>2:
            return nums[2]
        else:
            return nums[0]

    def addBinary(self, a, b):
        """
		:type a: str
		:type b: str
		:rtype: str
		"""
        return bin(int(a,2)+int(b,2))[2:]

    def mySqrt(self, x):
        """
		:type x: int
		:rtype: int
		"""
        if x==0 or x==1:
            return x
        n=int(x/2)
        while(n*n>x):
            if n*n==x:
                return n
            n=int(0.5*(n+x/n))
        return n

    def findJudge(self, N, trust):
        """
		:type N: int
		:type trust: List[List[int]]
		:rtype: int
		"""
        if N==1 or N==0:
            return N
        p=list(range(1,N+1))  #没有相信过别人的人
        res={}       #被相信的次数
        for key,val in trust:
            if key in p:
                p.remove(key)

            if val not in res and val in p:
                res[val]=1
            elif val in res and val in p:
                res[val]+=1
        if len(p)==1 and res[p[0]]==N-1:
            return p[0]
        else:
            return -1

    def merge(self, nums1, m, nums2, n):
        """
		:type nums1: List[int]
		:type m: int
		:type nums2: List[int]
		:type n: int
		:rtype: None Do not return anything, modify nums1 in-place instead.
		"""

        nums3=nums1[0:m]
        nums1[:]=[]

        i=j=0
        while(i<m and j<n):
            if nums3[i]<nums2[j]:
                nums1.append(nums3[i])
                i+=1
            else:
                nums1.append(nums2[j])
                j+=1
        if i<m:
            nums1[i+j:]=nums3[i:]
        if j<n:
            nums1[i+j:]=nums2[j:]


    def arrangeCoins(self, n):
        """
		:type n: int
		:rtype: int
		"""
        import math
        return int((math.sqrt(1 + 8 * n) - 1) / 2)

    def generate(self, numRows):
        """
		:type numRows: int
		:rtype: List[List[int]]
		"""

        triangle = []

        for row_num in range(numRows):
            # The first and last row elements are always 1.
            row = [None for _ in range(row_num + 1)]
            row[0], row[-1] = 1, 1
            # Each triangle element is equal to the sum of the elements
            # above-and-to-the-left and above-and-to-the-right.
            for j in range(1, len(row) - 1):
                row[j] = triangle[row_num - 1][j - 1] + triangle[row_num - 1][j]

            triangle.append(row)

        return triangle

    def getRow(self, rowIndex):
        """
		:type rowIndex: int
		:rtype: List[int]
		"""
        tmp = []
        for _ in range(rowIndex + 1):
            tmp.insert(0, 1)
            for i in range(1, len(tmp) - 1):
                tmp[i] = tmp[i] + tmp[i + 1]
        return tmp


    def dominantIndex(self, nums):
        """
		:type nums: List[int]
		:rtype: int
		"""
        maxnum=max(nums)
        maxind=nums.index(maxnum)
        nums.sort(reverse=True)
        ind=1
        while(ind<len(nums)):
            if maxnum>=nums[ind]*2:
                ind+=1
            else:
                return -1
        return maxind

    def detectCapitalUse(self, word):
        """
		:type word: str
		:rtype: bool
		"""
        # 内置函数法
        #return word.islower() or word.isupper() or word.istitle()



        import re
        patt=re.compile('[A-Z]')
        strlist=re.findall(patt,word)
        lens=len(strlist)
        if lens==len(word):
            return True
        elif lens==0:
            return True
        elif lens==1:
            if word.index(strlist[0])==0:
                return True
            else:
                return False
        else:
            return False

    def canPlaceFlowers(self, flowerbed, n):
        """
		:type flowerbed: List[int]
		:type n: int
		:rtype: bool
		"""
        lens=len(flowerbed)
        nullbed=0
        bed=0
        while(bed<lens):
            if flowerbed[bed]==1:
                bed+=2
                continue
            if bed==0:
                if bed + 1 < lens:
                    if flowerbed[bed + 1] == 0:
                        nullbed += 1
                        bed += 1
                else:
                    if flowerbed[bed] == 0:
                        nullbed += 1
                        bed += 1
            elif bed==lens-1:
                if flowerbed[bed-1]==0:
                    nullbed+=1
                    bed += 1
            else:
                if flowerbed[bed+1] ==0 and flowerbed[bed-1]==0:
                    nullbed+=1
                    bed+=1
            bed+=1

        if n<=nullbed:
            return True
        else:
            return False

    def findPoisonedDuration(self, timeSeries, duration):
        """
		:type timeSeries: List[int]
		:type duration: int
		:rtype: int
		"""
        start = 0
        end = 0
        ind=0
        lens=len(timeSeries)
        sumnum=0
        while(ind<lens):
            if ind==0:
                start=timeSeries[ind]
                end=timeSeries[ind]+duration
                ind+=1
                continue
            elif timeSeries[ind]<end:   #刷新持续时间
                end=timeSeries[ind]+duration
            else:
                sumnum+=end-start
                start = timeSeries[ind]
                end = timeSeries[ind] + duration
            ind += 1
        sumnum+=end-start
        return sumnum

    def sortArray(self, nums):
        """
		:type nums: List[int]
		:rtype: List[int]
		"""
        #内置函数
        return sorted(nums)


        res=[]
        lens=len(nums)
        if lens<2:
            return nums
        while(nums):
            minnum=min(nums)
            res.append(minnum)
            del nums[nums.index(minnum)]
        return res

    def firstBadVersion(self, n):
        """
		:type n: int
		:rtype: int
		"""
        left=1
        right=n
        while(left<right):
            mid=(right+left)//2
            if isBadVersion(mid):  #说明在前面，也可能就是自己
                right=mid
            else:  #说明错的在后面
                left=mid+1
        return left



    def numSubarrayBoundedMax(self, A, L, R):
        """
		:type A: List[int]
		:type L: int
		:type R: int
		:rtype: int
		"""
        res=0
        lens=len(A)
        l=0
        r=0
        while(r<lens):
            if A[r]>=L and A[r]<=R:
                res+=r-l+1
                r+=1
            elif A[r] <= L:
                j=r-1
                while (j >= l and A[j] < L):
                    j-=1
                res+= j - l + 1
                r+=1
            else:
                l=r+1
                r+=1
        return res

    def maxProfit(self, prices):
        """
		:type prices: List[int]
		:rtype: int
		"""
        buy=0
        sell=1
        lens=len(prices)
        res=0
        if lens<2:
            return 0
        money=0
        while(buy<lens and sell<lens):
            if prices[sell]-prices[buy]>money:
                money+=prices[sell]-prices[buy]
                sell+=1
            else:
                res+=money
                money=0
                buy=sell
                sell=buy+1
            res+=money
        return res

    def hWs(self, n):
        for a in range(n, 10 ** 8):
            b = str(a)
            lens = len(b)
            if lens % 2 == 0:
                if b[0:lens // 2] == b[lens - 1:lens // 2 - 1:-1]:
                    return a
            else:
                if b[0:lens // 2] == b[lens - 1:lens // 2:-1]:
                    return a

    def isprime(self,n):
        '''
        验证是否是质数
        :param n:
        :return:
        '''
        import math
        if n==1:
            return False
        for num in range(2,int(math.sqrt(n))+1):
            if n%num==0:
                return False
        return True

    def primePalindrome(self, N):
        """
		:type N: int
		:rtype: int
		"""
        import math
        n=N
        while(True):
            zhishubool=True
            if n == 1:
                n+=1
                continue
            zhinum=2
            while(zhinum<int(math.sqrt(n)) + 1):
                if n % zhinum == 0:
                    zhishubool=False
                    break
                zhinum+=1
            if not zhishubool:
                n+=1
                continue
            b = str(n)
            if b==b[-1::-1]:
                return n
            n+=1

    def maxSatisfied(self, customers, grumpy, X):
        """
		:type customers: List[int]
		:type grumpy: List[int]
		:type X: int
		:rtype: int
		"""
        temp = 0
        for i in range(len(customers)):
            if grumpy[i] == 0:
                temp += customers[i]
                customers[i] = 0
        ma = -float('inf')
        for j in range(len(customers) - X + 1):
            ma = max(ma, sum(customers[j:j + X]))
        return temp + ma

    def validMountainArray(self, A):
        """
		:type A: List[int]
		:rtype: bool
		"""
        lens=len(A)
        if lens<3:
            return False
        i=0

        while(i<lens-1):
            if A[i+1]>A[i]:
                i+=1
            else:
                j=i
                if j==0:
                    return False
                while(j<lens-1):
                    if A[j]>A[j+1]:
                        j+=1
                    else:
                        return False
                return True
        return False

    def majorityElement(self, nums):
        """
        169
		:type nums: List[int]
		:rtype: int
		"""
        from collections import Counter
        return Counter(nums).most_common()[0][0]

    def reverseVowels(self, s):
        """
		:type s: str
		:rtype: str
		"""
        words=['a','A','E','e','i','I','o','O','u','U']
        s=[val for val in s]
        lens=len(s)
        i=0
        j=lens-1
        while(i<j):
            if s[i] in words:
                while(j>i):
                    if s[j] in words:
                        temp=s[j]
                        s[j]=s[i]
                        s[i]=temp
                        j-=1
                        break
                    else:
                        j-=1
            i+=1
        return ''.join(s)

    def reverseString(self, s):
        """
		:type s: List[str]
		:rtype: None Do not return anything, modify s in-place instead.
		"""
        #s.reverse()
        i=0
        lens=len(s)
        if lens<2:
            return s
        j=lens-1
        while(i<j):
            temp=s[j]
            s[j]=s[i]
            s[i]=temp
            i+=1
            j-=1

    def findMaxConsecutiveOnes(self, nums):
        """
		:type nums: List[int]
		:rtype: int
		"""
        ind=0
        res=0
        temp = 0
        while(ind<len(nums)):
            if nums[ind]==1:
                temp+=1
            else:
                res=max(res,temp)
                temp=0
            ind+=1
        return max(res,temp)

    def findMaxAverage(self, nums, k):
        """
		:type nums: List[int]
		:type k: int
		:rtype: float
		"""
        lens = len(nums)
        if lens < k:
            return 0
        i = 1
        maxnum = sumnum = sum(nums[0:k])

        while (i < lens-k+1):
            sumnum = sumnum + nums[i + k-1] - nums[i - 1]
            maxnum = max(sumnum, maxnum)
            i += 1
        return maxnum / float(k)

    def permute(self, nums):
        """
		:type nums: List[int]
		:rtype: List[List[int]]
		"""
        import itertools
        return itertools.permutations(nums)

    def largestTriangleArea(self, points):
        """
		:type points: List[List[int]]
		:rtype: float
		"""
        #点斜式斜率K  (y2-y1)/(x2-x1)=k
        #海伦公式  半周长p=(a+b+c)/2  面积S= math.sqrt(p*(p-a)*(p-b)*(p-c))
        #两点间距离公式  L=abs(math.sqrt(pow((x1-x2),2)+pow((y1-y2),2))))
        import itertools
        import math
        maxs=-float('inf')
        iters=itertools.permutations(points,3)
        for p1,p2,p3 in iters:
            area=0.5*abs(p1[0]*p2[1]+p2[0]*p3[1]+p3[0]*p1[1]-p1[0]*p3[1]-p2[0]*p1[1]-p3[0]*p2[1])
            maxs=max(area,maxs)
        return maxs

    def reverseWords(self, s):
        """
		:type s: str
		:rtype: str
		"""
        import re
        s=re.sub('[ ]+',' ',s)
        s=s.split(' ')
        s.reverse()
        while '' in s:
            s.remove('')
        return ' '.join(s)

    def largestPerimeter(self, A):
        """
		:type A: List[int]
		:rtype: int
		"""
        A.sort(reverse=True)
        ind=0
        while(ind<len(A)-2):
            if A[ind+2]+A[ind+1]>A[ind]:
                return A[ind]+A[ind+1]+A[ind+2]
            ind+=1
        return 0

    def isPowerOfThree(self, n):
        """
		:type n: int
		:rtype: bool
		"""
        if str(n)[-1] not in ['1','3','9','7']:
            return False
        dicts=[1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323, 4782969, 14348907, 43046721, 129140163, 387420489, 1162261467, 3486784401, 10460353203, 31381059609, 94143178827, 282429536481, 847288609443, 2541865828329, 7625597484987, 22876792454961, 68630377364883, 205891132094649, 617673396283947, 1853020188851841, 5559060566555523, 16677181699666569, 50031545098999707, 150094635296999121, 450283905890997363, 1350851717672992089, 4052555153018976267, 12157665459056928801, 36472996377170786403, 109418989131512359209, 328256967394537077627, 984770902183611232881, 2954312706550833698643, 8862938119652501095929, 26588814358957503287787, 79766443076872509863361, 239299329230617529590083, 717897987691852588770249, 2153693963075557766310747, 6461081889226673298932241, 19383245667680019896796723, 58149737003040059690390169, 174449211009120179071170507, 523347633027360537213511521, 1570042899082081611640534563, 4710128697246244834921603689, 14130386091738734504764811067, 42391158275216203514294433201, 127173474825648610542883299603, 381520424476945831628649898809, 1144561273430837494885949696427, 3433683820292512484657849089281, 10301051460877537453973547267843, 30903154382632612361920641803529, 92709463147897837085761925410587, 278128389443693511257285776231761, 834385168331080533771857328695283, 2503155504993241601315571986085849, 7509466514979724803946715958257547, 22528399544939174411840147874772641, 67585198634817523235520443624317923, 202755595904452569706561330872953769, 608266787713357709119683992618861307, 1824800363140073127359051977856583921, 5474401089420219382077155933569751763, 16423203268260658146231467800709255289, 49269609804781974438694403402127765867, 147808829414345923316083210206383297601, 443426488243037769948249630619149892803, 1330279464729113309844748891857449678409, 3990838394187339929534246675572349035227, 11972515182562019788602740026717047105681, 35917545547686059365808220080151141317043, 107752636643058178097424660240453423951129, 323257909929174534292273980721360271853387, 969773729787523602876821942164080815560161, 2909321189362570808630465826492242446680483, 8727963568087712425891397479476727340041449, 26183890704263137277674192438430182020124347, 78551672112789411833022577315290546060373041, 235655016338368235499067731945871638181119123, 706965049015104706497203195837614914543357369, 2120895147045314119491609587512844743630072107, 6362685441135942358474828762538534230890216321, 19088056323407827075424486287615602692670648963, 57264168970223481226273458862846808078011946889, 171792506910670443678820376588540424234035840667]
        if n in dicts:
            return True
        else:
            return False

    def game(self, guess, answer):
        """
		:type guess: List[int]
		:type answer: List[int]
		:rtype: int
		"""
        ind=0
        res=0
        while(ind<len(guess)):
            if guess[ind]==answer[ind]:
                res+=1
            ind+=1
        return res

    def largestTimeFromDigits(self, A):
        """
		:type A: List[int]
		:rtype: str
		"""
        import itertools
        maxH=-float('inf')
        maxmin=-float('inf')
        for v1,v2,v3,v4 in itertools.permutations(A,4):
            H=10*v1+v2
            min=v3*10+v4
            if H>23:
                continue
            elif min>59:
                continue
            else:
                if H>maxH:
                    maxH=H
                    maxmin=min
                elif H==maxH:
                    if min>=maxmin:
                        maxmin=min
        if maxH==-float('inf') and maxmin==-float('inf'):
            return ''
        else:
            resstr=str(maxH)+':'
            if maxH<10:
                resstr='0'+ resstr
            if maxmin<10:
                resstr=resstr+'0'+str(maxmin)
            else:
                resstr=resstr+str(maxmin)
            return resstr


    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        lens=len(ratings)
        ratbool=[1 for _ in ratings]
        ind=1
        while(ind<lens):
            if ratings[ind]!=ratings[ind-1]:
                if ratings[ind]>ratings[ind-1] and ratbool[ind]<=ratbool[ind-1]:
                    ratbool[ind]=ratbool[ind-1]+1
            ind+=1
        ind-=1
        while(ind>0):
            if ratings[ind]<ratings[ind-1] and ratbool[ind-1]<=ratbool[ind]:
                ratbool[ind-1]=ratbool[ind]+1
            ind-=1
        return sum(ratbool)

    def reverseWords(self, s):
        """
		:type s: str
		:rtype: str
		"""
        s=s.split(' ')
        s.reverse()
        while('' in s):
            s.remove('')
        return ' '.join(s)

    def mostCommonWord(self, paragraph, banned):
        """
		:type paragraph: str
		:type banned: List[str]
		:rtype: str
		"""
        import re
        patt=re.compile('[^a-zA-Z ]+')
        paragraph=re.sub(patt,' ',paragraph)
        paragraph=paragraph.lower()
        paragraph=re.split('[\s]+',paragraph)
        while('' in paragraph):
            paragraph.remove('')
        from collections import Counter
        c=Counter(paragraph)
        for key,val in c.most_common():
            if key in banned:
                continue
            else:
                return key

    def lemonadeChange(self, bills):
        """
		:type bills: List[int]
		:rtype: bool
		"""
        lin=[]
        sumlin=0
        for bill in bills:
            if bill==5:
                lin.append(5)
                sumlin+=5
            else:
                if sumlin-bill-5>=0:
                    sumlin=sumlin+5
                    lin.append(bill)
                else:
                    return False
        return True

    def canConstruct(self, ransomNote, magazine):
        """
		:type ransomNote: str
		:type magazine: str
		:rtype: bool
		"""
        for ransom in ransomNote:
            if ransom not in magazine:
                return False
            else:
                magazine=magazine.replace(ransom,' ',1)
        return True

    def reverseOnlyLetters(self, S):
        """
		:type S: str
		:rtype: str
		"""
        import re
        patt=re.compile('[^a-zA-Z]+')
        postion={}
        temp=list(S)
        res=[]
        for ind,value in enumerate(temp):
            if re.findall(patt,value):
                postion[ind]=value
            else:
                res.append(value)

        res.reverse()

        for i in sorted(postion.keys()):
            res.insert(i,postion[i])
        return ''.join(res)

    def titleToNumber(self, s):
        """
		:type s: str
		:rtype: int
		"""
        lib={'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26}
        s=list(s)
        ind=0
        lens=len(s)
        quan=lens-1
        res=0
        while(quan>0):
            res+=lib[s[ind]]*(26**quan)
            ind+=1
            quan-=1
        res+=lib[s[lens-1]]

        return res

    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        lib={1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M',
         14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y',
         26: 'Z'}

        res=''
        while(n>26):
            n,temp=divmod(n,26)
            if temp==0:
                res='Z'+res
                n=n-1
            else:
                res=lib[temp]+res
        res=lib[n]+res
        return res

    def balancedStringSplit(self, s):
        """
		:type s: str
		:rtype: int
		"""
        res=0
        l=0
        r=0
        for ss in s:
            if ss=='L':
                l+=1
            if ss=='R':
                r+=1
            if l==r:
                res+=1
                l=r=0
        return res

    def minCostToMoveChips(self, chips):
        """
		:type chips: List[int]
		:rtype: int
		"""
        odd = 0
        even = 0
        for i in chips:
            if i % 2 == 0:
                odd+=1
            else:
                even+=1

        if odd>even:
            return even
        else:
            return odd

    def uniqueOccurrences(self, arr):
        """
		:type arr: List[int]
		:rtype: bool
		"""
        from collections import Counter
        c=Counter(arr)
        if len(set(c))!=len(set(c.values())):
            return False
        else:
            return True

    def minimumAbsDifference(self, arr):
        """
		:type arr: List[int]
		:rtype: List[List[int]]
		"""
        arr.sort()
        minnum=float('inf')
        res=[]

        for ind in range(1,len(arr)):
            cha=abs(arr[ind]-arr[ind-1])
            if cha<minnum:
                res=[]
                minnum=cha
                res.append([arr[ind-1],arr[ind]])
            elif cha==minnum:
                res.append([arr[ind - 1], arr[ind]])
        return res

    def firstUniqChar(self, s):
        """
		:type s: str
		:rtype: int
		"""
        num={}
        for ss in set(s):
            num[ss]=s.count(ss)
        for val in s:
            if num[val]==1:
                return s.index(val)
        return -1

    def countSegments(self, s):
        """
		:type s: str
		:rtype: int
		"""
        res=s.split()
        return len(res)


    def buddyStrings(self, A, B):
        """
        859
		:type A: str
		:type B: str
		:rtype: bool
		"""
        if len(A)!=len(B):
            return False

        all=dict(zip(A,B))
        c = []

        if A==B:
            for s in A:
                if s in c:
                    return True
                else:
                    c.append(s)
            return False
        else:

            for key in all.keys():
                if all[key]!=key:
                    c.append([key,all[key]])

            if len(c)==2:
                if c[1][0]==c[0][1] and c[0][0]==c[1][1]:
                    return True
                else:
                    return False
            else:
                return False

    def compress(self, chars):
        """
		:type chars: List[str]
		:rtype: int
		"""
        i=0
        while(i<len(chars)):
            j=1
            while(i+1<len(chars)):
                if chars[i+1]==chars[i]:
                    del chars[i+1]
                    j+=1
                else:
                    break
            if j==1:
                i+=1
                continue
            if j>9:
                while(j>9):
                    t=j//10
                    chars.insert(i+1,str(t))
                    i+=1
                    j=j-t*10

            chars.insert(i + 1, str(j))
            i+=2
        print(chars)
        return len(chars)

    def repeatedSubstringPattern(self, s):
        """
		:type s: str
		:rtype: bool
		"""
        sets=len(set(s))
        lens = len(s)
        if lens ==1:
            return False
        if sets==1:
            return True

        ind=1
        while(ind<lens):
            if lens%ind!=0:
                ind+=1
                continue
            repnum=lens//ind
            if s[0:ind]*repnum==s:
                return True
            ind+=1
        return False

    def findLUSlength(self, a, b):
        """
		:type a: str
		:type b: str
		:rtype: int
		"""
        lena=len(a)
        lenb=len(b)
        if b!=a:
            return max(lenb,lena)
        else:
            return -1

    def checkRecord(self, s):
        """
        551
		:type s: str
		:rtype: bool
		"""
        return s.count('A')<2 and s.count('LLL')<1

    def reverseStr(self, s, k):
        """
        541
		:type s: str
		:type k: int
		:rtype: str
		"""
        s=list(s)
        res=[]
        ind=0
        while(ind<len(s)):
            temp=s[ind:ind+k]
            temp.reverse()
            res=res+temp
            ind+=k
            res=res+s[ind:ind+k]
            ind+=k
        return ''.join(res)

    def reverseWords(self, s):
        """
        557
		:type s: str
		:rtype: str
		"""
        s=s.split()
        for ind,ss in enumerate(s):
            ss=list(ss)
            ss.reverse()
            s[ind]=''.join(ss)
        return ' '.join(s)

    def toLowerCase(self, str):
        """
        709
		:type str: str
		:rtype: str
		"""
        return str.lower()

    def judgeCircle(self, moves):
        """
        657
		:type moves: str
		:rtype: bool
		"""
        return moves.count('R')==moves.count('L') and moves.count('U')==moves.count('D')

    def repeatedStringMatch(self, A, B):
        """
        686
		:type A: str
		:type B: str
		:rtype: int
		"""
        if A not in B and len(A)<=len(B):
            return False
        res=1
        restr=A
        while(B not in restr):
            res+=1
            restr=A*res
        return res

    def twoSum(self, nums, target):
        """
        1
		:type nums: List[int]
		:type target: int
		:rtype: List[int]
		"""
        hashmap = {}
        for ind, num in enumerate(nums):
            hashmap[num] = ind
        for i, num in enumerate(nums):
            j = hashmap.get(target - num)
            if j is not None and i != j:
                return [i, j]

    def findPairs(self, nums, k):
        """
		:type nums: List[int]
		:type k: int
		:rtype: int
		"""

        if k < 0:
            return 0
        s = set()
        r = set()
        for n in nums:
            if n + k in s:
                r.add(n + k)
            if n - k in s:
                r.add(n)
            s.add(n)

        return len(r)

    def topKFrequent(self, nums, k):
        """
		:type nums: List[int]
		:type k: int
		:rtype: List[int]
		"""
        from collections import Counter
        c=Counter(nums)
        return [key for key,value in c.most_common(k)]

    def nextPermutation(self, nums: List[int]) -> None:
        """
        31
		Do not return anything, modify nums in-place instead.
		"""

        firstIndex = -1
        n = len(nums)

        def reverse(nums, i, j):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1

        for i in range(n - 2, -1, -1):
            if nums[i] < nums[i + 1]:
                firstIndex = i
                break
        # print(firstIndex)
        if firstIndex == -1:
            reverse(nums, 0, n - 1)
            return
        secondIndex = -1
        for i in range(n - 1, firstIndex, -1):
            if nums[i] > nums[firstIndex]:
                secondIndex = i
                break
        nums[firstIndex], nums[secondIndex] = nums[secondIndex], nums[firstIndex]
        reverse(nums, firstIndex + 1, n - 1)

    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(set(nums))!=len(nums)

    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        #503
        ans = [-1 for _ in nums]
        s = []
        for i, n in enumerate(nums):
            while s and n > nums[s[-1]]:
                ans[s.pop()] = n
            s.append(i)
        if s:
            i = s.pop(0)
            j = 0
            for j in range(i + 1):
                while s and nums[j] > nums[s[-1]]:
                    ans[s.pop()] = nums[j]
                if not s:
                    return ans
        return ans

    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        #937
        def f(log):
            id_, rest = log.split(" ", 1)
            return (0, rest, id_) if rest[0].isalpha() else (1,)

        return sorted(logs, key=f)

    def licenseKeyFormatting(self, S: str, K: int) -> str:
        #482
        s = S.upper().replace('-', '')[::-1]
        res = ''
        for i in range(len(s)):
            if i % K == 0 and i != 0:
                res = '-' + res
            res = s[i] + res
        return res

    def isNStraightHand(self, hand: List[int], W: int) -> bool:
        #846
        lens=len(hand)
        if lens%W!=0:
            return False
        if W==1:
            return True
        hand.sort()
        i=0
        roundnum=1
        lastnum=hand[0]
        del hand[0]
        while(i<len(hand)):
            if hand[i]==lastnum:
                i+=1
                continue
            elif hand[i]==lastnum+1:
                lastnum=hand[i]
                del hand[i]
                roundnum+=1
            else:
                i+=1
            if roundnum==W:
                i=0
                roundnum=1
                if hand:
                    lastnum=hand[0]
                    del hand[0]
        if hand:
            return False
        else:
            return True

    def wordSubsets(self, A: List[str], B: List[str]) -> List[str]:
        #916
        lib={}
        for checkword in B:
            temp=set(checkword)
            for check in temp:
                if check in lib:
                    lib[check]=max(lib[check],checkword.count(check))
                else:
                    lib[check]=checkword.count(check)
        print(lib)
        res=[]
        for word in A:
            findbool=True
            for l in lib.keys():
                if l not in word:
                    findbool=False
                    break
                elif word.count(l)<lib[l]:
                    findbool=False
                    break
            if findbool:
                res.append(word)
        return res

    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        #1010
        res=0
        lib={k:0 for k in range(60)}
        settime=list(set(time))
        for key in settime:
            yu=key%60
            lib[yu] = time.count(key)+lib[yu]

        for i in range(1,30):
            if 60-i in lib.keys():
                res += lib[i] * lib[60 - i]

        res += (lib[0] * (lib[0] - 1) + lib[30] * (lib[30] - 1)) // 2



        return res

    def majorityElement(self, nums: List[int]) -> List[int]:
        #229
        i=0
        res=[]
        lens=len(nums)
        while(i<lens):
            if nums[i] in res:
                i+=1
                continue
            if nums.count(nums[i])>(lens/3):
                res.append(nums[i])
            i+=1
        return res

    def shortestToChar(self, S: str, C: str) -> List[int]:
        #821
        '''
        S=S.replace(C,'0')
        S=list(S)
        i=0
        lens=len(S)
        while(i<lens):
            if S[i]=='0':
                i+=1
                continue
            r=l=float('inf')
            if '0' in S[i:]:
                r=S[i:].index('0')
            leftlist=S[:i+1]
            leftlist.reverse()
            if '0' in leftlist:
                l=leftlist.index('0')
            S[i]=min(r,l)
            i+=1

        while('0' in S):
            S[S.index('0')]=0

        return S
        '''
        i = 0
        lens = len(S)
        res=[]
        while (i < lens):
            x = 0
            while True:
                if 0 <= (i - x) <= lens-1:  # 判断左查找索引是否存在
                    if S[i - x] == C:
                        res.append(x)
                        break
                if 0 <= (i + x) <= lens-1:  # 判断右查找索引是否存在
                    if S[i + x] == C:
                        res.append(x)
                        break
                x += 1
            i+=1
        return res

    def heightChecker(self, heights: List[int]) -> int:
        #1051
        res=0
        temp=sorted(heights[:])
        lens=len(heights)
        i=0
        while(i<lens):
            if temp[i]!=heights[i]:
                res+=1
            i+=1
        return res

    def maxNumberOfBalloons(self, text: str) -> int:
        #1189
        lib={'b':0,'a':0,'l':0,'o':0,'n':0}
        minnum=float('inf')
        for t in lib:
            if t=='l' or t=='o':
                lib[t]=text.count(t)//2
            else:
                lib[t] = text.count(t)
            minnum=min(minnum,lib[t])
        return minnum

    def numSmallerByFrequency(self, queries: List[str], words: List[str]) -> List[int]:
        #1170
        def f(s: str):
            s=sorted(s)
            return s.count(s[0])
        lib=[f(s) for s in words]
        res=[0 for _ in queries]
        for i,q in enumerate(queries):
            q=f(q)
            for l in lib:
                if q<l:
                    res[i]+=1
        return res

    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        #1128
        ans = 0
        d = collections.defaultdict(int)

        for i, j in dominoes:
            if i < j:
                num = 10 * i + j
            else:
                num=10 * j + i
            d[num] += 1
        for k in d.values():
            ans += int(k * (k - 1) / 2)
        return ans

    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        #491
        # dfs(状态)[3]
        #     if 状态 是 目标状态then
        #         dosomething
        #     elif
        #         for 每个新状态
        #             if 新状态合法
        #                 »dfs(新状态)
        # #·主程序：
        # #·dfs(初始状态)

        res = []
        def dfs(start, tmp):
            dic = {}
            if len(tmp) > 1:
                res.append(tmp)
            for i in range(start, len(nums)):
                if dic.get(nums[i], 0):
                    continue

                if len(tmp) == 0 or nums[i] >= tmp[-1]:
                    dic[nums[i]] = 1
                    dfs(i + 1, tmp + [nums[i]])


        dfs(0, [])
        return res

    def findDuplicate(self, nums: List[int]) -> int:
        #287
        nums.sort()
        for i in range(len(nums)):
            if nums[i]==nums[i+1]:
                return nums[i]

    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        if board[click[0]][click[1]]=='M':
            board[click[0]][click[1]]='X'
            return board
        ylen=len(board)-1
        xlen=len(board[0])-1

        nextlist=[]
        if click[0]!=ylen:
            nextlist.append([click[0] + 1, click[1]])  # 上
            if click[1]!=xlen:
                nextlist.append([click[0],click[1]+1])   #右
                nextlist.append([click[0]+1,click[1]+1])  #右上

        if click[0]!=0 and click[1]!=0:
            nextlist.append([click[0]-1,click[1]])   #下
            nextlist.append([click[0]-1,click[1]-1]) #左下
            nextlist.append([click[0],click[1]-1])   #左
        if click[0]!=ylen and  click[1]!=0:
            nextlist.append([click[0]+1,click[1]-1])   #左上
        if click[0]!=0 and click[1]!=xlen:
            nextlist.append([click[0]-1,click[1]+1])   #右下

        if nextlist:
            for y,x in nextlist:
                if board[y][x] is not 'E':
                    nextlist.remove([y,x])
        if nextlist:
            for p in nextlist:
                self.updateBoard(board,p)
        else:
            return board

    def longestWPI(self, hours: List[int]) -> int:
        lens=len(hours)
        point=[]
        for hour in hours:
            if hour>8:
                point.append(1)
            else:
                point.append(-1)


        presum = [0] * (lens + 1)   #前缀和
        for i in range(1, lens + 1):
            presum[i] = presum[i - 1] + point[i - 1]


        ans=0
        stack = []
        # 顺序生成单调栈，栈中元素从第一个元素开始严格单调递减，最后一个元素肯定是数组中的最小元素所在位置
        for i in range(lens + 1):
            if not stack or presum[stack[-1]] > presum[i]:
                stack.append(i)
        # 倒序扫描数组，求最大长度坡
        i = lens
        while i > ans:
            while stack and presum[stack[-1]] < presum[i]:
                ans = max(ans, i - stack[-1])
                stack.pop()
            i -= 1


        return ans

    def wordPattern(self, pattern: str, str: str) -> bool:
        #290

        pattern=list(pattern)
        tempstr=str.split()
        plen=len(pattern)
        tlen=len(tempstr)

        if plen!=tlen:
            return False

        res={}

        for i,patt in enumerate(pattern):
            if patt in res:
                if tempstr[i]==res[patt]:
                    continue
                else:
                    return False
            else:
                res[patt]=tempstr[i]

        res.clear()
        for i,patt in enumerate(tempstr):
            if patt in res:
                if res[patt]==pattern[i]:
                    continue
                else:
                    return False
            else:
                res[patt]=pattern[i]
        return True

    def oddCells(self, n: int, m: int, indices: List[List[int]]) -> int:
        #1252
        res=0
        nums=[0]*n
        for i in range(n):
            nums[i]=[0]*m
        for r,c in indices:
            x=0
            while(x<m):   #先做行
                nums[r][x]+=1
                x+=1

            x=0
            while(x<n):
                nums[x][c]+=1
                x+=1

        for j in range(n):
            for k in range(m):
                if nums[j][k]%2!=0:
                    res+=1
        return res

    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        #1232
        #直线斜率
        #k = (y1 - y0) / (x1 - x0) = (yi - y0) / (xi - x0)
        #转化为乘法
        #(y1 - y0) * (xi - x0) = (yi - y0) * (x1 - x0)

        y=coordinates[1][1] - coordinates[0][1]
        x=coordinates[1][0] - coordinates[0][0]
        lens=len(coordinates)
        i=0
        j=lens-1
        while(i<j):
            if i==j:
                j+=1
            if ( y*(coordinates[j][0]-coordinates[i][0]) != (coordinates[j][1] - coordinates[i][1]) * x):
                return False
            i+=1
        return True

    def allCellsDistOrder(self, R: int, C: int, r0: int, c0: int) -> List[List[int]]:
        #1030
        #曼哈顿距离|r1 - r2| + |c1 - c2|
        res=[]
        for r in range(R):
            for c in range(C):
                res.append([abs(r-r0)+abs(c-c0),[r,c]])
        res=sorted(res,key=lambda item:item[0])
        return [t for d,t in res]

    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        #1029
        costs.sort(key=lambda x:x[0]-x[1])

        n=len(costs)//2

        total=0
        for i in range(n):
            total += costs[i][0] + costs[i + n][1]
        return total


    def sumZero(self, n: int) -> List[int]:
        #1304
        '''

        lib=[i for i in range(-n,n+1)]

        if n%2!=0:
            return lib[n//2+1:n+n//2+1]
        else:
            return lib[0:n-n//2]+lib[n+n//2+1:]

        res=list(range(1,n))
        if n!=0:
            res.append(-sum(res))
        return res

        '''
        return list(range(1-n,n,2))


    def replaceElements(self, arr: List[int]) -> List[int]:
        #1299
        res=[-1]
        lens=len(arr)-1
        maxnum=arr[-1]

        for i in range(lens,0,-1):
            if maxnum<arr[i]:
                maxnum=arr[i]
                #res.insert(0,maxnum)
                res.append(maxnum)
            else:
                #res.insert(0,maxnum)
                res.append(maxnum)

        return res[::-1]

    def freqAlphabets(self, s: str) -> str:
        #1309
        res=[]
        i=len(s)-1
        while(i>=0):
            if s[i]=='#':
                res.append(string.ascii_lowercase[int(s[i-2:i])-1])
                print(s[i-2:i])
                i-=3
            else:
                res.append(string.ascii_lowercase[int(s[i])-1])
                i-=1
        return ''.join(res[::-1])

    def findNumbers(self, nums: List[int]) -> int:
        #1295
        res=0
        for num in nums:
            if len(str(num))%2==0:
                res+=1
        return res

    def findSpecialInteger(self, arr: List[int]) -> int:
        #1287
        from collections import Counter

        return collections.Counter(arr).most_common(1)[0][0]

    def subtractProductAndSum(self, n: int) -> int:
        #1281
        n=list(str(n))
        num1=1
        num2=0
        lens=len(n)-1
        while(lens>=0):
            num1=num1*int(n[lens])
            num2=num2+int(n[lens])
            lens-=1
        return num1-num2

    def fib(self, N: int) -> int:
        #509
        if N==1 or N==0:
            return N
        self.cache={0:0,1:1}
        return self.memoize(N)

    def memoize(self,N)->int:

        if N in self.cache.keys():
            return self.cache[N]

        self.cache[N]=self.memoize(N-1)+self.memoize(N-2)
        return self.memoize(N)


    def majorityElement(self, nums: List[int]) -> int:
        #17.10. 主要元素
        c=Counter(nums)
        res=c.most_common(1)[0][0]
        if nums.count(res)>len(nums)/2:
            return res
        else:
            return -1

    def missingNumber(self, nums: List[int]) -> int:
        #17.04. 消失的数字
        return list(set(range(nums.__len__()+1)) - set(nums))[0]

    def countBits(self, num: int) -> List[int]:
        #338. 比特位计数
        '''
        res=[]
        for i in range(num+1):
            res.append(list(bin(i)).count('1'))
        return res
        '''
        res=[0]*(num+1)
        for i in range(1,num+1):
            if i%2==1:
                res[i]=res[i-1]+1
            else:
                res[i]=res[i//2]
        return res

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        #90. 子集 II
        import itertools
        res=[[k] for k in set(nums)]
        res.append([])
        if nums not in res:
            res.append(nums)
        for i in range(2,len(nums)):
            res+= [list(j) for j in set(itertools.combinations(nums,i))]
        return res

    def decodeString(self, s: str) -> str:
        #394. 字符串解码
        stack=[]
        res =""
        num =0
        for c in s:
            if c == '[':
                stack.append([num, res])
                res, num = "", 0
            elif c == ']':
                cur_num, last_res = stack.pop()
                res = last_res + cur_num * res
            elif '0' <= c <= '9':
                num = num * 10 + int(c)
            else:
                res += c
        return res


    def longestConsecutive(self, nums: List[int]) -> int:
        #128. 最长连续序列
        res=1
        nums=sorted(set(nums))

        i=0
        j=i+1
        if len(nums)<2:
            return len(nums)
        while(j<len(nums)):
            n=1
            while(nums[j]==nums[i]+j-i):
                n+=1
                if j+1>=len(nums):
                    break
                else:
                    j+=1
            i=j
            j=i+1

            res=max(res,n)

        return res



    class UnionFind():
        def __init__(self):
            # parent[0]=1, 表示0的父节点是1
            # 根节点的父节点是自己
            self.parent = list(range(26))

        # 用于寻找x的根节点
        def find(self, x):
            if x == self.parent[x]:
                return x

            # 继续向上找父节点
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        # 两个节点的合并
        def union(self, x, y):
            # x的根节点直接指向y的根节点
            self.parent[self.find(x)] = self.find(y)

    def equationsPossible(self, equations: List[str]) -> bool:
        # 990. 等式方程的可满足性
        uf = Solution.UnionFind()

        for item in equations:
            if item[1] == '=':
                x = ord(item[0]) - ord('a')
                y = ord(item[3]) - ord('a')
                # 相等的进行合并操作
                uf.union(x, y)

        for item in equations:
            if item[1] == '!':
                x = ord(item[0]) - ord('a')
                y = ord(item[3]) - ord('a')
                # 判断两节点的根节点是否相同
                if uf.find(x) == uf.find(y):
                    return False

        return True




if __name__ == "__main__":
    s = Solution()

    print(s.equationsPossible(["a==b","b==c","c!=a"]))




