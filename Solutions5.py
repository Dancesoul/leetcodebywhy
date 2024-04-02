# -*- coding:utf-8 -*-
# @FileName  :Solution5.py
# @Time      :2023/7/28 0028 9:59
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
""" 
    Description:   
"""
import collections
import functools
import itertools
import math
import re
import string
from typing import *
import datetime


class Solution:
    def findMagicIndex(self, nums: List[int]) -> int:
        """
        面试题 08.03. 魔术索引
        :param nums:
        :return:
        """
        for i in range(len(nums)):
            if nums[i] == i:
                return i
        return -1

    def findString(self, words: List[str], s: str) -> int:
        """
        面试题 10.05. 稀疏数组搜索
        :param words:
        :param s:
        :return:
        """
        if s in words:
            return words.index(s)
        else:
            return -1

    def reverseBits(self, num: int) -> int:
        """
        面试题 05.03. 翻转数位
        :param num:
        :return:
        """

        left = 0
        cur = 0
        maxnum = 1
        for i in range(32):
            if num & (1 << i):
                cur += 1
            else:
                maxnum = max(maxnum, left + cur)
                left = cur + 1
                cur = 0
        maxnum = max(maxnum, left + cur)
        return maxnum

    def exchangeBits(self, num: int) -> int:
        """
        面试题 05.07. 配对交换
        :param num:
        :return:
        """
        return ((num & 0xaaaaaaaa) >> 1) | ((num & 0x55555555) << 1)

    def finalString(self, s: str) -> str:
        """
        2810. 故障键盘
        :param s:
        :return:
        """
        res = ''
        s = list(s)
        while s:
            if s[0] != 'i':
                res += s.pop(0)
            else:
                res = res[::-1]
                s.pop(0)
        return res

    def accountBalanceAfterPurchase(self, purchaseAmount: int) -> int:
        """
        2806. 取整购买后的账户余额
        :param purchaseAmount:
        :return:
        """
        return 100 - (purchaseAmount + 5 // 10) * 10

    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        """
        2798. 满足目标工作时长的员工数目
        :param hours:
        :param target:
        :return:
        """
        res = 0
        for i in hours:
            if i >= target:
                res += 1
        return res

    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        """
        2788. 按分隔符拆分字符串
        :param words:
        :param separator:
        :return:
        """
        res = []
        patt = re.escape(separator)
        for word in words:
            res += re.split(patt, word)
        while '' in res:
            res.remove('')
        return res

    def isGood(self, nums: List[int]) -> bool:
        """
        2784. 检查数组是否是好的
        :param nums:
        :return:
        """
        nums.sort()
        lens = len(nums)
        if lens == nums[-1] + 1:
            for i in range(1, lens):
                if i == nums[i - 1]:
                    continue
                else:
                    return False
            if nums[-1] == lens - 1:
                return True
            else:
                return False
        else:
            return False

    def maxSum(self, nums: List[int]) -> int:
        """
        2815. 数组中的最大数对和
        :param nums:
        :return:
        """
        libs = {}
        for num in nums:
            maxbit = max(str(num))
            if maxbit in libs:
                libs[maxbit].append(num)
            else:
                libs[maxbit] = [num]
        maxnum = -1
        for l in libs:
            if len(libs[l]) > 1:
                maxnum = max(sum(sorted(libs[l], reverse=True)[:2]), maxnum)
        return maxnum

    def sumOfSquares(self, nums: List[int]) -> int:
        """
        2778. 特殊元素平方和
        :param nums:
        :return:
        """
        lens = len(nums)
        res = 0
        for i in range(1, lens + 1):
            if lens % i == 0:
                res += nums[i - 1] * nums[i - 1]
        return res

    def checkDynasty(self, places: List[int]) -> bool:
        """
        LCR 186. 文物朝代判断
        :param places:
        :return:
        """

        places.sort()
        zeronum = places.count(0)
        if zeronum == 5:
            return True
        minnum = places[zeronum]
        if minnum + 4 <= 13:  # 因为本身有一个数，所以+4。 窗口向右构造
            target = [i for i in range(minnum, minnum + 5)]
        else:
            target = [9, 10, 11, 12, 13]

        for v in target:  # 遍历完，能清空places,表示可以连续
            if v in places:
                places.remove(v)
            elif 0 in places:
                places.remove(0)
            else:
                return False
        return True

    def fileCombination(self, target: int) -> List[List[int]]:
        """
        LCR 180. 文件组合
        :param target:
        :return:
        """
        i = 1
        j = 1
        sum = 0
        res = []
        while i <= target // 2:
            if sum < target:
                sum += j
                j += 1
            elif sum > target:
                sum -= i
                i += 1
            else:
                res.append(list(range(i, j)))
                sum -= i
                i += 1
        return res

    def isAcronym(self, words: List[str], s: str) -> bool:
        """
        2828. 判别首字母缩略词

        :param words:
        :param s:
        :return:
        """
        s = list(s)
        lens = len(words)
        if len(s) != lens:
            return False
        for i in range(lens):
            if words[i][0] == s[i]:
                continue
            else:
                return False
        return True

    def countPairs(self, nums: List[int], target: int) -> int:
        """
        2824. 统计和小于目标的下标对数目
        :param nums:
        :param target:
        :return:
        """
        # 因为只是统计数量 a+b = b+a ,所以排序改变了下标不会影响加法的结果
        nums.sort()
        i = 0
        res = 0
        j = len(nums) - 1
        while i < j:
            if nums[i] + nums[j] < target:
                res += j - i  # 因为排序过，所以这个区间内的组合都是小于目标的
                i += 1
            else:
                j -= 1
        return res

    def furthestDistanceFromOrigin(self, moves: str) -> int:
        """
        2833. 距离原点最远的点

        :param moves:
        :return:
        """
        l = moves.count("L")
        r = moves.count("R")
        fix = moves.count("_")
        if l >= r:
            return abs(-l - fix + r)
        else:
            return abs(-l + r + fix)

    def validNumber(self, s: str) -> bool:
        """
        LCR 138. 有效数字
        有效数字 若干空格 + 小数或者整数 + e或者E + 整数 + 若干空格

        小数： +/- + [0-9]*

        :param s:
        :return:
        """
        patt = re.compile("^[ ]*(?:\+|\-)?(?:(?:[0-9]+\.[0-9]?|\.[0-9]+)|[0-9]+)(?:(?:e|E)(?:\+|\-)?[0-9]+)?[ ]*$")

        lens = len(re.findall(patt, s))

        return lens > 0

    def printBin(self, num: float) -> str:
        """
        面试题 05.02. 二进制数转字符串
        :param num:
        :return:
        """
        res = "0."
        while len(res) < 33 and num != 0:
            num = num * 2
            intnum = int(num)
            res = res + str(intnum)
            num = num - intnum
        if len(res) < 33:
            return res
        else:
            return "ERROR"

    def swapNumbers(self, numbers: List[int]) -> List[int]:
        """
        面试题 16.01. 交换数字

        :param numbers:
        :return:
        """
        return numbers[::-1]

    def getTriggerTime(self, increase: List[List[int]], requirements: List[List[int]]) -> List[int]:
        """
        LCP 08. 剧情触发时间
        思路：
        先遍历触发条件  把值 按照 C{} R{} H{} 存储起来，  key是值，value是原始索引
        C R H 对key做排序
        遍历 天数，模拟三个属性的增长


        :param increase:
        :param requirements:
        :return:
        """
        c = {}
        r = {}
        h = {}
        special = []  # 存[0,0,0]的索引

        lens = len(increase)
        for ind in range(len(requirements)):
            cur_c = requirements[ind][0]
            cur_r = requirements[ind][1]
            cur_h = requirements[ind][2]
            if cur_c not in c:
                c[cur_c] = [ind]
            else:
                c[cur_c].append(ind)
            if cur_r not in r:
                r[cur_r] = [ind]
            else:
                r[cur_r].append(ind)
            if cur_h not in h:
                h[cur_h] = [ind]
            else:
                h[cur_h].append(ind)

            if cur_r == 0 and cur_c == 0 and cur_h == 0:
                special.append(ind)

        cc = sorted(c.keys())  # 排序后的值
        rr = sorted(r.keys())
        hh = sorted(h.keys())

        sumc = 0
        sumr = 0
        sumh = 0

        for i in range(lens):
            sumc += increase[i][0]
            while cc and sumc >= cc[0]:
                for j in c[cc[0]]:
                    if type(requirements[j][0]) is str:
                        continue
                    requirements[j][0] = str(i)
                cc.pop(0)

            sumr += increase[i][1]
            while rr and sumr >= rr[0]:
                for d in r[rr[0]]:
                    if type(requirements[d][1]) is str:
                        continue
                    requirements[d][1] = str(i)
                rr.pop(0)

            sumh += increase[i][2]
            while hh and sumh >= hh[0]:
                for k in h[hh[0]]:
                    if type(requirements[k][2]) is str:
                        continue
                    requirements[k][2] = str(i)
                hh.pop(0)

        res = []
        for ind in range(len(requirements)):
            maxnum = -1
            if type(requirements[ind][0]) is str and type(requirements[ind][1]) is str and type(
                    requirements[ind][2]) is str:
                maxnum = max(int(requirements[ind][0]), int(requirements[ind][1]), int(requirements[ind][2]))
                if ind not in special:
                    maxnum += 1
            res.append(maxnum)
        return res

    def cutSquares(self, square1: List[int], square2: List[int]) -> List[float]:
        """
        面试题 16.13. 平分正方形
        思路就是  平分两个矩形的一定是两个矩形中点的连线
        一：
        计算两个矩形的中点坐标
        二：
        当 中心x1 = 中心x2 那么斜率无穷大，y1取最小的y点，y2取最高的y点也就是最大的 y + 边长
        当 中心x1 != 中心x2 ，计算斜率和系数
            斜率绝对值 > 1: 上下边相交。那么y可以先求得。y1 = 最小的矩形左底点的y  那么通过斜截式得出 x1. y2= 最大的(矩形左底点y+边长)
            斜率绝对值 <= 1: 左右边相交。那么x可以先求得。x1 = 最小的左底点的x，通过斜截式得出 y1. x2=最大的(左底点x+边长)
        三：
        最后再按要求返回  {X1,Y1,X2,Y2}，要求若X1 != X2，需保证X1 < X2，否则需保证Y1 <= Y2  这里我们交换两点的坐标就可以
        :param square1:
        :param square2:
        :return:
        """
        # 第一个中点
        centerx1 = square1[0] + square1[2] / 2
        centery1 = square1[1] + square1[2] / 2
        # 第二个中点
        centerx2 = square2[0] + square2[2] / 2
        centery2 = square2[1] + square2[2] / 2

        # 斜率无穷大
        if centerx1 == centerx2:
            x1 = centerx1
            y1 = min(square1[1], square2[1])
            x2 = centerx1
            y2 = max(square1[1] + square1[2], square2[1] + square2[2])

        else:
            # 两点斜率计算
            k = (centery2 - centery1) / (centerx2 - centerx1)
            # 斜截式计算 b
            b = centery1 - k * centerx1
            if abs(k) > 1:  # 上下边相交
                y1 = min(square1[1], square2[1])
                x1 = (y1 - b) / k
                y2 = max(square1[1] + square1[2], square2[1] + square2[2])
                x2 = (y2 - b) / k
            else:  # 左右边相交
                x1 = min(square1[0], square2[0])
                y1 = k * x1 + b
                x2 = max(square1[0] + square1[2], square2[0] + square2[2])
                y2 = k * x2 + b

        if x1 > x2:
            temp = x1
            x1 = x2
            x2 = temp
            temp = y1
            y1 = y2
            y2 = temp
        return [x1, y1, x2, y2]

    def countSeniors(self, details: List[str]) -> int:
        """
        2678. 老人的数目
        :param details:
        :return:
        """
        res = 0
        for detail in details:
            age = int(detail[11:13])
            if age > 60:
                res += 1
        return res

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        139. 单词拆分
        :param s:
        :param wordDict:
        :return:
        """
        lens = len(s)
        dp = [False] * (lens + 1)
        dp[0] = True
        for i in range(1, lens + 1):
            for j in range(0, i):
                if dp[j] and s[j:j + i - j] in wordDict:
                    dp[i] = True
                    break
        return dp[lens]

    def maxProduct(self, nums: List[int]) -> int:
        """
        152. 乘积最大子数组
        :param nums:
        :return:
        """
        res = nums[0]
        pre_max = nums[0]
        pre_min = nums[0]
        for num in nums[1:]:
            cur_max = max(pre_max * num, pre_min * num, num)
            cur_min = min(pre_max * num, pre_min * num, num)
            res = max(res, cur_max)
            pre_max = cur_max
            pre_min = cur_min
        return res

    def largestNumber(self, nums: List[int]) -> str:
        """
        179. 最大数  看的宫水三叶的题解，自己做不出来
        使用贪心获得所有的组合，然后做比较
        :param nums:
        :return:
        """
        strs = map(str, nums)

        def cmp(a, b):
            if a + b == b + a:
                return 0
            elif a + b > b + a:
                return 1
            else:
                return -1

        strs = sorted(strs, key=functools.cmp_to_key(cmp), reverse=True)
        return ''.join(strs) if strs[0] != '0' else '0'

    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        """
        1155. 掷骰子等于目标和的方法数
        :param n:
        :param k:
        :param target:
        :return:
        """
        if not (n <= target <= n * k):
            return 0  # 无法组成 target
        MOD = 10 ** 9 + 7
        f = [1] + [0] * (target - n)
        for i in range(1, n + 1):
            max_j = min(i * (k - 1), target - n)  # i 个骰子至多掷出 i*(k-1)
            for j in range(1, max_j + 1):
                f[j] += f[j - 1]  # 原地计算 f 的前缀和
            for j in range(max_j, k - 1, -1):
                f[j] = (f[j] - f[j - k]) % MOD  # f[j] 是两个前缀和的差
        return f[-1]

    def rotate(self, nums: List[int], k: int) -> None:
        """
        189. 轮转数组
        :param nums:
        :param k:
        :return:
        """
        k = k % len(nums)
        if k != 0:
            nums[:] = nums[-k:] + nums[:-k]

    def isPowerOfTwo(self, n: int) -> bool:
        """
        231.    2 的幂
        因为2的幂的结果，在二进制中只有一个1。 所以与（n-1 也是就二进制1的位置变为0，后面全变为1的数。）做与运算等于0 说明n是2的幂。
        特殊判断n大于0
        :param n:
        :return:
        """
        return (n > 0) and ((n & (n - 1)) == 0)

    def punishmentNumber(self, n: int) -> int:
        """
        2698. 求一个整数的惩罚数
        :param n:
        :return:
        """
        for i in range(1, n + 1):
            temp = i * i

    def countDigits(self, num: int) -> int:
        """
        2520. 统计能整除数字的位数
        :param num:
        :return:
        """
        res = 0
        temp = num
        while temp:
            if num % (temp%10) == 0:
                res += 1
            temp //= 10
        return res



if __name__ == '__main__':
    s = Solution()
    print(s.countDigits(1248))
