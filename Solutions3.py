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

import collections
import itertools
import math
import string
from typing import *
import datetime


class Solution:
    def tribonacci(self, n: int) -> int:
        memo = {}

        def helper(n):
            if n in memo:
                return memo[n]

            if n == 0 or n == 1:
                memo[n] = n

            elif n == 2:
                memo[n] = 1

            else:
                memo[n] = helper(n - 1) + helper((n - 2)) + helper(n - 3)
            return memo[n]

        return helper(n)

    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        # 766. 托普利茨矩阵
        m = len(matrix)
        n = len(matrix[0])
        if m == 1:
            return True

        exel = matrix[0]
        i = 1
        while (i < m):  # 行
            j = 1
            while (j < n):  # 列
                if matrix[i][j] != exel[j - 1]:
                    return False
                j += 1
            exel = matrix[i]
            i += 1
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
            res[i] = s - nums[i] * lens + (i * nums[i] - n) * 2
            n += nums[i]
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
        for i in range(1, lens + 1):
            for v in itertools.combinations(nums, i):
                if list(v) not in res:
                    res.append(list(v))
        return res

    def hammingWeight(self, n: int) -> int:
        # 191. 位1的个数
        i, j = 0, 31
        res = 0

        while (i < j):
            if n & (1 << i):
                res += 1
            if n & (1 << j):
                res += 1
            i += 1
            j -= 1
        return res

    def findIntegers(self, num: int) -> int:
        # 600. 不含连续1的非负整数
        res = 0
        for i in range(num):
            if bin(i).find("11") != -1:
                res += 1
        return res

    def hammingDistance(self, x: int, y: int) -> int:
        # 461. 汉明距离
        return bin(x ^ y).count("1")

    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        # 852. 山脉数组的峰顶索引
        lens = len(arr)
        i = 0
        j = lens - 1
        while (i < j):
            if arr[i] < arr[i + 1]:
                i += 1
            else:
                return i
            if arr[j] < arr[j - 1]:
                j -= 1
            else:
                return j
        return i

    def minPairSum(self, nums: List[int]) -> int:
        # 1877. 数组中最大数对和的最小值
        nums.sort()
        n = len(nums)
        num = 0
        for i in range(n // 2):
            num = max(num, nums[i] + nums[n - 1 - i])
        return num

    def trap(self, height: List[int]) -> int:
        # 42. 接雨水
        lens = len(height)
        left, right = [0] * (lens + 1), [0] * (lens + 1)
        ans = 0
        for i in range(1, len(height) + 1):  # 求出从1开始每个位置的左侧最大值
            left[i] = max(left[i - 1], height[i - 1])

        for i in range(len(height) - 1, 0, -1):  # 求出从最后一个（lens-1）开始 右侧最大值
            right[i] = max(right[i + 1], height[i])

        for i in range(len(height)):  # 遍历整个数组， 求当前位置能集的水 = 最小的左右最大值 - 当前高度
            ans += max(0, min(left[i + 1], right[i]) - height[i])
        return ans

    def maxLengthBetweenEqualCharacters(self, s: str) -> int:
        # 1624. 两个相同字符之间的最长子字符串
        lens = len(s)
        if lens == len(set(s)):
            return -1

        i = 0
        maxnum = 0
        while i < lens:
            j = i + 1
            while j < lens:
                if s[i] == s[j]:
                    maxnum = max(maxnum, j - i - 1)
                j += 1
            i += 1
        return maxnum

    def kthFactor(self, n: int, k: int) -> int:
        # 1492. n 的第 k 个因子
        factor = [1, n]
        i = 2
        j = n
        while i <= j:
            num1 = n % i  # 本次检查是不是因数
            j = n // i  # j应该到的位置
            if num1 == 0:
                factor.append(i)
                factor.append(j)
            i += 1
        factor = list(set(factor))
        factor.sort()
        print(factor)
        if k > len(factor):
            return -1
        else:
            return factor[k - 1]

    def closestDivisors(self, num: int) -> List[int]:
        # 1362. 最接近的因数
        import math
        num1 = num + 1
        num2 = num + 2
        mid = math.sqrt(num1)  # 开根 如果直接得到一个整数 那么直接返回, 如果有小数，那么从mid处开始使用快慢指针，向两边扫描
        mid2 = math.sqrt(num2)  # 第二个数开根
        print(mid, mid2)

    def getStrongest(self, arr: List[int], k: int) -> List[int]:
        # 1471 1471. 数组中的 k 个最强值
        arr = sorted(arr)
        lens = len(arr)
        mid = arr[int((lens - 1) // 2)]

        for i in range(lens):
            arr[i] = [abs(arr[i] - mid), arr[i]]

        arr = sorted(arr, reverse=True, key=lambda x: (x[0], x[1]))
        res = []
        for j in range(lens):
            if j > k - 1:
                break
            res.append(arr[j][1])
        return res

    def strToInt(self, str: str) -> int:
        # 剑指 Offer 67. 把字符串转换成整数
        import string
        max_int = 2 ** 31 - 1
        min_int = -2 ** 31
        res = str.split()
        if res:
            if res[0][0] in string.digits or res[0][0] in ["-", "+"]:
                symbol = ""
                temp = res[0]
                ans = ""
                if temp[0] in ["-", "+"]:
                    symbol = temp[0]
                    temp = temp[1:]

                for i in temp:
                    if i not in string.digits:
                        break
                    else:
                        ans += i
                if ans != "":
                    ans = int(symbol + ans)
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
        lib = {"l": {}, "r": {}}

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
        maxindex = nums.index(max(nums[minindex + 1:]))
        i = minindex
        j = maxindex
        # 结果肯定在最小值和最大值之间， 因为总是有结果  所有最小一定在最大的的左边,即 最小值左边都应该在左边，最大值右边的，都应该在右边
        maxnum = max(nums[:i + 1])
        minnum = min(nums[i + 1:])
        r = sorted(nums[i + 1:])
        while i < j:
            if maxnum <= minnum:
                return i + 1
            i += 1
            r.remove(nums[i])
            minnum = r[0]
            maxnum = max(maxnum, nums[i])

        return i

    def interpret(self, command: str) -> str:
        # 1678. 设计 Goal 解析器
        return command.replace("()", "o").replace("(al)", "al")

    def nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
        # 1201. 丑数 III  有问题
        res = []
        l = sorted([a, b, c])
        a = l[0]
        b = l[1]
        c = l[2]
        ab = (b - a) // a  # a-b之间a的数量
        bc = (c - b) // b  # b-c之间b的数量
        x = 1  # a的数量
        y = 1  # b的数量
        z = 1  # c的数量
        if n <= 3:
            return l[n - 1]

        while x + y + z <= n:
            abi = 0
            while abi <= ab:
                res.append(a * x)
                abi += 1
                x += 1
            bci = 0
            while bci <= bc:
                res.append(b * y)
                bci += 1
                y += 1
            res.append(c * z)
            z += 1
        return res[n - 1]

    def findLongestSubarray(self, array: List[str]) -> List[str]:
        # 面试题 17.05.  字母与数字
        i = 0
        j = 0
        ind = 0
        lens = len(array)
        res = 0
        while ind < lens:
            if array[ind] in string.ascii_letters:
                i += 1
            else:
                j += 1
            if i == j:
                res = ind
            ind += 1
        if res:
            return array[:res + 1]
        else:
            return []

    def findComplement(self, num: int) -> int:
        # 476. 数字的补数

        b = str(bin(num)[2:])
        res = ""
        for bb in b:
            res = res + str(1 - int(bb))
        return int(res, base=2)

    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        # 剑指 Offer 04. 二维数组中的查找
        r = len(matrix) - 1
        j = 0
        while r >= 0 and j < len(matrix[0]):
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
        while j >= 0:
            if quan == 1:
                temp = digits[j] + 1
                quan = 0
            else:
                temp = digits[j] + 1

            if temp == 10:
                quan = 1
                digits[j] = 0
                j -= 1
            else:
                digits[j] = temp
                return digits
        if quan == 1:
            digits.insert(0, 1)
        return digits

    def minDominoRotations(self, tops: List[int], bottoms: List[int]) -> int:
        # 1007. 行相等的最少多米诺旋转
        from collections import Counter
        all = tops + bottoms
        lens = len(tops)
        c = Counter(all)
        value = c.most_common(1)[0][0]  # 最多的元素
        v_lens = c.most_common(1)[0][1]  # 最多元素的个数 小于一半则无法成功
        if v_lens < lens:
            return -1
        i = 0
        top_v = []
        bottoms_v = []
        while i < lens:
            if tops[i] == value:
                top_v.append(i)
            if bottoms[i] == value:
                bottoms_v.append(i)
            if bottoms[i] != value and tops[i] != value:
                return -1
            i += 1
        t_lens = len(top_v)
        b_lens = len(bottoms_v)
        return lens - max(t_lens, b_lens)

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
            if x % 2 == 0:
                return self.multiply(base, x // 2) + self.multiply(base, x // 2)
            else:
                return self.multiply(base, x // 2) + self.multiply(base, x // 2) + self.multiply(base, 1)

    def majorityElement(self, nums: List[int]) -> List[int]:
        # 229. 求众数 II
        lens = len(nums)
        c = Counter(nums)
        res = []
        for key, ind in c.most_common():
            if ind > lens / 3:
                res.append(key)
            else:
                break
        return res

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        #  496. 下一个更大元素 I
        nums1 = {i: -1 for i in nums1}
        i = 0
        lens = len(nums2)
        while i < lens:
            temp = nums2[i]
            if temp in nums1:
                j = i + 1
                while j < lens:
                    if nums2[j] > temp:
                        nums1[temp] = nums2[j]
                        break
                    j += 1
            i += 1
        return list(nums1.values())

    def missingNumber(self, nums: List[int]) -> int:
        # 268. 丢失的数字
        lens = len(nums)
        return (set(range(lens + 1)) - set(nums)).pop()

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
                maxnum = max(maxnum, now)
                last = s[i]
                now = 1
            else:
                now += 1
            i += 1
        return max(maxnum, now)

    def subsets(self, nums: List[int]) -> List[List[int]]:
        # 78. 子集
        import itertools
        return list(itertools.product("".join(nums), repeat=2))

    def minimumDifference(self, nums: List[int], k: int) -> int:
        # 1984. 学生分数的最小差值
        if k == 1:
            return 0
        nums.sort()
        i = 0
        lens = len(nums)
        minnum = float("inf")
        while i + k - 1 < lens:
            minnum = min(minnum, nums[k - 1 + i] - nums[i])
            i += 1

        return minnum

    def removeAnagrams(self, words: List[str]) -> List[str]:
        """
        2273. 移除字母异位词后的结果数组
        :param words:
        :return:
        """
        i = 1
        j = 0
        while i < len(words):
            left = sorted(words[j])
            right = sorted(words[i])
            if left == right:
                words.pop(i)
            else:
                j += 1
                i += 1
        return words

    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        """
        219. 存在重复元素 II
        给你一个整数数组 nums 和一个整数 k ，判断数组中是否存在两个 不同的索引i和j
        满足 nums[i] == nums[j] 且 abs(i - j) <= k 。如果存在，返回 true ；否则，返回 false 。

        使用哈希的方式， key存数值，value 存下标。 这样保证值的下标一直都是最大的 最靠近的。
        :param nums:
        :param k:
        :return:
        """
        numsdict = {}
        for key, num in enumerate(nums):
            if num in numsdict and key - numsdict[num] <= k:
                return True
            numsdict[num] = key
        return False

    def largestWordCount(self, messages: List[str], senders: List[str]) -> str:
        """
        2284. 最多单词数的发件人
        给你一个聊天记录，共包含 n条信息。给你两个字符串数组messages 和senders，其中messages[i]是senders[i]发出的一条信息。
        一条信息是若干用单个空格连接的 单词，信息开头和结尾不会有多余空格。发件人的 单词计数是这个发件人总共发出的 单词数。注意，一个发件人可能会发出多于一条信息。
        请你返回发出单词数 最多的发件人名字。如果有多个发件人发出最多单词数，请你返回 字典序最大的名字。

        :param messages:
        :param senders:
        :return:
        """
        res = {}
        lens = len(messages)
        # 先做统计
        for i in range(lens):
            if senders[i] not in res:
                res[senders[i]] = 0
            res[senders[i]] += len(messages[i].split())
        # 找出发消息最多的人 可能有多个
        maxnum = max(res.values())
        maxlist = []
        for item in res:
            if res[item] == maxnum:
                maxlist.append(item)
        return sorted(maxlist, reverse=True)[0]

    def halvesAreAlike(self, s: str) -> bool:
        """
        1704. 判断字符串的两半是否相似
        给你一个偶数长度的字符串 s 。将其拆分成长度相同的两半，前一半为 a ，后一半为 b 。
        两个字符串 相似 的前提是它们都含有相同数目的元音（'a'，'e'，'i'，'o'，'u'，'A'，'E'，'I'，'O'，'U'）。注意，s 可能同时含有大写和小写字母。
        如果 a 和 b 相似，返回 true ；否则，返回 false 。

        :param s:
        :return:
        """
        vowel = ('a', 'e', 'i', 'o', 'u')
        mid = int(len(s) / 2)
        s = s.lower()
        s1 = s[:mid]
        s2 = s[mid:]
        vowel1 = 0
        vowel2 = 0
        for i in range(mid):
            if s1[i] in vowel:
                vowel1 += 1
            if s2[i] in vowel:
                vowel2 += 1
        return vowel1 == vowel2

    def diagonalSum(self, mat: List[List[int]]) -> int:
        """
        1572. 矩阵对角线元素的和
        给你一个正方形矩阵 mat，请你返回矩阵对角线元素的和。
        请你返回在矩阵主对角线上的元素和副对角线上且不在主对角线上元素的和。
        :param mat:
        :return:
        """
        # 对角线的坐标 x = y  副对角线坐标 x+y = len
        lens = len(mat)
        diagonal = 0
        counterdiagonal = 0
        for x in range(lens):
            diagonal += mat[x][x]
            y = lens - 1 - x
            if y == x:
                continue
            counterdiagonal += mat[x][y]
        return diagonal + counterdiagonal

    def splitArraySameAverage(self, nums: List[int]) -> bool:
        """
        805. 数组的均值分割
        给定你一个整数数组nums
        我们要将nums数组中的每个元素移动到A数组 或者B数组中，使得A数组和B数组不为空，并且average(A) == average(B)。
        如果可以完成则返回true， 否则返回 false。
        注意：对于数组arr, average(arr)是arr的所有元素的和除以arr长度。

        :param nums:
        :return:
        """
        lens = len(nums)
        if lens == 1:
            return False

        s = sum(nums)
        for i in range(lens):
            nums[i] = nums[i] * lens - s  # 将列表的均值调整为整数

        m = lens // 2
        left = set()
        # 遍历列表，进行元素的选择，借助位运算  左半部分总查找 例如： mid 为4 一共有 2^4-1 为15 所以用左移位运算
        # 知识点： 有m 需要 2^m-1 那么使用左位移运算 1<<m
        for i in range(1, 1 << m):
            #  该步用位运算模拟当前位置元素是否参选 比如6 = 0110 即1号与2号为元素选中
            #  则此步骤的total = nums[1] + nums[2]
            tot = sum(x for j, x in enumerate(nums[:m]) if i >> j & 1)

            # 如果找到总和为0的，那么返回
            if tot == 0:
                return True
            # 没找到则添加到集合中，继续后续查找
            left.add(tot)

        # 求右边的和
        rsum = sum(nums[m:])
        # 同样的方法 借助位运算进行元素的选择
        for i in range(1, 1 << (lens - m)):
            tot = sum(x for j, x in enumerate(nums[m:]) if i >> j & 1)
            if tot == 0 or rsum != tot and -tot in left:
                return True
        return False

    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        """
        1710. 卡车上的最大单元数
        请你将一些箱子装在 一辆卡车 上。给你一个二维数组 boxTypes ，其中 boxTypes[i] = [numberOfBoxesi, numberOfUnitsPerBoxi] ：
        numberOfBoxesi 是类型 i 的箱子的数量。
        numberOfUnitsPerBoxi 是类型 i每个箱子可以装载的单元数量。
        整数 truckSize 表示卡车上可以装载 箱子 的 最大数量 。只要箱子数量不超过 truckSize ，你就可以选择任意箱子装到卡车上。

        返回卡车可以装载单元 的 最大 总数。

        :param boxTypes:
        :param truckSize:
        :return:
        """
        boxTypes.sort(key=lambda x: x[1], reverse=True)
        i = 0
        res = 0
        while truckSize > 0 and i < len(boxTypes):
            numberOfBoxes = boxTypes[i][0]
            numberOfUnitsPerBox = boxTypes[i][1]
            if numberOfBoxes >= truckSize:
                res += truckSize * numberOfUnitsPerBox
                return res
            else:
                res += numberOfBoxes * numberOfUnitsPerBox
                truckSize -= numberOfBoxes
                i += 1
        return res

    def isIdealPermutation(self, nums: List[int]) -> bool:
        """
        775. 全局倒置与局部倒置

        :param nums:
        :return:
        """
        return all(abs(x - i) <= 1 for i, x in enumerate(nums))

    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        """
        792. 匹配子序列的单词数
        :param s:
        :param words:
        :return:
        """
        s = "".join(sorted(s))
        print(s)
        res = 0
        for word in words:
            if "".join(sorted(word)) in s:
                res += 1
        return res

    def divideString(self, s: str, k: int, fill: str) -> List[str]:
        """
        2138. 将字符串拆分为若干长度为 k 的组
        :param s:
        :param k:
        :param fill:
        :return:
        """
        lens = len(s)
        start = 0
        res = []
        stop = k
        while stop < lens:
            temp = s[start:stop]
            res.append(temp)
            start = stop
            stop += k
        res.append(s[start:lens] + (stop - lens) * fill)
        return res

    def numberOfPairs(self, nums: List[int]) -> List[int]:
        """
        2341. 数组能形成多少数对
        :param nums:
        :return:
        """
        nums.sort()
        i = 1
        res = [0]
        while i < len(nums):
            if nums[i] == nums[i - 1]:
                res[0] += 1
                nums.pop(i)
                nums.pop(i - 1)
            else:
                i += 1
        res.append(len(nums))
        return res

    def finalValueAfterOperations(self, operations: List[str]) -> int:
        """
        2011. 执行操作后的变量值
        :param operations:
        :return:
        """
        res = 0
        for i in operations:
            if '+' in i:
                res += 1
            else:
                res -= 1
        return res

    def mostVisited(self, n: int, rounds: List[int]) -> List[int]:
        """
        1560. 圆形赛道上经过次数最多的扇区
        :param n:
        :param rounds:
        :return:
        """
        start, end = rounds[0], rounds[-1]
        if start <= end:
            return list(range(start, end + 1))
        else:
            leftPart = range(1, end + 1)
            rightPart = range(start, n + 1)
            return list(itertools.chain(leftPart, rightPart))

    def minNumBooths(self, demand: List[str]) -> int:
        """
        LCP 66. 最小展台数量
        :param demand:
        :return:
        """
        res = ""
        for demands in demand:
            temp = res
            for word in demands:
                if word not in res:  # 判断有没有这个类型 没有就加上
                    res += word
                elif word not in temp:  # 判断这个类型的台够不够
                    res += word
                else:
                    temp = temp.replace(word, "", 1)  # 当天使用过的台，就去掉
        return len(res)

    def temperatureTrend(self, temperatureA: List[int], temperatureB: List[int]) -> int:
        """
        LCP 61. 气温变化趋势
        :param temperatureA:
        :param temperatureB:
        :return:
        """
        lens = len(temperatureA)
        maxnum = 0  # 存连续变化相同的最大天数
        i = 0

        while i < lens - 1:
            curnum = 0  # 本轮连续天数
            j = i
            while j < lens - 1:
                if temperatureA[j + 1] > temperatureA[j] and temperatureB[j + 1] > temperatureB[j]:
                    curnum += 1
                    j += 1
                    continue

                elif temperatureA[j + 1] == temperatureA[j] and temperatureB[j + 1] == temperatureB[j]:
                    curnum += 1
                    j += 1
                    continue

                elif temperatureA[j + 1] < temperatureA[j] and temperatureB[j + 1] < temperatureB[j]:
                    curnum += 1
                    j += 1
                    continue

                else:
                    break

            i = j + 1
            maxnum = max(maxnum, curnum)
        return maxnum

    def getMinimumTime(self, time: List[int], fruits: List[List[int]], limit: int) -> int:
        """
        LCP 55. 采集果实
        :param time:
        :param fruits:
        :param limit:
        :return:
        """
        res = 0
        for fruit in fruits:
            unittime = time[fruit[0]]  # 采集当前这个类型1次需要花的时间
            unit = math.ceil(fruit[1] / limit)  # 采集当前水果需要花费的次数
            res += unittime * unit
        return res

    def perfectMenu(self, materials: List[int], cookbooks: List[List[int]], attribute: List[List[int]],
                    limit: int) -> int:
        """
        LCP 51. 烹饪料理
        :param materials:
        :param cookbooks:
        :param attribute:
        :param limit:
        :return:
        """
        global res  # 存放可以满足饱腹感的最大美味值
        res = -1

        def helper(materials, cookbooks, attribute, surplus, need, delicious, index):
            """
            用来递归遍历可以做的料理  把所有结果都遍历出来
            :param surplus:  剩余的食材数量
            :param delicious:  美味值
            :param need:   饱腹感
            :param index:  第 index 道料理
            :return:
            """
            global res
            if need <= 0:  # 已经吃饱 那么就结算当前美味值
                res = max(res, delicious)
            for i in range(index, len(cookbooks)):  # 遍历料理
                enough = True
                temp = [0] * len(materials)
                for j in range(len(materials)):  # 遍历食材，看能不能做这个料理
                    temp[j] = surplus[j] - cookbooks[i][j]
                    if temp[j] >= 0:
                        continue
                    else:
                        enough = False
                        break
                if enough:  # 可以做
                    helper(materials, cookbooks, attribute, temp, need - attribute[i][1], delicious + attribute[i][0],
                           i + 1)
                else:  # 不可以做
                    helper(materials, cookbooks, attribute, surplus, need, delicious, i + 1)

        helper(materials, cookbooks, attribute, materials, limit, 0, 0)
        return res

    def giveGem(self, gem: List[int], operations: List[List[int]]) -> int:
        """
        LCP 50. 宝石补给
        :param gem:
        :param operations:
        :return:
        """
        for i, j in operations:
            give = math.floor(gem[i] / 2)
            gem[i] -= give
            gem[j] += give

        return max(gem) - min(gem)

    def countEven(self, num: int) -> int:
        """
        2180. 统计各位数字之和为偶数的整数个数
        :param num:
        :return:
        """
        res = 0
        for n in range(1, num + 1):
            num = 0
            while n > 0:
                temp = divmod(n, 10)
                n = temp[0]
                num += temp[1]
            if num % 2 == 0:
                res += 1
        return res

    def maxmiumScore(self, cards: List[int], cnt: int) -> int:
        """
        LCP 40. 心算挑战
        :param cards:
        :param cnt:
        :return:
        """
        cards.sort(reverse=True)
        odd = []  # 奇数
        even = []  # 偶数
        for card in cards:
            if card & 1:
                odd.append(card)
            else:
                even.append(card)
        res = 0
        if cnt & 1:
            if len(even):  # 判断奇偶，奇数的话，直接取偶数最大来成对
                res += even.pop(0)
            else:
                return 0
        cnt >>= 1
        alls = []
        for i in range(len(odd) >> 1):
            alls.append(odd[2 * i] + odd[2 * i + 1])

        for i in range(len(even) >> 1):
            alls.append(even[2 * i] + even[2 * i + 1])

        card_ = sorted(alls, reverse=True)

        return res + sum(card_[:cnt]) if cnt <= len(card_) else 0

    def minOperations(self, nums: List[int], x: int) -> int:
        """
        1658. 将 x 减到 0 的最小操作数
        :param nums:
        :param x:
        :return:
        """
        lens = len(nums)
        total = sum(nums)

        if total < x:
            return -1

        right = 0
        lsum = 0
        rsum = total
        res = lens + 1
        for left in range(-1, lens - 1):
            if left != -1:
                lsum += nums[left]
            while right < lens and lsum + rsum > x:
                rsum -= nums[right]
                right += 1
            if lsum + rsum == x:
                res = min(res, (left + 1) + (lens - right))

        return -1 if res > lens else res

    def minimumSwitchingTimes(self, source: List[List[int]], target: List[List[int]]) -> int:
        """
        LCP 39. 无人机方阵
        :param source:
        :param target:
        :return:
        """
        lens = len(source)
        lenss = len(source[0])
        libs = {str(i): 0 for i in range(10 ** 4)}

        for i in range(lens):
            for j in range(lenss):
                s = source[i][j]
                t = target[i][j]
                libs[str(t)] += 1
                libs[str(s)] -= 1

        res = 0
        for tt in libs:
            if libs[tt] < 0:
                res += libs[tt]
        return res

    def reinitializePermutation(self, n: int) -> int:
        """
        1806. 还原排列的最少操作步数
        :param n:
        :return:
        """
        perm = list(range(n))
        num = 0
        while 1:
            num += 1
            perm = [perm[int(i / 2)] if i % 2 == 0 else perm[int(n / 2 + (i - 1) / 2)] for i in range(n)]
            if perm == list(range(n)):
                return num

    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        """
        剑指 Offer II 075. 数组相对排序
        :param arr1:
        :param arr2:
        :return:
        """
        c = collections.Counter(arr1)
        res = []
        notin = []
        for a in arr2:
            n = c[a]
            while n:
                res.append(a)
                n -= 1
            c.pop(a)
        for k, v in c.most_common():
            while v:
                notin.append(k)
                v -= 1
        return res + sorted(notin)

    def mySqrt(self, x: int) -> int:
        """
        剑指 Offer II 072. 求平方根  这里用牛顿迭代法
        :param x:
        :return:
        """

        def helper(num, rx, e):
            """
            递归求解
            :param num: 需要求平方根的目标
            :param rx:  迭代区间
            :param e:  精度
            :return:
            """
            num *= 1  # 目标初始化
            if abs(num - rx * rx) < e:
                # return int(rx)
                return rx
            else:
                return helper(num, (num / rx + rx) / 2, e)

        return helper(x, 1, 0.1)

    def searchInsert(self, nums: List[int], target: int) -> int:
        """
        剑指 Offer II 068. 查找插入位置
        :param nums:
        :param target:
        :return:
        """
        lens = len(nums)
        i = 0
        j = lens - 1
        if i == j:
            return 0 if target <= nums[0] else 1

        while i < lens:
            left = nums[i]
            right = nums[j]
            if left == target:
                return i
            if right == target:
                return j
            if left > target:
                return i if i > 0 else 0
            if right < target:
                return j + 1
            i += 1
            j -= 1

    def countAsterisks(self, s: str) -> int:
        """
        2315. 统计星号
        :param s:
        :return:
        """
        index = 0  # 这里判断星星是不是在偶数个| 内
        i = 0
        nums = 0
        temp = 0
        while i < len(s):
            if s[i] == "*":
                if index == 0 or index % 2 == 0:
                    nums += 1
                else:
                    temp += 1
            elif s[i] == "|":
                index += 1
                if index % 2 == 0:
                    temp = 0
            i += 1
        return nums + temp

    def beautifulBouquet(self, flowers: List[int], cnt: int) -> int:
        """
        LCP 68. 美观的花束   经典滑动窗口  找到最大的符合要求区间。 那么其区间内的所有子区间也都符合
        :param flowers:
        :param cnt:
        :return:
        """
        c = Counter()
        res = 0
        left = 0
        for right, value in enumerate(flowers):
            c[value] += 1
            while c[value] > cnt:
                c[flowers[left]] -= 1
                left += 1
            res += right - left + 1
        return res % (10 ** 9 + 7)

    def canPartition(self, nums: List[int]) -> bool:
        """
        剑指 Offer II 101. 分割等和子集
        :param nums:
        :return:
        """

        sums = sum(nums)
        if sums % 2 != 0 or len(nums) < 2:
            return False
        mid = sums // 2
        if nums[0] > mid:
            return False

        dp = [True] + [False] * mid  # 可加入的集合加入到第i个数时，能否使为j的背包恰好填满
        for i, num in enumerate(nums):
            for j in range(mid, num - 1, -1):
                dp[j] |= dp[j - num]

        return dp[mid]

    def isAlienSorted(self, words: List[str], order: str) -> bool:
        """
        剑指 Offer II 034. 外星语言是否排序
        :param words:
        :param order:
        :return:
        """
        lens = len(words)
        if lens == 1:
            return True
        i = 0
        j = 1
        while j < lens:
            a = list(words[i])
            b = list(words[j])
            n = 0
            minlen = min(len(a), len(b))
            inbool = True  # 判断一直相同
            while n < minlen:

                if a[n] == b[n]:
                    n += 1
                else:
                    inbool = False
                    if order.index(a[n]) < order.index(b[n]):
                        break
                    else:
                        return False
            if inbool:
                if len(a) != minlen:
                    return False
            i += 1
            j += 1

        return True

    def checkXMatrix(self, grid: List[List[int]]) -> bool:
        """
        2319. 判断矩阵是否是一个 X 矩阵   正对角线： i = j 反对角线: i+j+1 = length
        :param grid:
        :return:
        """
        lens = len(grid)
        for i in range(lens):
            for j in range(lens):
                if i == j or i + j + 1 == lens:
                    if grid[i][j] == 0:
                        return False
                else:
                    if grid[i][j] != 0:
                        return False
        return True

    def isAnagram(self, s: str, t: str) -> bool:
        """
        剑指 Offer II 032. 有效的变位词
        :param s:
        :param t:
        :return:
        """
        cs = collections.Counter(s)
        ct = collections.Counter(t)
        if cs != ct:
            return False
        lens = len(s)
        find = False
        for i in range(lens):
            if s[i] != t[i]:
                return True
        if not find:
            return False

    def decodeMessage(self, key: str, message: str) -> str:
        """
        2325. 解密消息
        :param key:
        :param message:
        :return:
        """
        libs = []  # 这里的下标对应 小写字母表的下标 形成对应关系
        key = key.replace(" ", "")
        for _ in string.ascii_lowercase:
            while key[0] in libs:
                key = key[1:]
            libs.append(key[0])
            key = key[1:]
        message = list(message)
        for index, word in enumerate(message):
            if word in libs:
                message[index] = string.ascii_lowercase[libs.index(word)]
        return "".join(message)

    def alertNames(self, keyName: List[str], keyTime: List[str]) -> List[str]:
        """
        1604. 警告一小时内使用相同员工卡大于等于三次的人
        :param keyName:
        :param keyTime:
        :return:
        """

        libs = {}
        for index, key in enumerate(keyName):
            if key not in libs:
                libs[key] = [keyTime[index]]
            else:
                libs[key].append(keyTime[index])

        res = []
        for k in libs:
            if len(libs[k]) < 3:
                continue

            num = libs[k]
            num.sort()
            while len(num) >= 3:
                l = datetime.timedelta(hours=int(num[0][0:2]), minutes=int(num[0][3:]))
                r = datetime.timedelta(hours=int(num[2][0:2]), minutes=int(num[2][3:]))
                if r - l <= datetime.timedelta(hours=1):
                    res.append(k)
                    break
                else:
                    num.pop(0)

        return sorted(res)

    def addNegabinary(self, arr1: List[int], arr2: List[int]) -> List[int]:
        """
        1073. 负二进制数相加
        :param arr1:
        :param arr2:
        :return:
        """
        inta = 0
        intb = 0
        lena = len(arr1)
        lenb = len(arr2)
        arr1 = arr1[::-1]
        arr2 = arr2[::-1]
        for i in range(lena):
            inta += arr1[i] * ((-2) ** i)

        for j in range(lenb):
            intb += arr2[j] * ((-2) ** j)

        resint = inta + intb
        if resint == 0:
            return [0]
        res = []
        while resint != 0:
            if resint % -2 == 0:
                res.insert(0, 0)
                resint //= -2
            else:
                res.insert(0, 1)
                resint = (resint - 1) // -2
        return res

    def removeSubfolders(self, folder: List[str]) -> List[str]:
        """
        1233. 删除子文件夹
        :param folder:
        :return:
        """
        folder.sort()
        res = []
        for i in range(len(folder)):
            if i == 0 or not folder[i].startswith(res[-1] + '/'):
                res.append(folder[i])
        return res

    def reversePairs(self, nums: List[int]) -> int:
        """
        493. 翻转对
        :param nums:
        :return:
        """
        res = 0
        for l, current in enumerate(nums):
            temp = nums[l + 1:]
            temp.sort()
            while temp:
                if current > 2 * temp[0]:
                    res += 1
                    temp.pop(0)
                else:
                    break
        return res

    def mergeSimilarItems(self, items1: List[List[int]], items2: List[List[int]]) -> List[List[int]]:
        """
        2363. 合并相似的物品
        :param items1:
        :param items2:
        :return:
        """
        ret = {}
        for i, j in items1:
            ret[str(i)] = j

        for n, m in items2:
            if str(n) in ret:
                ret[str(n)] += m
            else:
                ret[str(n)] = m

        temp = []
        for k in ret:
            temp.append([int(k), ret[k]])
        return sorted(temp)

    def canConvertString(self, s: str, t: str, k: int) -> bool:
        """
        1540. K 次操作转变字符串
        所有位置切换次数不超过K
        相同需要切换次数的：  取 k 的余数 例如 a->b 可以第1次变换。也可以第27次变换 1%26 ，27%26
        求切换次数:  1+26*n 有相同需要变换的字符。那么使用这个来判断。
        例： abc -> bcd
        每个位置都需要一次变换，1,1,1那么转换成 1,27,53 如果k大于等于53 那么满足
        需要一个字典来存需要的变换次数 {1:已知的1的最大变换数,2:已知2的最大变换数}
        :param s:
        :param t:
        :param k:
        :return:
        """
        lens = len(s)
        if len(s) != len(t):
            return False
        libs = {}  # 用于记录相同变换次数的辅助字典

        def getfixnum(word1, word2):
            wordfix = ord(word2) - ord(word1)
            if wordfix < 0:
                wordfix += 26
            return wordfix

        for i in range(lens):
            if s[i] == t[i]:
                continue
            else:
                fixnum = getfixnum(s[i], t[i])
                temp = fixnum
                if fixnum in libs:
                    temp = libs[fixnum] + 26
                if temp > k:
                    return False
                libs[fixnum] = temp
        return True

    def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
        """
        2373. 矩阵中的局部最大值
        :param grid:
        :return:
        """
        import numpy as np
        lens = len(grid)
        res = []
        for i in range(lens - 2):  # 定位3*3的左上角y  也就是二维索引
            t = []
            for j in range(lens - 2):  # 定位 3*3的左上角x  一维索引
                temp = np.array([grid[i][j:j + 3], grid[i + 1][j:j + 3], grid[i + 2][j:j + 3]])
                t.append(max(temp.flatten()))
            res.append(t)
        return res

    def minimumRecolors(self, blocks: str, k: int) -> int:
        """
        2379. 得到 K 个黑块的最少涂色次数
        :param blocks:
        :param k:
        :return:
        """
        lens = len(blocks)

        maxblack = 0
        for n in range(k):  # 先算基础的前k个  后续遍历 就只遍历lens - k
            if blocks[n] == "B":
                maxblack += 1
            if maxblack == k:
                return 0
        j = k
        i = 1
        lastblack = maxblack
        while j < lens:
            if blocks[i - 1] == "B":
                lastblack -= 1

            if blocks[j] == "B":
                lastblack += 1

            # print(blocks[i:j+1],len(blocks[i:j+1]),blocks[i:j+1].count("B"))
            if lastblack == k:
                return 0
            maxblack = max(maxblack, lastblack)
            j += 1
            i += 1
        return k - maxblack

    def getFolderNames(self, names: List[str]) -> List[str]:
        """
        1487. 保证文件名唯一
        1. 名字相同  min后缀 1  保存执行记录
        要处理的就是  本身带后缀的  带后缀直接判断是否重复
            1. 后缀和已经执行的重复 那么后缀后再加后缀
            2. 后缀不和已经执行的重复  那么保存执行记录即可


        :param names:
        :return:
        """

        def wordindex(word: str) -> (str, int):
            """
            判断是不是带后缀，是的话返回 (名字,后缀数字）
            :param word:
            :return:
            """
            import re
            pattnum = re.compile("\(([\d]+)\)$")
            res = re.findall(pattnum, word)

            if res:
                return re.sub(pattnum, "", word), int(res[0])
            else:
                return False

        libs = {}  # 存储名字和最小后缀  name: minnum
        exed = []  # 储存比最小后缀大，但是有名字的特例
        for i, name in enumerate(names):
            # 先做判断有没有后缀
            substr = wordindex(name)
            if substr:  # 有后缀
                curname = substr[0]
                curnum = substr[1]
                if curname in libs:  # 是否在已存在的名字内
                    if curnum > libs[curname]:  # 存在 并且大于最小数 ,这时可以直接用。 但是要保存到exed
                        exed.append(name)
                    else:  # 但等于最小数，或者小于。 都取最小数来做后缀
                        names[i] = ""

        return names

    def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
        """
        2367. 算术三元组的数目
        :param nums:
        :param diff:
        :return:
        """
        lens = len(nums)
        res = 0
        for i in range(lens - 2):
            for j in range(i + 1, lens - 1):
                if nums[i] + diff == nums[j]:
                    for k in range(j + 1, lens):
                        if nums[j] + diff == nums[k]:
                            res += 1
        return res

    def halfQuestions(self, questions: List[int]) -> int:
        """
        LCS 02. 完成一半题目
        :param questions:
        :return:
        """
        n = len(questions) // 2
        c = collections.Counter(questions)
        nums = 0
        for key, value in c.most_common():
            nums += 1
            n -= value
            if n <= 0:
                return nums
        return nums

    def averageValue(self, nums: List[int]) -> int:
        """
        2455. 可被三整除的偶数的平均值
        :param nums:
        :return:
        """
        res = 0
        lenres = 0
        for num in nums:
            if num % 3 == 0:
                if num % 2 == 0:
                    res += num
                    lenres += 1
        return math.floor(res / lenres) if res else 0

    def pivotIndex(self, nums: List[int]) -> int:
        """
        剑指 Offer II 012. 左右两边子数组的和相等
        :param nums:
        :return:
        """
        left = 0
        right = sum(nums)
        lens = len(nums)

        for index in range(lens):
            if index == 0:
                left = 0
            else:
                left += nums[index - 1]
            if index == lens - 1:
                right = 0
            else:
                right -= nums[index]
            if left == right:
                return index
        return -1

    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        """
        剑指 Offer II 006. 排序数组中两个数字之和
        :param numbers:
        :param target:
        :return:
        """
        i = 0
        j = len(numbers) - 1
        while i < j:
            res = numbers[i] + numbers[j]
            if res == target:
                return [i, j]
            elif res > target:  # 和比目标大，那么往左移
                j -= 1
            else:  # 和比目标小，那么往右移
                i += 1

    def categorizeBox(self, length: int, width: int, height: int, mass: int) -> str:
        """
        2525. 根据规则将箱子分类
        :param length:
        :param width:
        :param height:
        :param mass:
        :return:
        """
        Bulky = False
        Heavy = False
        area = length * width * height
        if length >= math.pow(10, 4) or width >= math.pow(10, 4) or height >= math.pow(10, 4) or area >= math.pow(10,
                                                                                                                  9):
            Bulky = True
        if mass >= 100:
            Heavy = True

        if Bulky and Heavy:
            return "Both"
        elif Bulky:
            return "Bulky"
        elif Heavy:
            return "Heavy"
        else:
            return "Neither"

    def getWinner(self, arr: List[int], k: int) -> int:
        """
        1535. 找出数组游戏的赢家
        设置索引和当前数字  当前数字大于后续的数字为k次，就返回当前数字

        一定会有最大，所以遍历完arr ，直接返回当前最大的即可。
        :param arr:
        :param k:
        :return:
        """
        lens = len(arr)

        if arr[0] > arr[1]:
            curnum = arr[0]

        else:
            curnum = arr[1]
        curk = 1
        i = 2

        while i < lens:
            if curk == k:
                return curnum
            if curnum > arr[i]:
                curk += 1
            else:
                curnum = arr[i]
                curk = 1
            i += 1

        return curnum

    def firstUniqChar(self, s: str) -> str:
        """
        剑指 Offer 50. 第一个只出现一次的字符
        :param s:
        :return:
        """
        libs = {}
        for word in s:
            if word not in libs:
                libs[word] = 1
            else:
                libs[word] += 1
        for word in s:
            if libs[word] == 1:
                return word
        return " "

    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        """
        剑指 Offer 40. 最小的k个数
        :param arr:
        :param k:
        :return:
        """
        return sorted(arr)[0:k]

    def printNumbers(self, n: int) -> List[int]:
        """
        剑指 Offer 17. 打印从1到最大的n位数
        :param n:
        :return:
        """
        maxnum = int('9' * n) + 1
        return [i for i in range(1, maxnum)]

    def exchange(self, nums: List[int]) -> List[int]:
        """
        剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
        :param nums:
        :return:
        """
        left = []
        right = []
        for num in nums:
            if num & 1:
                left.append(num)
            else:
                right.append(num)
        return left + right

    def replaceSpace(self, s: str) -> str:
        """
        剑指 Offer 05. 替换空格
        :param s:
        :return:
        """
        return s.replace(" ","%20")

    def findRepeatNumber(self, nums: List[int]) -> int:
        """
        剑指 Offer 03. 数组中重复的数字
        :param nums:
        :return:
        """
        libs = []
        for num in nums:
            if num in libs:
                return num
            libs.append(num)

    def expectNumber(self, scores: List[int]) -> int:
        """
        LCP 11. 期望个数统计
        :param scores:
        :return:
        """
        return len(set(scores))

    def maximum(self, a: int, b: int) -> int:
        """
        面试题 16.07. 最大数值
        使用数学解法:
        max(a,b) = (|a-b|+a+b)/2
        :param a:
        :param b:
        :return:
        """
        return int((abs(a-b)+a+b)/2)

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        """
        18. 四数之和
        先排序， 然后使用两重循环+双指针 前两个数是两重循环。 后两个数用双指针。
        nums[i] + nums[j] + nums[left] + nums[right]  left = j + 1  right = n- 1
        如果 = target 则加入到答案内， 然后left右移到不同的数,然后right左移到不同的数
        如果 < target ,left 右移
        如果 > target , right 左移
        剪枝：
        1. 每层循环如果当前元素与上一次元素相同，则跳过当前元素.第一和第二重都是。
        2. 确定第一个数后：nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target 最小的和都大于目标，则退出第一层循环
        3. 确定第一个数后：nums[i] + nums[n-3] + nums[n-2] + nums[n-1] < target
            当前组合最大都小于目标，不会有更大，则第一层循环进入num[i+1]循环。
        4. 确定第二个数后：nums[i] + nums[j] + nums[j+1] + nums[j+2] > target 最小的和都大于目标，则退出第二层循环
        5. 确定第二个数后：nums[i] + nums[j] + nums[n-2] + nums[n-1] < target 最小的和都大于目标，第二层循环进入nums[j+1]循环,
            因为最大都小于目标，不会有更大的了
        :param nums:
        :param target:
        :return:
        """
        nums.sort()
        lens = len(nums)
        if lens < 4:
            return []
        res = []
        for i in range(lens-3):
            if i > 0 and nums[i] == nums[i-1]:  # 剪枝1
                continue
            if nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target:  # 剪枝2
                break
            if nums[i] + nums[lens-3] + nums[lens-2] + nums[lens-1] < target:  # 剪枝3
                continue
            for j in range(i+1, lens-2):
                if j > i + 1 and nums[j] == nums[j-1]:  # 剪枝1
                    continue
                if nums[i] + nums[j] + nums[j+1] + nums[j+2] > target:  # 剪枝4
                    break
                if nums[i] + nums[j] + nums[lens - 2] + nums[lens - 1] < target:  # 剪枝5
                    continue
                left = j + 1
                right = lens - 1
                while left < right:
                    cur_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    if cur_sum == target:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left+1]:
                            left += 1
                        left += 1
                        while left < right and nums[right] == nums[right-1]:
                            right -= 1
                        right -= 1
                    elif cur_sum < target:  # 小于目标，左指针右移变大
                        left += 1
                    else:  # 大于目标，右指针左移变小
                        right -= 1
        return res

    def twoSum2(self, nums: List[int], target: int) -> List[int]:
        """
        剑指 Offer 57. 和为s的两个数字
        排序 + 双指针
        :param nums:
        :param target:
        :return:
        """
        nums.sort()
        lens = len(nums)
        i = 0
        j = lens - 1
        while i < j:
            if nums[i] + nums[j] == target:
                return [nums[i], nums[j]]
            elif nums[i] + nums[j] > target:    # 大于目标 右坐标左移
                j -= 1
            else:
                i += 1
        return []

    def findString(self, words: List[str], s: str) -> int:
        """
        面试题 10.05. 稀疏数组搜索
        :param words:
        :param s:
        :return:
        """




if __name__ == '__main__':
    s = Solution()
    print(s.twoSum2(nums = [10,26,30,31,47,60], target = 40))
