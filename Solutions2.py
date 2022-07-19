#coding=utf-8
#Author miracle.why@qq.com
import collections
import string
from collections import Counter
from typing import List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
    @staticmethod
    def build(values:list):
        res=ListNode(values[0])
        for index in range(1,len(values)):
            res.next=ListNode(values[index])
        return res

class Solution(object):

    def distributeCandies(self, candies):
        """
        :type candies: List[int]
        :rtype: int
        """
        lens=len(candies)
        mid=int(lens/2)
        candies.sort()
        cand_type=1
        for ind in range(1,lens):
            if candies[ind]!=candies[ind-1]:
                cand_type+=1
            if cand_type>=mid:
                return cand_type
        return cand_type

    def isPalindrome(self, x: int) -> bool:
        x=str(x)
        y=x[::-1]
        lens=len(x)
        if lens%2==0:
            if x[0:lens // 2] == y[0:lens // 2]:
                return True
        else:
            if x[0:lens//2]==y[0:lens//2]:
                return True

        return False

    def dailyTemperatures(self, T: List[int]) -> List[int]:
        #739. 每日温度
        res=[0]
        i=len(T)-1
        while(i>0):
            j=i-1
            while(j>0):
                if T[j]>T[i]:
                    j-=1
                else:
                    break
            if j>0:
                res.insert(0,i-j)
            else:
                res.insert(0,0)
            i-=1
        return res

    def numberOfSteps(self, num: int) -> int:
        #1342. 将数字变成 0 的操作次数
        n=0
        while(num!=0):
            n+=1
            if num%2==0:
                 num=num/2
            else:
                 num-=1
        return n

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        #21. 合并两个有序链表
        if not l1:   #如果l1为空  就只剩l2了
            return l2
        if not l2:   #如果l2为空 就只剩l1了
            return l1
        if l1.val<=l2.val:   #如果l1的当前节点小于l2 那么l1的下一个节点就指向下一个判断剩下的l1 和 l2    递归条件
            l1.next=self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next=self.mergeTwoLists(l1,l2.next)
            return l2

    def daysBetweenDates(self, date1: str, date2: str) -> int:
        #1360. 日期之间隔几天
        def run(year):
            return year % 4 == 0 and year % 100 != 0 or year % 400 == 0

        def get_to1971(date:str):
            mouth = [0,31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            y=int(date[0:4])
            m=int(date[5:7])
            d=int(date[8:10])

            num=0
            for year in range(1971,y):
                num+=365
                if run(year):
                    num+=1
            if run(y) and m>2:
                num+=1
            num+=sum(mouth[:m])
            num+=d
            return num

        return abs(get_to1971(date1)-get_to1971(date2))

    def longestCommonPrefix(self, strs: List[str]) -> str:
        #14. 最长公共前缀
        if not strs:
            return ""
        str0 = min(strs)
        str1 = max(strs)
        for i in range(len(str0)):
            if str0[i] != str1[i]:
                return str0[:i]
        return str0

    def sequentialDigits(self, low: int, high: int) -> List[int]:
        #1291. 顺次数
        res=[]
        nums="123456789"
        lenlow=len(str(low))
        lenhigh=len(str(high))

        for start in range(lenlow,lenhigh+1):  #控制位数
            i=0  #控制位置
            num=int(nums[i:start + i])
            while (start+i<10):
                if num not in res and num>=low and num<=high:
                    res.append(num)
                i += 1
                num = int(nums[i:start + i])

        return res

    def trailingZeroes(self, n: int) -> int:
        #面试题 16.05. 阶乘尾数
        res=0
        while(n>0):
            n=n//5
            res+=n
        return res

    def findIntegers(self, num: int) -> int:
        #600. 不含连续1的非负整数  目前是超时
        binnum=str(bin(num))
        lens=len(binnum)
        repeat=[]
        resnum=0
        base=["0b"]
        i=2
        while(i<lens):
            temp=[]
            for b in base:
                r=b+"1"
                l=b+"0"
                intnuml=int(l,2)
                if intnuml<=num and intnuml not in repeat:
                    repeat.append(intnuml)
                    temp.append(l)
                    resnum+=1
                if "11" not in r:
                    intnum=int(r,2)
                    if intnum<=num and intnum not in repeat:
                        resnum+=1
                        repeat.append(intnum)
                        temp.append(r)
            i+=1
            base=temp
        return resnum

    def sumNums(self, n: int) -> int:
        #面试题64. 求1+2+…+n
        return n != 0 and n + self.sumNums(n - 1)


    def longestWord(self, words: List[str]) -> str:
        #720. 词典中最长的单词
        set_words=set(words)
        lens=len(words)
        words.sort()  #字典序排序
        words.sort(key=len,reverse=True)   #长度排序
        i=0
        while(i<lens):
            temp=words[i]
            lentemp=len(temp)
            error=False
            for j in range(lentemp,0,-1):
                if temp[:j] not in set_words:
                    error=True
                    break
            if not error:
                return temp
            i+=1
        return ""

    def maxScoreSightseeingPair(self, A: List[int]) -> int:
        #1014. 最佳观光组合
        lens=len(A)
        i=1
        inum=A[0]
        res=0
        while(i<lens):
            res=max(inum+A[i]-i,res)
            inum = max(inum, A[i] + i)
            i+=1
        return res

    def findLength(self, A: List[int], B: List[int]) -> int:
        #718. 最长重复子数组
        pass

    def isUnique(self, astr: str) -> bool:
        #面试题 01.01. 判定字符是否唯一
        #return len(astr)==len(set(astr))
        i=0
        while(i<len(astr)):
            if astr.count(astr[i])>1:
                return False
            i+=1
        return True

    def CheckPermutation(self, s1: str, s2: str) -> bool:
        #面试题 01.02. 判定是否互为字符重排
        return sorted(s1)==sorted(s2)

    def replaceSpaces(self, S: str, length: int) -> str:
        #面试题 01.03. URL化
        return S[:length].replace(" ","%20")

    def canPermutePalindrome(self, s: str) -> bool:
        #面试题 01.04. 回文排列
        lens=len(s)
        if lens == 1 or lens==0:
            return True
        lib={"1":0,"2":0}

        for word in set(s):
            if s.count(word)%2==0:
                lib["2"]+=1
            else:
                lib["1"]+=1
                if lib["1"]>1:
                    return False
        if lens%2!=0:
            if lib["1"]!=1:
                return False
        else:
            if lib["1"]!=0:
                return False

        return True

    def oneEditAway(self, first: str, second: str) -> bool:
        #面试题 01.05. 一次编辑
        if first==second:
            return True
        len1=len(first)
        len2=len(second)
        if abs(len1-len2)>1:
            return False
        change=0
        firstindex=len1-1
        secondindex=len2-1
        while(firstindex>=0 and secondindex>=0 and change<2):
            if first[firstindex] !=second[secondindex]:
                change+=1
                if len(first)>len(second):
                    firstindex-=1
                    continue
                elif len(first)<len(second):
                    secondindex-=1
                    continue
            firstindex-=1
            secondindex-=1
        if change+firstindex+1+secondindex+1<2:
            return True
        else:
            return False

    def compressString(self, S: str) -> str:
        #面试题 01.06. 字符串压缩
        i=0
        res=""
        lens=len(S)
        while(i<lens):
            j=1
            while(j+i<lens):
                if S[i]==S[i+j]:
                    j+=1
                else:
                    break
            res = res + S[i] + str(j)
            i=i+j
        if len(res)<lens:
            return res
        else:
            return S

    def rotate(self, matrix: List[List[int]]) -> None:
        """
		Do not return anything, modify matrix in-place instead.
		"""
        '''
        lens=len(matrix)
        temp=[[0]*lens for i in range(lens)]
        for i in range(lens):
            for j in range(lens):
                temp[j][lens-1-i]=matrix[i][j]
        matrix[:]=temp
        '''

        n = len(matrix)
        # 水平翻转
        for i in range(n // 2):
            for j in range(n):
                matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
        # 主对角线翻转
        for i in range(n):

            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    def addBinary(self, a: str, b: str) -> str:
        #67. 二进制求和
        return bin(int(a,2)+int(b,2)).replace("0b","")

    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        #209. 长度最小的子数组
        if not nums:
            return 0

        n = len(nums)
        ans = n + 1
        start = 0
        end = 0
        total = 0
        while end < n:
            total += nums[end]
            while total >= s:
                ans = min(ans, end - start + 1)
                total -= nums[start]
                start += 1
            end += 1

        return 0 if ans == n + 1 else ans

    def findKthLargest(self, nums: List[int], k: int) -> int:
        #215. 数组中的第K个最大元素
        return sorted(nums,reverse=True)[k]

    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        #378. 有序矩阵中第K小的元素
        temp=[]
        for v in range(len(matrix)):
            temp+=matrix[v]
        temp.sort()
        return temp[k-1]

    def setZeroes(self, matrix: List[List[int]]) -> None:
        #编写一种算法，若M × N矩阵中某个元素为0，则将其所在的行与列清零。
        lenx=len(matrix)
        leny=len(matrix[0])
        x=[]
        y=[]
        for i in range(lenx):
            for j in range(leny):
                if matrix[i][j] ==0:
                    if i not in x:
                        x.append(i)
                    if j not in y:
                        y.append(j)
        for cx in x:
            for cy in range(leny):
                matrix[cx][cy]=0

        for cy in y:
            for cx in range(lenx):
                matrix[cx][cy] = 0

    def isFlipedString(self, s1: str, s2: str) -> bool:
        #面试题 01.09. 字符串轮转
        len1=len(s1)
        len2=len(s2)
        if len1 !=len2:
            return False
        if len1 >0:
            ind=s2.find(s1[0])
            while(ind<len(s2)):
                if s2[ind:]+s2[:ind]==s1:
                    return True
                ind = s2.find(s1[0],ind+1)
                if ind == -1:
                    return False
        else:
            return True

    def removeDuplicateNodes(self, head: ListNode) -> ListNode:

        if not head:
            return head
        occurred = head.val
        pos = head
        # 枚举前驱节点
        while pos:
            # 当前待删除节点
            cur = pos.next
            if cur.val not in occurred:
                occurred.append(cur.val)
                pos = pos.next
            else:
                pos.next = pos.next.next
        return head

    def countSmaller(self, nums: List[int]) -> List[int]:
        #315. 计算右侧小于当前元素的个数
        res=[]
        for i in range(len(nums)):
            temp=sorted(nums[i:])
            index=temp.index(nums[i])
            res.append(index)
        return res

    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        #350. 两个数组的交集 II
        res=[]
        lib={}
        for v in nums1:
            if v not in lib:
                lib[v]=1
            else:
                lib[v]+=1

        for vv in nums2:
            if vv in lib and lib[vv]>=1:
                res.append(vv)
                lib[vv]-=1
        return res

    def myPow(self, x: float, n: int) -> float:
        #50. Pow(x, n)  没通过 超时
        negative=False
        if n==0:
            return 1
        if n<0:
            negative=True
            n=abs(n)
        res=1
        while(n):
            res*=x
            n-=1
        if negative:
            res=1/res
        return round(res,5)


    def searchInsert(self, nums: List[int], target: int) -> int:
        #35. 搜索插入位置
        if target in nums:
            return nums.index(target)
        else:
            nums.append(target)
            nums.sort()
            return nums.index(target)

        #法2

        lens=len(nums)
        if target<nums[0]:
            return 0
        if target>nums[lens-1]:
            return lens
        mid=lens//2
        ind=0

        while(mid!=0):
            add=False
            if target >=nums[mid]:
                nums=nums[mid:]
                add=True
            else:
                nums=nums[:mid]
            if add:
                ind+=mid
            mid//=2

        for num in range(len(nums)):
            if target==nums[num]:
                break
            elif target>nums[num]:
                ind+=1
            elif target<nums[num]:
                break

        return ind

    def minArray(self, numbers: List[int]) -> int:
        #剑指 Offer 11. 旋转数组的最小数字
        i=0
        j=len(numbers)-1
        while(i<j):
            if numbers[i]<numbers[j]:
                return numbers[i]
            elif numbers[j]<numbers[j-1]:
                return numbers[j]
            else:
                j-=1
        return numbers[0]

    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        #336. 回文对
        lens=len(words)
        res=[]
        for i in range(lens):
            for j in range(lens):
                if i==j:
                    continue
                temp=words[i]+words[j]
                if temp==temp[::-1]:
                    res.append([i,j])
        return res

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        #49. 字母异位词分组
        temp={}
        for word in strs:
            trans="".join(sorted(word))
            if trans in temp:
                temp[trans].append(word)
            else:
                temp[trans]=[word]
        return list(temp.values())


    def reverseWords(self, s: str) -> str:
        #剑指 Offer 58 - I. 翻转单词顺序
        import re
        patt=re.compile("[ ]+")
        s=re.split(patt,s)
        while("" in s):
            s.remove("")
        s.reverse()
        return " ".join(s)

    def countSubstrings(self, s: str) -> int:
        #647. 回文子串
        lens=len(s)
        res=lens
        step=2
        while(step<=lens):
            i=0
            while(i<=lens-step):
                value=s[i:i+step]
                if value=="".join(list(reversed(value))):
                    res+=1
                i+=1
            step+=1
        return res

    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        #
        from functools import reduce

        Trie = lambda: collections.defaultdict(Trie)
        trie = Trie()
        END = True

        for root in dictionary:
            reduce(dict.__getitem__, root, trie)[END] = root

        def replace(word):
            cur = trie
            for letter in word:
                if letter not in cur or END in cur: break
                cur = cur[letter]
            return cur.get(END, word)

        return " ".join(map(replace, sentence.split()))

    def backspaceCompare(self, S: str, T: str) -> bool:
        #844. 比较含退格的字符串
        '''
        法1
        runstrs=""
        runstrt=""
        while(S or T):
            if S:
                if S[0] =="#":
                    if runstrs:
                        runstrs=runstrs[:-1]
                else:
                    runstrs+=S[0]
                S = S[1:]
            if T:
                if T[0]=="#":
                    if runstrt:
                        runstrt=runstrt[:-1]
                else:
                    runstrt+=T[0]
                T = T[1:]
        return runstrt==runstrs
        '''

        def doit(s):
            sindex=0
            s=list(s)
            for i in range(len(s)-1,-1,-1):
                if s[i]=="#":
                    sindex+=1
                    s.pop(i)
                elif sindex!=0:
                    s.pop(i)
                    sindex-=1
            return s
        return doit(S)==doit(T)


    def longestMountain(self, A: List[int]) -> int:
        #845. 数组中的最长山脉
        lens=len(A)
        if lens<3:
            return 0
        res=0
        i=0

        while(i<lens):
            temp = []
            goon = True
            symbol=True  # True 上坡 i>i-1 , False 下坡 i<i-1
            while(goon and i<lens):

                if not temp:  #temp为空
                    temp.append(A[i])
                    i+=1
                else:  #temp不为空

                    if symbol:  #上坡
                        if A[i]>A[i-1]:
                            temp.append(A[i])
                            i+=1
                        elif A[i]<A[i-1]:
                            if len(temp)==1:
                                goon=False
                            else:
                                symbol=False
                                temp.append(A[i])
                                i+=1
                        else:
                            goon=False
                    else:
                        if A[i]<A[i-1]:
                            temp.append(A[i])
                            i+=1
                        else:
                            goon=False


            if not symbol:
                print(temp)
                res=max(res,len(temp))
                i-=1
        return res

    def singleNumber(self, nums: List[int]) -> int:
        #剑指 Offer 56 - II. 数组中数字出现的次数 II
        from collections import Counter

        return Counter(nums).most_common()[-1][0]

    def reverseString(self, s: List[str]) -> None:
        #递归学习
        def helper(i: int,j:int, s: List[str]):
            if i>=j:
                print(s)
                return
            temp=s[i]
            s[i]=s[j]
            s[j]=temp

            helper(i+1,j-1,s)

        helper(0,len(s)-1,s)

    def generate(self, numRows: int) -> List[List[int]]:
        #递归学习 -杨辉三角
        momo = {}

        def helper(row,col):

            if col==1 or row==col:
                momo[(row, col)]=1
            elif (row,col) in momo:
                return momo[(row,col)]
            else:
                momo[(row,col)]=helper(row-1,col-1)+helper(row-1,col)
            return momo[(row,col)]

        res=[]
        for row in range(1,numRows+1):
            rows=[]
            for col in range(1,row+1):
                rows.append(helper(row,col))
            res.append(rows)

        return res



    def getRow(self, rowIndex: int) -> List[int]:
        # 递归学习 -杨辉三角
        momo = {}

        def helper(row, col):

            if (row, col) in momo:
                return momo[(row, col)]
            elif col == 1 or row == col:
                momo[(row, col)] = 1
            else:
                momo[(row, col)] = helper(row - 1, col - 1) + helper(row - 1, col)
            return momo[(row, col)]

        res = []

        for col in range(1,rowIndex+2):
            res.append(helper(rowIndex+1,col))

        return res

    def partitionLabels(self, S: str) -> List[int]:
        #763. 划分字母区间

        spos={_:-1 for _ in string.ascii_lowercase}

        for ind,val in enumerate(S):
            spos[val]=max(spos[val],ind)

        res=[]
        l=-1
        r=-1
        for i,v in enumerate(S):
            if l==-1:
                l=i
                r=spos[v]
            if i==r:
                res.append(r-l+1)
                l=-1
                r=-1
                continue
            r=max(r,spos[v])

        return res


    def isPalindrome(self, head: ListNode) -> bool:
        #234. 回文链表
        res = []
        if head is None:
            return True

        def helper(x: ListNode):
            res.append(x.val)
            if x.next is None:
                return res
            else:
                return helper(x.next)

        helper(head)
        return res == list(reversed(res))

    def videoStitching(self, clips: List[List[int]], T: int) -> int:
        # 1024. 视频拼接
        def add_done(l,r):
            while(l<=r):
                done[l]=True
                l+=1

        done=[False for _ in range(T+1) ]
        res=[]
        def helper(start,stop,clips:List[List[int]]):
            if start>=stop:
                return
            for l,r in clips:
                if l<=start:
                    add_done(l,r)
                    if [l,r] not in res:
                        res.append([l,r])
                    start=r
                    clips.remove([l,r])
                    return helper(start, stop, clips)
                if r>=stop:
                    add_done(l,r)
                    if [l,r] not in res:
                        res.append([l,r])
                    stop=l
                    clips.remove([l, r])
                    return helper(start,stop,clips)

        helper(0,T,clips)

        if False in done:
            return -1

        return len(res)


    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        #1365. 有多少小于当前数字的数字
        temp=sorted(nums)
        res=[]
        for i in nums:
            res.append(temp.index(i))
        return res

    def majorityElement(self, nums: List[int]) -> int:
        # 剑指 Offer 39. 数组中出现次数超过一半的数字
        for i in set(nums):
            if nums.count(i)>len(nums)//2:
                return i

    def removeComments(self, source: List[str]) -> List[str]:
        # 722. 删除注释
        '''
        block=False
        line="//"
        blockstart="/*"
        blockend="*/"
        res=[]
        lens=len(source)
        i=0
        temp=""
        while(i<lens):   #控制整个list循环
            current=source[i]    #当前字符串
            goon=True       #是否继续处理当前字符串 如果是行注释 就不需要继续处理
            while(goon):    #控制当前字符串循环
                if block==True:  #如果是块注释中 直接找块结束 找到继续处理字符串
                    if  blockend in current:  #找块结束点
                        current=current[current.index(blockend)+2:]
                        block=False
                    else:        #没找到结束点 这一行直接跳过
                        goon=False

                else:           #不是块注释中
                    if blockstart in current:
                        blockind = current.index(blockstart)
                        if  line in current:   #两个都存在的情况

                            lineind=current.index(line)
                            if lineind<blockind: #行注释在前,行注释之后的抛弃
                                temp+=current[:lineind]
                                goon=False
                            else:
                                temp += current[:blockind]
                                current = current[blockind + 2:]
                                block = True

                        else:   # 块注释在前或者只有块注释  后面的字符串还是需要处理
                            temp+=current[:blockind]
                            current=current[blockind+2:]
                            block=True
                    elif line in current:  #只有行注释  后面内容不需要处理
                        temp+=current[:current.index(line)]
                        goon=False
                    else:
                        temp+=current
                        goon=False

            if not block:
                if temp!="":
                    res.append(temp)
                    temp=""
            i+=1
        return res
        '''
        subsymbol="$$,$$"
        source_str =source[0]
        for i in range(1,len(source)):
            source_str+=subsymbol
            source_str+=source[i]
        source_str+=subsymbol

        res=""
        while(source_str):
            if source_str[0] =="/":

                if source_str[1]=="*":  #块注释
                    end=source_str[2:].index("*/")
                    source_str=source_str[2:][end+2:]
                    continue
                elif source_str[1]=="/": #行注释
                    end=source_str.index(subsymbol)
                    source_str=source_str[end:]
                    continue
            res+=source_str[0]
            source_str=source_str[1:]
        res=res.split(subsymbol)
        while("" in res):
            res.remove("")
        return res

    def preorderTraversal(self, root: TreeNode) -> List[int]:
        #144. 二叉树的前序遍历
        res=[]
        def helper(root:TreeNode):
            if root is not None:
                res.append(root.val)
            helper(root.left)
            helper(root.right)
        helper(root)
        return res

    def myPow(self, x: float, n: int) -> float:
        #实现 pow(x, n) ，即计算 x 的 n 次幂函数。
        if n==0:
            return 1
        if n<0:
            return 1/self.myPow(x,-n)
        res=self.myPow(x,n//2)
        if n%2>0:  #奇数
            return x*res*res

        return res*res

    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # 349. 两个数组的交集
        return list(set(nums1)&set(nums2))

    def validMountainArray(self, A: List[int]) -> bool:
        # 941. 有效的山脉数组
        lens=len(A)
        if lens<3:
            return False
        l=0
        r=lens-1

        while(l<r and A[l] < A[l+1]):
            l+=1
        while(r>l and A[r]<A[r-1]):
            r-=1

        return l==r and l!=0 and r!=lens-1


    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        # 973. 最接近原点的 K 个点

        return sorted(points,key= lambda x:abs(pow(x[0],2)+pow(x[1],2)))[:K]


    def missingNumber(self, nums: List[int]) -> int:
        #
        nums.sort()
        lens=len(nums)

        i=0
        if nums[i]!=i:
            return i
        while(i<lens):
            if  nums[i]!=nums[i+1]-1:
                return nums[i]+1
            i+=1
        return lens


    def advantageCount(self, A: List[int], B: List[int]) -> List[int]:
        # 870. 优势洗牌

        res={b:[] for b in B}
        sortedB=sorted(B)
        A.sort()
        remin=[]

        j=0
        for a in A:
            if a >sortedB[j]:
                res[sortedB[j]].append(a)
                j+=1
            else:
                remin.append(a)

        realres=[]
        for b in B:
            if res[b]:
                realres.append(res[b].pop())
            else:
                realres.append(remin.pop())

        return realres



    def sortArrayByParityII(self, A: List[int]) -> List[int]:
        # 922. 按奇偶排序数组 II
        '''
        singleA=[]
        doubleA=[]
        for a in A:
            if a%2==0:
                doubleA.append(a)
            else:
                singleA.append(a)

        for ind,val in enumerate(A):
            if ind%2==0:
                A[ind]=doubleA.pop()
            else:
                A[ind]=singleA.pop()

        return A

        '''
        lens=len(A)
        i=0  #偶数下标
        j=1  #奇数下标

        errori=False
        errorj=False
        while(j<lens and i<lens):
            if A[i]%2==0:
                i+=2
            else:
                errori=True
            if A[j]%2!=0:
                j+=2
            else:
                errorj=True

            if errorj and errori:
                temp=A[i]
                A[i]=A[j]
                A[j]=temp
                i+=2
                j+=2
                errorj=False
                errori=False
        return A

    def maxSubArray(self, nums: List[int]) -> int:
        # 剑指 Offer 42. 连续子数组的最大和
        lens=len(nums)
        maxnum=nums[0]
        for i in range(1,lens):
            if nums[i-1]>0:
                nums[i]=nums[i]+nums[i-1]
            maxnum=max(maxnum,nums[i])
        return maxnum


    def kthGrammar(self, N: int, K: int) -> int:

        def exestr(s):
            res=""
            while(s):
                if s[0]=="0":
                    res+="01"
                else:
                    res+="10"
                s=s[1:]
            return res

        def helper(n) -> str:
            if n==1:
                return "0"
            if n==2:
                return "01"
            return exestr(helper(n-1))

        return int(helper(N)[K-1])

    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        # 406. 根据身高重建队列
        people.sort(key=lambda x:(-x[0],x[1]))  #按 x[0]降序排,x[1]升序排
        res=[]
        for p in people:
            if p[1]>=len(res):  #如果K符合规则 直接添加到答案
                res.append(p)
            else:               # 不符合K 把当前p插入到K位置 以符合答案
                res.insert(p[1],p)
        return res


    def lengthLongestPath(self, input: str) -> int:
        # 388. 文件的最长绝对路径
        res=[]
        temp=""
        symbol="\\t"
        if symbol not in input:
            return 0
        num=1
        i=0
        lens=len(input)
        while(i<lens):
            f=input.find(symbol*num,i) # 每次尝试找更多的级数 num表示级数
            if f==-1:  # 没找到更大的级数 说明上一次就是最大的级数  开始回溯
                num-=1
                i=lens
                while(num>=1):
                    f = input.find(symbol * num)  #倒着找
                    ff=input.find(symbol*num,f,lens)
                    start=f+len(symbol*num) #
                    r=start
                    while(r<lens and input[r]!="\\"): # 找当前这个级的目录名
                        r+=1
                    res.insert(0,input[start:r])
                    num-=1
                    i=f

            else:
                i=f
                num+=1

        return len("/".join(res))

    def sortString(self, s: str) -> str:
        # 1370. 上升下降字符串
        res=""
        lib={}
        for key in s:
            if key not in lib:
                lib[key]=1
            else:
                lib[key]+=1

        if len(lib)==1:
            return s

        getbool=True  # True 表示从左边拿 也就是升1-3  False 表示从右边拿 也就是降
        while(lib.keys()):

            if getbool:
                for v in sorted(lib.keys()):
                    res+=v
                    lib[v]-=1
                    if lib[v]==0:
                        lib.pop(v)
                getbool=False
            else:
                for v in sorted(lib.keys(),reverse=True):
                    res+=v
                    lib[v]-=1
                    if lib[v]==0:
                        lib.pop(v)
                getbool=True
        return res

    def breakfastNumber(self, staple: List[int], drinks: List[int], x: int) -> int:
        # LCP 18. 早餐组合
        staple.sort()
        drinks.sort()
        lens=len(staple)
        lend=len(drinks)
        i=0
        j=lend-1
        res=0
        while(i<lens and j>=0):
            if staple[i]+drinks[j]<=x:
                res+=j+1
                i+=1
                continue
            else:
                j-=1
        return res

    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        # 187. 重复的DNA序列

        l = []
        for i in range(len(s)-9):
            l.append(s[i:i+10])
        c = Counter(l)
        res = []
        for k,v in c.items():
            if v > 1:
                res.append(k)
        return res

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # 剑指 Offer II 074. 合并区间
        intervals = sorted(intervals,key= lambda x:x[0])
        res = []
        start = intervals[0][0]
        end = intervals[0][1]
        for i in range(1,len(intervals)):
            l = intervals[i][0]
            r = intervals[i][1]
            if l > end:
                res.append([start,end])
                start = l
                end = r
            else:
                if r > end:
                    end = r
        res.append([start,end])
        return res

    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        # 867. 转置矩阵
        return list(zip(*matrix))

    def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
        # 1380. 矩阵中的幸运数
        res = []
        m = 0
        lens = len(matrix)
        while m < lens:
            value = min(matrix[m])
            c = matrix[m].index(value)  # 这一行中最小的位置 也就是列
            cmax = value
            find = True
            for i in range(lens):
                if matrix[i][c] > cmax:
                    find = False
                    break
            if find:
                res.append(value)
            m+=1
        return res

    def findErrorNums(self, nums: List[int]) -> List[int]:
        # 645. 错误的集合
        setnums = set(nums)
        return [Counter(nums).most_common(2)[0][0],list(set(range(1,len(nums)+1)) - setnums)[0]]

    def singleNumber(self, nums: List[int]) -> int:
        # 137. 只出现一次的数字 II
        return Counter(nums).most_common()[-1][0]

    def validTicTacToe(self, board: List[str]) -> bool:
        # 794. 有效的井字游戏
        # 数量如果相同 那么一定不是X胜利 O有可能胜利   需要判断是不是X胜利  对角线048 642  横向012 345 678 纵向036 147 258
        # 数量不相同情况下  X 一定比 O多1
        # 仅有一个 那么一定是X
        def winner(char):
            for i in range(3):
                if char==board[i][0] == board[i][1] == board[i][2]:
                    return True
                if char==board[0][i] == board[1][i] == board[2][i]:
                    return True
            if char== board[0][0] == board[1][1] == board[2][2]:
                return True
            if char== board[0][2] == board[1][1] == board[2][0]:
                return True

        boardstr = "".join(board)
        x = 0
        o = 0
        for v in boardstr:
            if v == "X":
                x +=1
            if v =="O":
                o += 1
        if x + o ==1:
            if x !=1:
                return False
        if x!=o and x-o !=1:
            return False
        if winner("X") and x -o !=1:
            return False
        if winner("O") and x != o:
            return False
        return True


    def fizzBuzz(self, n: int) -> List[str]:
        # 412. Fizz Buzz
        res= []
        for i in range(1,n+1):
            if i % 15 == 0:
                res.append("FizzBuzz")
            elif i%3 == 0:
                res.append("Fizz")
            elif i%5 == 0:
                res.append("Buzz")
            else:
                res.append(str(i))
        return res

    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        # 475. 供暖器

        maxnum = -float("inf") # 找出离加热器最远的距离
        len1 = len(houses)
        len2 = len(heaters)
        i = 0
        j = 0
        while True:
            maxnum = max(abs(heaters[j]-houses[i]),maxnum)
            if len2 -1 == j and len1 -1 == i:
                break
            if len2 - 1 != j:
                j+=1
            if len1 -1 != i:
                i+=1

        return maxnum

    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        # 剑指 Offer II 069. 山峰数组的顶部
        lens = len(arr)
        i = 1
        j = lens -2
        while i<=j:
            if arr[i-1] < arr[i] > arr[i+1]:
                return i
            else:
                i+=1
            if arr[j-1] < arr[j] > arr[j+1]:
                return j
            else:
                j-=1

    def countAndSay(self, n: int) -> str:
        # 38. 外观数列
        i = 1
        if n == 1:
            return str(n)
        def helper(num: str) -> str:
            last = num[0]
            lastnum = 1
            lens = len(num)
            j = 1
            res = ""
            while j < lens:
                if num[j] == last:
                    lastnum+=1
                else:
                    res = res + str(lastnum)+last
                    last = num[j]
                    lastnum = 1
                j+=1
            return res + str(lastnum)+last
        resstr = str(i)
        while i <n:
            resstr = helper(resstr)
            i+=1
        return resstr



if __name__ == "__main__":
    s = Solution()

    print(s.partitionDisjoint([5,0,3,8,6]))

