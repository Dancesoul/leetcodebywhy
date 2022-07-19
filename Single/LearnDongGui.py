#coding=utf-8
#Author miracle.why@qq.com 
class LearnDP():
	'''
	这个类用来学习动态规划算法。
	'''
	def fib(self,num):
		'''
		获取num个斐波那锲数列
		:param num: 获取第num个斐波那锲数
		:return:
		'''
		if num==1 or num==2:
			return 1
		return self.fib(num-1)+self.fib(num-2)



if __name__=="__main__":
	dp=LearnDP()
	print(dp.fib(20))