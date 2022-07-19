#coding=utf-8
#Author miracle.why@qq.com 
class MagicDictionary(object):

	def __init__(self):
		"""
		Initialize your data structure here.
		"""
		self.alldict=[]

	def buildDict(self, dict):
		"""
		Build a dictionary through a list of words
		:type dict: List[str]
		:rtype: None
		"""
		self.alldict=dict

	def search(self, word):
		"""
		Returns if there is any word in the trie that equals to the given word after modifying exactly one character
		:type word: str
		:rtype: bool
		"""
		import string
		allword=string.ascii_lowercase
		word=list(word)
		for ind,olds in enumerate(word):
			for news in allword:
				if olds==news:
					continue
				temp=word[:]
				temp[ind]=news
				newword=''.join(temp)
				if newword in self.alldict:
					return True
		return False


if __name__ == '__main__':
	s=MagicDictionary()
	s.buildDict(["hello", "leetcode"])
	print(s.search('hhllo'))
