#coding=utf-8
#Author miracle.why@qq.com 
class MyHashMap:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.myhash={}

    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """

        if key in self.myhash:
            self.myhash[key]=value
        else:
            self.myhash[key]=value


    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        if key in self.myhash:
            return self.myhash.get(key)
        else:
            return -1

    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        if key in self.myhash:
            self.myhash.pop(key)


# Your MyHashMap object will be instantiated and called as such:
hashMap=MyHashMap()
hashMap.put(1, 1)
hashMap.put(2, 2)
print(hashMap.get(1))           #// 返回 1)
print(hashMap.get(3) )          #// 返回 -1 (未找到)
hashMap.put(2, 1)     #// 更新已有的值
print(hashMap.get(2))          #// 返回 1
hashMap.remove(2)        #// 删除键为2的数据
print(hashMap.get(2))          #// 返回 -1 (未找到)