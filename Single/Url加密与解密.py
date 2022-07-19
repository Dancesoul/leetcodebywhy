# -*- coding:utf-8 -*-
# @FileName  :Url加密与解密.py
# @Time      :2021/10/8 0008 15:52
# @Author    :Miracle.Why
# Email      :miracle.why@qq.com
# Description:
import base64


class Codec:

    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL.
        """
        self.short=(longUrl.encode()).__str__()
        return self.host + self.short


    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL.
        """
        shortUrl=shortUrl.replace(self.host,"")
        return base64.b64decode(shortUrl).decode()


# Your Codec object will be instantiated and called as such:
codec = Codec()
print(codec.decode(codec.encode("https://leetcode.com/problems/design-tinyurl")))

