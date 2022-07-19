#coding=utf-8
#Author miracle.why@qq.com
class ali_code:
    def __init__(self):
        pass
    def sixNumberGetTimeMaxandMIN(self,numbers):
        numbers.sort()
        i=0
        j=1
        mins=""
        maxs=""

        while(i<len(numbers)-1):
            if numbers[i]*10+numbers[j]>60:
                mins="error"
                break
            mins+=str(numbers[i]*10+numbers[j])
            i+=2
            j+=2

        numbers.sort(reverse=True)


        lager=[]
        while(numbers):
            if numbers[0]>5:
                lager.append(numbers[0])
                numbers.pop(0)
                continue
            else:
                if lager:
                    maxs+=str(numbers[0]*10+lager[0])
                    lager.pop(0)
                else:
                    maxs+=str(numbers[0])
                numbers.pop(0)
        if lager:
            maxs="error"

        return mins,maxs





if __name__ == '__main__':
    a=ali_code()

    nums=[1,3,2,4,5,6]
    print(a.sixNumberGetTimeMaxandMIN(nums))
