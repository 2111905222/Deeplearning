import numpy as np

print(np.random.randint(0, high=2, size=10))

a = {'1': 10, '2': 11, '3': 13}
print(a['1'])
print('dict.pop', a.pop('1'))


l = [{'1': 1, '2': 2}, {'1': 1, '2': 2}]
l1 = [[1, 2, 3], [1, 2, 3]]
l2 = []
for i in range(len(l)):
    print(l[i]['1'])
for i in range(len(l1)):
    l2.append(l1[i])
    print('list of list', l1[i])

spam = ['cat', 'dog', 'mouse']
# print('llllllllllll', range(spam))  # range(5)生成的序列是从0开始小于5的整数 这样写会报错
# t = ({1:10,2:11,3:13})
# for history in t:
#     print(history['1'])
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 11)

fig = plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
l1, = ax1.plot(x, x*x, 'r')             #这里关键哦
l2, = ax2.plot(x, x*x, 'b')           # 注意

plt.legend([l1, l2], ['first', 'second'], loc = 'upper right')             #其中，loc表示位置的；

plt.show()