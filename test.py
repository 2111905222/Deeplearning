import numpy as np
print(np.random.randint(0, high=2, size=10))

a = {'1':10,'2':11,'3':13}
print(a['1'])
print('dict.pop', a.pop('1'))

def test_mainkw(*k, **kw):
    print(type(kw))
    print(type(k))
    print(kw.keys())
    print('kw.item', type(kw.values()))
    print(list(kw.values()))
    print(kw.popitem())

l = [{'1':1, '2':2}, {'1':1, '2':2}]
l1 = [[1,2,3], [1,2,3]]
l2 = []
for i in range(len(l)):
    print(l[i]['1'])
for i in range(len(l1)):
    l2.append(l1[i])
    print('list of list',l1[i])
test_mainkw(name=12,list=1)
# t = ({1:10,2:11,3:13})
# for history in t:
#     print(history['1'])
