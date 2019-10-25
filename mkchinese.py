import os
import pickle

allchars = set()
with open('all.txt', 'r', encoding='utf-8') as f:
    total = f.read().strip()
    for c in total:
        allchars.add(c)

partchars = set()
with open('all_ch.txt', 'r', encoding='utf-8') as f:
    parts = f.read().strip('{}\n')
    for c in parts:
        partchars.add(c)

tot = allchars.union(partchars)
total = []
for t in tot:
    total.append(t)
print(len(total))
indices = range(len(total))

d = dict(zip(indices, total))

with open('clabels', 'w') as f:
    f.write('(dp0\n')
    f.write('I0\n')
    print(total[0])
    a = str(total[0].encode('unicode_escape'))[4:-1]
    print(a)
    f.write('V\{}'.format(a))
    f.write('\n')
    for i in range(1, len(total)):
        f.write('p{}\nsI{}\n'.format(i, i))
        a = str(total[i].encode('unicode_escape'))[4:-1]
        f.write(
            'V\{}'.format(a))
        f.write('\n')
    f.write('p{}\ns.'.format(len(total)))

