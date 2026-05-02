# -*- coding:UTF8 -*-

path = './senticnet-5.0/senticnet5.txt'
w_path = './senticNet/senticnet_word.txt'

fp = open(path, 'r')
w_fp = open(w_path,  'w')

count = 0
sentic_dic = {}

for line in fp:
    line  =  line.strip()
    if not line:
        continue
    items = line.split('\t')
    phrase = items[0]
    if '_' in phrase:
        continue
    #phrase = ' '.join(items[0].split('_'))
    score = items[-1]
    sentic_dic[phrase] = score
fp.close()

#sentic_dic = sorted(sentic_dic.items(), key=lambda a: len(a[0].split(' ')), reverse=True)
for word in sentic_dic:
    score = sentic_dic[word]
    string = word + '\t' + score + '\n'
    w_fp.write(string)
w_fp.close()
