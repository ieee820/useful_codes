# coding: utf-8
from glob import glob
import pickle
from crnn.keys import alphabetChinese

# print(alphabetChinese[20:30])
key_num = len(alphabetChinese)
print('key num: ', key_num)
line_length = 10
iter_len = int(len(alphabetChinese) / 10)
print('iter_len', iter_len)

counter = 0
quit = 1

#the key dict has two '|' char , so should reduce
signal = 0
with open('./balance_sample_len10.txt', 'w') as f:
    for i in alphabetChinese:
        if i == '|':
            print('found: |')
            continue
            # signal += 1  #for reduce the '|' char
            # print('char: ', i , 'signal: ', signal)
            # if signal == 2:
            #     print('the second | loop continue')
            #     continue
        # if signal > 1:
        #     print('the second | loop continue')
        #     continue
        # ucode = ord(i)
        # if counter == quit:
        #     break
        for j in range(1, iter_len + 1):
            # print('j', j)
            # print(line_length * (j - 1), line_length * j, '%%', key_num - j*line_length)
            if key_num - j*line_length > line_length:
                line = i + alphabetChinese[line_length * (j - 1):line_length * j]
            else:
                line = i + alphabetChinese[line_length * (j - 1):key_num]
                # print(line)
            #handle the double '|' in keys
            if '|' in line:
                if j % 2 == 0:
                    line = line.replace('|', '')
                    print(line)
            f.write('%s\n' % line)
        counter += 1
