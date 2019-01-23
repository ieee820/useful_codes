# coding: utf-8
from collections import defaultdict
from glob import glob
import pickle
"""Count words."""

def count_words(s):
    """Return the n most frequently occuring words in s."""
    split_s = list(s)
    map_list = [(k,1) for k in split_s]
    output = defaultdict(int)
    i = 0
    for d in map_list:
        # if i % 1000 == 0:
        #     # print(d)
        output[d[0]] += d[1]
        i += 1

    output1 = dict(output)
    top_n = sorted(output1.items(), key=lambda pair:pair[0], reverse=False)
    top_n = sorted(top_n, key=lambda pair:pair[1], reverse=True)

    return top_n


def count_dir_and_save():
    trainP = glob('/data/share_data/crnn_train/gen_from_360w/train_1w/*/*.txt')
    txt_sum = ''
    for txt_f in trainP:
        with open(txt_f)  as f:
            label = f.read().strip()
        txt_sum += label
    print('begin count_words...')
    freq_dict = count_words(txt_sum)
    with open('./dic.pkl', 'wb') as f:
        pickle.dump(freq_dict, f)


def count_txt_and_save():
    txt_sum = ''
    with open('balance_sample_len10_25w.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            txt_sum += line.strip()

    freq_dict = count_words(txt_sum)
    with open('./dic.pkl', 'wb') as f:
        pickle.dump(freq_dict, f)


def load_pkl():
    with open('./dic.pkl', 'rb') as f:
        uppickler = pickle.Unpickler(f)
        freq_dict = uppickler.load()
        # print(freq_dict)
    return freq_dict


if __name__ == '__main__':
     # count_txt_and_save()
     freq_dict = load_pkl()
     print('keys: ', len(freq_dict), '\ncontent: ', freq_dict)
     for char in freq_dict:
         if char[0] == 'S':
             print(char)
