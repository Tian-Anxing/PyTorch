import torch
import random
import zipfile
import sys

with zipfile.ZipFile('../jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
print(corpus_chars[:40])
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:10000]
print(corpus_chars)

#我们将每个字符映射成一个从0开始的连续整数，又称索引，来方便之后的数据处理。
# 为了得到索引，我们将数据集里所有不同字符取出来，然后将其逐一映射到索引来构造词典。接着，打印vocab_size，即词典中不同字符的个数，又称词典大小。
idx_to_char = list(set(corpus_chars)) #set()去掉重复函数
print(idx_to_char)
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
print(char_to_idx)
vocab_size = len(char_to_idx)
print(vocab_size )# 1027

#之后，将训练数据集中每个字符转化为索引，并打印前20个字符及其对应的索引。
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample])) #str.join("a","b","c")将str插入到a,b,c之间,astrbstrc
print('indices:', sample)