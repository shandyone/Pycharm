#!/user/bin/python
#coding:utf-8
__author__ = 'shandyone'
from gensim.corpora import WikiCorpus#use wiki class to process related open
import opencc
import jieba
import codecs
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

'''
读取中文wiki语料库，并解析提取xml中的内容
'''
def dataprocess():
    space=b' '
    i=0
    output=open('/home/shandyone/Pycharm/word2vec/zhwiki/zhwiki-raw.txt','wb')
    wiki=WikiCorpus('/home/shandyone/Pycharm/word2vec/zhwiki/zhwiki-latest-pages-articles.xml.bz2',lemmatize=False,dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text)+b'\n')
        i=i+1
        if(i%10000==0):
            print('Saved '+str(i)+' articles')
    output.close()
    print('Finished Saved '+str(i)+' articles')

'''
加载停用词表
'''
def createstoplist(stoppath):
    print('load stopwords...')
    stoplist=[line.strip() for line in codecs.open(stoppath,'r',encoding='utf-8').readlines()]
    stopwords={}.fromkeys(stoplist)
    return stopwords

'''
过滤英文
'''
def isAlpha(word):
    try:
        return word.encode('ascii').isalpha()
    except UnicodeEncodeError:
        return False

'''
opencc繁体转简体，jieba中文分词
'''
def trans_seg():
    stopwords=createstoplist('/home/shandyone/Pycharm/word2vec/zhwiki/stopwords.txt')
    cc=opencc.OpenCC('t2s')
    i=0
    with codecs.open('/home/shandyone/Pycharm/word2vec/zhwiki/zhwiki-segment.txt','w','utf-8') as wopen:
        print('开始...')
        with codecs.open('/home/shandyone/Pycharm/word2vec/zhwiki/zhwiki-raw.txt','r','utf-8') as ropen:
            while True:
                line=ropen.readline().strip()
                i+=1
                print('line '+str(i))
                text=''
                for char in line.split():
                    if isAlpha(char):
                        continue
                    char=cc.convert(char)
                    text+=char
                words=jieba.cut(text)
                seg=''
                for word in words:
                    if word not in stopwords:
                        if len(word)>1 and isAlpha(word)==False: #去掉长度小于1的词和英文
                            if word !='\t':
                                seg+=word+' '
                wopen.write(seg+'\n')
    print('结束!')


def is_alpha(tok):
    try:
        #return tok.encode('ascii').isalpha()
        return tok.encode('utf-8').isalpha()
    except UnicodeEncodeError:
        return False


def zhwiki_segment():
    i = 0
    output = codecs.open('/home/shandyone/Pycharm/word2vec/zhwiki/zhwiki-segment.txt', 'w', encoding='utf-8')
    print('Start...')
    with codecs.open('/home/shandyone/Pycharm/word2vec/zhwiki/zhwiki-t2s.txt', 'r', encoding='utf-8') as raw_input:
        for line in raw_input.readlines():
            line = line.strip()
            i += 1
            print('line ' + str(i))
            text = line.split()
            text = [w for w in text if not is_alpha(w)]

            word_cut_seed = [jieba.cut(t) for t in text]
            tmp = ''
            for sent in word_cut_seed:
                for tok in sent:
                    tmp += tok + ' '
            tmp = tmp.strip()
            if tmp:
                output.write(tmp + '\n')
        output.close()

'''
利用gensim中的word2vec训练词向量
'''
def word2vec():
    print('Start...')
    rawdata='/home/shandyone/Pycharm/word2vec/zhwiki/zhwiki-segment.txt'
    modelpath='/home/shandyone/Pycharm/word2vec/model/modeldata.model'
    #vectorpath='E:\word2vec\vector'
    model=Word2Vec(LineSentence(rawdata),size=400,window=5,min_count=5,workers=multiprocessing.cpu_count())#参数说明，gensim函数库的Word2Vec的参数说明
    model.save(modelpath)
    #model.wv.save_word2vec_format(vectorpath,binary=False)
    print("Finished!")

def wordsimilarity():
    model=Word2Vec.load('/home/shandyone/Pycharm/word2vec/model/modeldata.model')
    semi=''
    try:
        semi=model.most_similar('足球'.decode('utf-8'),topn=10)#python3以上就不需要decode
    except KeyError:
        print('The word not in vocabulary!')

    #print(model[u'日本'])#打印词向量
    for term in semi:
        print('%s,%s' %(term[0],term[1]))


if __name__=='__main__':
    #dataprocess()
    #trans_seg()
    #zhwiki_segment()
    #word2vec()
    wordsimilarity()