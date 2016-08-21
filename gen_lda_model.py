#!/usr/sbin/python3
#-*- encoding:utf-8 -*-

from gensim import corpora,models,similarities,utils
import jieba
import jieba.posseg as pseg

def etl(s): 
    punct = set(u''':!),.: ;?]}¢'%#="、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')
    # 对str/unicode
    filterpunt = lambda s: ''.join(filter(lambda x: x not in punct, s))
    # 对list
    # filterpuntl = lambda l: list(filter(lambda x: x not in punct, l))
    return filterpunt(s)

num_topics = 5
corpus_path = './data/in/all.txt'
user_dic_path = './data/in/user_dic.txt'
dict_path = './data/out/all.dic'
model_path = './data/out/allTFIDF.mdl'
model_index_path = './data/out/allTFIDF.idx'
topic_model_path = './data/out/allLDA50Topic.mdl'
topic_model_index_path = './data/out/allLDA50Topic.idx'

#载入自定义的词典，主要有一些计算机词汇和汽车型号的词汇
jieba.load_userdict(user_dic_path) 

#定义原始语料集合
train_set=[]
with open(corpus_path) as f:
    lines=f.readlines()
    for line in lines:
        title = (line.lower()).split("\t")[1]
        paragraphs = (line.lower()).split("\t")[2:]
        content = ''.join(paragraphs) + title
        #切词，etl函数用于去掉无用的符号，cut_all表示非全切分
        word_list = filter(lambda x: len(x) > 0, map(etl, jieba.lcut(content,cut_all=False)))
        train_set.append(list(word_list))

#生成字典
dictionary = corpora.Dictionary(train_set)
#去除极低频的杂质词
dictionary.filter_extremes(no_below=1,no_above=1,keep_n=None)
#将词典保存下来，方便后续使用

dictionary.save(dict_path)

doc_bow = [dictionary.doc2bow(text) for text in train_set]


#使用数字语料生成TFIDF模型
tfidfModel = models.TfidfModel(doc_bow)
#存储tfidfModel
tfidfModel.save(model_path)


#把全部语料向量化成TFIDF模式，这个tfidfModel可以传入二维数组
tfidfVectors = tfidfModel[doc_bow]
#建立索引并保存
indexTfidf = similarities.MatrixSimilarity(tfidfVectors)
indexTfidf.save(model_index_path)


#通过TFIDF向量生成LDA模型，id2word表示编号的对应词典，num_topics表示主题数，我们这里设定的50，主题太多时间受不了。
lda = models.LdaModel(tfidfVectors, id2word=dictionary, num_topics=num_topics)
#把模型保存下来
lda.save(topic_model_path)
#把所有TFIDF向量变成LDA的向量
doc_bow_lda = lda[tfidfVectors]
#建立索引，把LDA数据保存下来
indexLDA = similarities.MatrixSimilarity(doc_bow_lda)
indexLDA.save(topic_model_index_path)

def prettifyVec(bow, dictionary):
    ret = []
    for tup in bow:
        index, weight = tup
        word = dictionary[index]
        ret.append((word,weight))
    return ret

#载入字典
dictionary = corpora.Dictionary.load(dict_path)
#载入TFIDF模型和索引
tfidfModel = models.TfidfModel.load(model_path)
indexTfidf = similarities.MatrixSimilarity.load(model_index_path)
#载入LDA模型和索引
ldaModel = models.LdaModel.load(topic_model_path)
indexLDA = similarities.MatrixSimilarity.load(topic_model_index_path)

for topicid in range(0,num_topics):
    ldaModel.print_topic(topicid)
    print(ldaModel.get_topic_terms(topicid), dictionary)


#doc就是测试数据，先切词
for line in lines:
	# query = '今日力推：Android 中文字体压缩神器 / 详解 AIDL in、out、inout / React Native 实现的 GMTC 客户端 / iOS 可用的设备信息工具库'
    doc = line
    print(line)
    doc_bow = dictionary.doc2bow(filter(lambda x: len(x)>0,map(etl,jieba.cut(doc,cut_all=False))))
    #使用TFIDF模型向量化
    tfidfvect = tfidfModel[doc_bow]
    #然后LDA向量化，因为我们训练时的LDA是在TFIDF基础上做的，所以用itidfvect再向量化一次
    ldavec = ldaModel[tfidfvect]
    #TFIDF相似性
    simstfidf = indexTfidf[tfidfvect]
    #LDA相似性
    simlda = indexLDA[ldavec]
    print(prettifyVec(doc_bow, dictionary))
    print(prettifyVec(tfidfvect, dictionary))
    print(prettifyVec(ldavec, dictionary))
    print('\n\n')

