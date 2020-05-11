
<font face="微软雅黑" size=7>Using CNN to predict the rating of comment

<font face="微软雅黑" size=6>1 Model principle

<font face="微软雅黑" size=5>1.1 Introduction 

<font face="微软雅黑" size=3>As a kind of deep neural network, convolutional neural network (CNN) is most commonly used in computer vision applications.
However, studies in recent years have found that not only in the visual aspect, but also in the text classification of CNN has a significant effect.

<font face="微软雅黑" size=5>1.2 Word Embedding

<font face="微软雅黑" size=3>Word embedding as a textual representation serves the same purpose as one-hot encoding and integer encoding, but it has more advantages.Word embedding is to map each word through space and convert One hot representation into a Distributed representation, so that we can obtain a low-dimensional, dense word vector to represent each word. In this way, each word is a one-dimensional vector, and a sentence can be represented by several one-dimensional vectors, so we have a matrix to represent a sentence.Word2vec, as one of the mainstream methods of word embedding, is based on the statistical method to obtain the word vector.

<font face="微软雅黑" size=5>1.3 CNN

<font face="微软雅黑" size=3>Convolutional neural network is mainly composed of these layers: input layer, convolutional layer, Pooling layer and fully connected layer (the fully connected layer is the same as that in conventional neural network).

<font face="微软雅黑" size=4>1.3.1 convolutional layer

<font face="微软雅黑" size=3>The convolutional layer is the core layer for constructing the convolutional neural network. The convolution operation is actually the dot product multiplication of the convolution kernel matrix and a small matrix in the corresponding input layer. The convolution kernel slides to extract the features in the input layer according to the step length by means of weight sharing.

<font face="微软雅黑" size=4>1.3.2 Pooling layer

<font face="微软雅黑" size=3>NLP pool generally adopts the maximum pool, it will convolution layer vector of each channel to get maximum pool, get a scalar, so you can see simple convolution network can only be extracted to a sentence of whether there is a n - "gramm, it doesn't get this n -" gramm appear in statements, more can't extract to this "gramm and the link between the 2 pet" gramm dependencies. So there's going to be as many maxima scalars as there are in the convolution kernel, and then we're going to put those scalars together as a vector; The vectors obtained by the convolution kernel of all sizes are spliced again, so that a final one-dimensional vector is obtained, as shown in the figure above, at a glance. This allows you to transfer the final vector to the full connection layer or directly to the softmax layer for classification.

<font face="微软雅黑" size=4>1.3.3 Fully connected layer

<font face="微软雅黑" size=3>The full connectivity layer serves to map the learned "distributed feature representation" into the sample tag space.It's essentially a linear transformation from one feature space to another.

<font face="微软雅黑" size=6>2 Data processing

<font face="微软雅黑" size=5>2.1 Data cleaning

<font face="微软雅黑" size=3> In this step, I read the CSV file, deleted the data with empty comments, and sorted the training data and test data.
    The get_label_sets function is used to read files and delete empty data.
    The get_train_test function is used to sort the training data and test data.


```python
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
import math

def get_label_sets():
    rating = []
    label_set = pd.read_csv('/Users/ssssshi/Desktop/Arlington/DM/project/boardgamegeek-reviews/bgg-13m-reviews.csv',encoding='utf-8',keep_default_na=False)
    print("Length of original data: ",len(label_set))
    label_set = label_set[label_set['comment'] != '']  # 删除"缺陷内容"为空的  (空格 \u3000)
    # print(len(label_set))
    label_rating = label_set['rating']
    for value in label_rating:
        rating.append(math.ceil(value))
    label_set['rating'] = rating
    label_set = shuffle(label_set)
    label_set = label_set.reset_index(drop=True)
    label_set.to_csv('/Users/ssssshi/Desktop/Arlington/DM/project/data/data_process.csv')
#     label_set = pd.read_csv('/Users/ssssshi/Desktop/Arlington/DM/project/data/data_process.csv',encoding='utf-8',keep_default_na=False)
    return label_set

def get_train_test(label_set):
    len_train = int(len(label_set) * 0.7)
    train_label_set = label_set[:len_train]
    test_label_set = label_set[len_train-1:len(label_set)]
    return train_label_set,test_label_set

label_set = get_label_sets()
print("Length of data after preprocessing: ",len(label_set))

train_label_set,test_label_set = get_train_test(label_set)
print("Length of training data: ",len(train_label_set))
print("Length of testing data: ",len(test_label_set))
```

    Length of original data:  13170073
    Length of data after preprocessing:  2638172
    Length of training data:  1846720
    Length of testing data:  791453


<font face="微软雅黑" size=3>In this step, I remove the stop words and the non-english words in the comments.
    The check_contain_english function is used to check whether the string is English or not.
    The cut_sentence function is used to remove stopwords and remove the string that is not English.


```python
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def check_contain_english(check_str):  # 删除包含非中文ch的词语
    for ch in check_str:
        if (ch >= u'\u0041' and ch<=u'\u005a') or (ch >= u'\u0061' and ch<=u'\u007a'):
            return True
    return False

def cut_sentence(sentences):
    new_cut_sentences1 = []
    for i in tqdm(range(len(sentences))):
        word_list = ""
#         print(sentences[i])
        for word in sentences[i]:
#             print(word)
            flag = check_contain_english(word)  # 词语全是汉字
            if flag == True and word not in stop_words:
                word_list = word_list + word +" "
        new_cut_sentences1.append(word_list)
    return new_cut_sentences1
```

    [nltk_data] Error loading stopwords: <urlopen error [SSL:
    [nltk_data]     CERTIFICATE_VERIFY_FAILED] certificate verify failed
    [nltk_data]     (_ssl.c:749)>


<font face="微软雅黑" size=6>3 Contributions & Optimization

<font face="微软雅黑" size=5>3.1 Word embedding

<font face="微软雅黑" size=3>In this step, I used word2vec to get the word vector and train the word vector model.


```python
from gensim.models import word2vec

def word_model(sentences):
    num_features = 200  # Word vector dimensionality
    min_word_count = 10  # Minimum word count
    num_workers = 16  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    print("Initialization done")
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                              size=num_features, min_count=min_word_count, \
                              window=context, sg=1, sample=downsampling)
    print("Complete the training")
    model.init_sims(replace=True)
    model.save("/Users/ssssshi/Desktop/Arlington/DM/project/model/word_model")
    print("Save success")
       
train_sentences = train_label_set.comment  # "缺陷描述"
cut_sentence_set = cut_sentence(train_sentences)
train_label_set['cut_sentence'] = cut_sentence_set  # 在清洗后的原始数据加上一列, 内容为: 将清洗后的原始数据"缺陷描述"部分分词 (过滤了包含非汉字的词语)
new_set = train_label_set[train_label_set.cut_sentence != ''].reset_index(drop=True) 
cut_sentence = list(new_set.cut_sentence) 
word_model(cut_sentence)
```

    100%|██████████| 1846720/1846720 [03:20<00:00, 9217.54it/s]
    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/ipykernel_launcher.py:20: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy


    Initialization done
    Complete the training
    Save success


<font face="微软雅黑" size=5>3.2 Building the CNN model and training it

In this step, I built the CNN model using keras lib and trained it


```python
import gensim
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import Model
from keras.layers import *


def make_embedding_matrix():
    word_model = gensim.models.Word2Vec.load("/Users/ssssshi/Desktop/Arlington/DM/project/model/word_model")
    word2idx = {"_PAD": 0}  # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
    vocab_list = [(k, word_model.wv[k]) for k, v in word_model.wv.vocab.items()]  # 所有训练好的word2vec = List[('词语1',array([200维向量],float类型标记))] 8499个
    # 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
    embeddings_matrix = np.zeros((len(word_model.wv.vocab.items()) + 2, word_model.vector_size))  # matrix = 8051 * 200维度  TODO:为什么+2,多两行 ?
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1  # word2id[词语]=word_id [1~8499]
        embeddings_matrix[i + 1] = vocab_list[i][1]  # embeddings_matrix[word_id] = 词语vec
    embeddings_matrix[len(vocab_list) + 1] = np.random.rand(1, 200)  # TODO: 什么作用 ? keras进行embedding的时候必须进行len(vocab)+1,因为Keras需要预留一个全零层， 所以+1
    return embeddings_matrix, word2idx  # embeddings_matrix[word_id]=词语vec(=8501*200) 和 word2id[词语]=word_id [1~8499]


def text_to_index_array(p_new_dic, p_sen):  # 文本转为索引数字模式
    new_sentences = []
    for sen in p_sen:   # sen = 每一个句子
        new_sen = []  # ['表明','现象']->[231,2276], 每个句子'词语'转化为对应的'索引编号'
        for word in sen.split():  # 句子里每一个'词语'->
            try:
                new_sen.append(p_new_dic[word])  # 单词转索引数字(映射表在word2idx中)
            except:
                new_sen.append(0)  # 索引字典里没有的词转为数字0  # TODO: 为什么选0?
                #         for i in  range(max_len - len(new_sen)):
                #             new_sen.append(0)
        new_sentences.append(new_sen)  # 全部句子的索引形式
    return np.array(new_sentences)


def trans_value_y(y_all):  # 把4个类别变成one-hot形式
    encoder = LabelEncoder()
    #     y_dim = len(pd.Series(train_label1).value_counts())
    encoded_Y1 = encoder.fit_transform(y_all)
    # convert integers to dummy variables (one hot encoding)
    y_train_label = np_utils.to_categorical(encoded_Y1)
    return y_train_label, encoder

# train_sentences = train_label_set.comment  # "缺陷描述"
# cut_sentence_set = cut_sentence(train_sentences)
# train_label_set['cut_sentence'] = train_cut_sentence  # 在清洗后的原始数据加上一列, 内容为: 将清洗后的原始数据"缺陷描述"部分分词 (过滤了包含非汉字的词语)
# new_set = train_label_set[train_label_set.cut_sentence != ''].reset_index(drop=True)  # 删除分词后,结果cut_sentence为空的数据行
tag = new_set.rating  # List
cut_sentence = new_set.cut_sentence  # List
label, encoder = trans_value_y(tag)  # 把4个类别变成one-hot形式. label=该条数据tag的onehot=len为7的list; encoder为tag与onehot对应关系

embeddings_matrix, word2idx = make_embedding_matrix()  # embeddings_matrix[word_id]=词语vec(=8501*200) 和 word2id[词语]=word_id [1~8499]
pickle.dump(embeddings_matrix, open('/Users/ssssshi/Desktop/Arlington/DM/project/embeddings_matrix.pickle', 'wb'))  # 给test.py的用
pickle.dump(word2idx, open('/Users/ssssshi/Desktop/Arlington/DM/project/word2idx.pickle', 'wb'))
pickle.dump(encoder, open('/Users/ssssshi/Desktop/Arlington/DM/project/encoder.pickle', 'wb'))

def get_train_data(cut_sentence):
    all_train_ids = text_to_index_array(word2idx, cut_sentence)  # 每句话一个List=List[List句子1(词语1的index,词2的index),...]
    train_padded_seqs = pad_sequences(all_train_ids, maxlen=100)  # 每篇文章都扩充成100个词的index(不够的用index=0在前面占位)
    # 我们需要重新整理数据集 TODO:为什么这样做
    left_train_word_ids = [[len(word2idx)] + x[:-1] for x in all_train_ids]  # We shift the document to the right to obtain the left-side contexts. x=[1273, 7736, 3051, 1457]->[7736, 3051, 1457]+[固定的word2id长度8500]=[8500, 1273, 7736, 3051, 1457]
    right_train_word_ids = [x[1:] + [len(word2idx)] for x in all_train_ids]  # We shift the document to the left to obtain the right-side contexts.
    left_train_padded_seqs = pad_sequences(left_train_word_ids, maxlen=100)
    right_train_padded_seqs = pad_sequences(right_train_word_ids, maxlen=100)
    return train_padded_seqs,left_train_padded_seqs, right_train_padded_seqs

train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs = get_train_data(cut_sentence)

# 词嵌入（使用预训练的词向量）
# 神经网络的第一层，词向量层，本文使用了预训练词向量，可以把trainable那里设为False
# 生成
K.clear_session()  # 建模之前,清空'缓存'
EMBEDDING_DIM = 200  # 词向量维度
embedding_layer = Embedding(len(embeddings_matrix), # len(embeddings_matrix)=len(vocal)+1 keras进行embedding的时候必须进行len(vocab)+1
                            EMBEDDING_DIM,  # 200
                            weights=[embeddings_matrix],
                            trainable=False)

def create_CNN_model(train_padded_seqs, label):
# https: // github.com / airalcorn2 / Recurrent - Convolutional - Neural - Network - Text - Classifier / blob / master / recurrent_convolutional_keras.py
    # 参数
    NUM_CLASSES = 10  # 模型3个分类类别
    filter_size = [3,5]
    # 模型输入,词向量
    document = Input(shape=(100,), dtype="int32")  # 每篇数据取100个词
    # 构建词向量
    doc_embedding = embedding_layer(document)
    x = Conv1D(128, 5,activation='relu')(doc_embedding)#卷积层
    x = MaxPooling1D(5)(x)#池化层，池化窗口为5
    x = Flatten()(x)#把多维输入一维化
    #全连接层
    output = Dense(NUM_CLASSES, input_dim=128,activation="softmax")(x)  # 等式(6)和(7)  softmax对应的输出为各个类别的概率,和为1;因为最后只是一个类别所以用softmax,否则对应多个类别的用sigmod. 因为输出类别为3类,softmax函数作用于pool_rnn，将其转换每个class的概率 TODO:可调参数
    model = Model(inputs=document, outputs=output)  # Keras的函数式模型为Model，即广义的拥有输入和输出的模型，我们使用Model来初始化一个函数式模型
    #
    model.compile(loss='categorical_crossentropy',  # 多类的对数损失
                  optimizer='adam',  # 目前认为adam比较好 TODO:optimizer可调为adadelta
                  metrics=['accuracy'])

    #训练模型
    train_res = model.fit(train_padded_seqs, label,
                batch_size=1024,  # TODO:2,4,8,16,32,64...每次用128条数据训练(整个数据集samples个数 = batch_size * n_batch)
                epochs=20)  # 整体数据训练6轮
                # 还可以加上 验证集 validation_data=(x_val, y_val))
    loss = train_res.history["loss"]  # 最后一次的效果,一般最好
    print("\n\n\nloss: ", loss)  # 打印每一次epoch的损失函数,看是否收敛(即,是否越来越小)'
    return model

# 训练模型
print("the shape of train:", train_padded_seqs.shape)
print("the train:", train_padded_seqs)
print("the label:", label)
cnn_model = create_CNN_model(train_padded_seqs, label)
print("CNN success")
cnn_model.save('/Users/ssssshi/Desktop/Arlington/DM/project/model/CNN.h5')

#rnn_model = creat_RNN_model(train_padded_seqs,label)
#print("RNN success")
#rnn_model.save('./RNN.h5')
```

    Using TensorFlow backend.


    the shape of train: (1811724, 100)
    the train: [[ 6  4 10 ...  6 11 11]
     [ 0  0  0 ... 11  6  4]
     [ 0  0  0 ...  7  6  4]
     ...
     [ 3  6  7 ... 16  9 13]
     [ 0  0  0 ... 10 10  6]
     [ 0  0  0 ... 13  1 14]]
    the label: [[0. 0. 0. ... 1. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 1.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 1. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]
    Epoch 1/20
    1811724/1811724 [==============================] - 461s 255us/step - loss: 1.9042 - accuracy: 0.2625
    Epoch 2/20
    1811724/1811724 [==============================] - 458s 253us/step - loss: 1.8476 - accuracy: 0.2790
    Epoch 3/20
    1811724/1811724 [==============================] - 458s 253us/step - loss: 1.8263 - accuracy: 0.2842
    Epoch 4/20
    1811724/1811724 [==============================] - 470s 259us/step - loss: 1.8145 - accuracy: 0.2874
    Epoch 5/20
    1811724/1811724 [==============================] - 472s 261us/step - loss: 1.8059 - accuracy: 0.2899
    Epoch 6/20
    1811724/1811724 [==============================] - 466s 257us/step - loss: 1.7999 - accuracy: 0.2913
    Epoch 7/20
    1811724/1811724 [==============================] - 469s 259us/step - loss: 1.7955 - accuracy: 0.2927
    Epoch 8/20
    1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7922 - accuracy: 0.2939
    Epoch 9/20
    1811724/1811724 [==============================] - 470s 259us/step - loss: 1.7892 - accuracy: 0.2942
    Epoch 10/20
    1811724/1811724 [==============================] - 470s 260us/step - loss: 1.7874 - accuracy: 0.2948
    Epoch 11/20
    1811724/1811724 [==============================] - 472s 260us/step - loss: 1.7852 - accuracy: 0.2955
    Epoch 12/20
    1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7833 - accuracy: 0.2959
    Epoch 13/20
    1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7819 - accuracy: 0.2961
    Epoch 14/20
    1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7806 - accuracy: 0.2965
    Epoch 15/20
    1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7792 - accuracy: 0.2969
    Epoch 16/20
    1811724/1811724 [==============================] - 469s 259us/step - loss: 1.7783 - accuracy: 0.2971
    Epoch 17/20
    1811724/1811724 [==============================] - 470s 259us/step - loss: 1.7774 - accuracy: 0.2974
    Epoch 18/20
    1811724/1811724 [==============================] - 470s 260us/step - loss: 1.7767 - accuracy: 0.2973
    Epoch 19/20
    1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7751 - accuracy: 0.2980
    Epoch 20/20
    1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7749 - accuracy: 0.2979
    
    
    
    loss:  [1.9041693251199674, 1.8475539142681934, 1.826276781419597, 1.8144872036788675, 1.8059244947348765, 1.79994642072302, 1.7954695213357712, 1.7922353819333248, 1.789212237456284, 1.7873895611991106, 1.7851640862520515, 1.7833370288329953, 1.7818998512750652, 1.7805694353198025, 1.7791522975350968, 1.778344136639869, 1.7774241576255434, 1.7766880420620335, 1.7751168662869212, 1.7748565096822537]
    CNN success


<font face="微软雅黑" size=5>3.3 Testing model

In this step, the model is tested using a test set


```python
import pandas as pd
import numpy as np
import pickle
from keras.models import Model, save_model, model_from_json, load_model
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input,Dense,concatenate,Lambda,Conv1D, MaxPooling1D, Embedding,LSTM,Activation, AveragePooling1D,Dropout

def text_to_index_array(p_new_dic, p_sen):  # 文本转为索引数字模式
    max_len = 0
    new_sentences = []
    for sen in p_sen:
        new_sen = []
        for word in sen.split():
            try:
                new_sen.append(p_new_dic[word])  # 单词转索引数字
            except:
                new_sen.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(new_sen)

    return np.array(new_sentences)

def transform_sentence(cut_sentence, word2idx):
    all_test_ids =text_to_index_array(word2idx,cut_sentence)
    test_padded_seqs = pad_sequences(all_test_ids, maxlen=100)
    left_left_word_ids = [[len(word2idx)] + x[:-1] for x in all_test_ids]
    right_left_word_ids = [x[1:] + [len(word2idx)] for x in all_test_ids]
    left_test_padded_seqs = pad_sequences(left_left_word_ids, maxlen=100)
    right_test_padded_seqs = pad_sequences(right_left_word_ids, maxlen=100)
    return test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs,

def get_test_data(test_path):
    label_set = pd.read_csv(test_path,encoding='utf-8',keep_default_na=False)
    len_train = int(len(label_set) * 0.7)   
    test_label_set = label_set[len_train:len(label_set)]
    test_sentence = list(test_label_set.comment)
#     print(test_sentence)
    cut_sentence_set = cut_sentence(test_sentence)
    actual = list(test_label_set.rating)
    embeddings_matrix = pickle.load(open('/Users/ssssshi/Desktop/Arlington/DM/project/embeddings_matrix.pickle', 'rb'))
    word2idx = pickle.load(open('/Users/ssssshi/Desktop/Arlington/DM/project/word2idx.pickle', 'rb'))
    test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs = transform_sentence(cut_sentence_set, word2idx )
    return test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs,actual

def invers_y(actual, y_value,encoder, result_path):  #这个函数什么意思
    list1 = []
    print(y_value.shape)
    print(y_value)
    correct = 0
    for i in range(y_value.shape[0]):
        index = np.where(y_value[i] == np.max(y_value[i]))[0][0]
        value = encoder.classes_[index]
        if value == actual[i]:
            correct += 1
    score = correct / float(len(actual)) * 100.0
    print("the accuracy is: ",score)
    return score

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

test_path = '/Users/ssssshi/Desktop/Arlington/DM/project/data/data_process.csv'
result_path = '/Users/ssssshi/Desktop/Arlington/DM/project/data/result.txt'
test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs, actual = get_test_data(test_path)
print('testtsest')
encoder = pickle.load(open('/Users/ssssshi/Desktop/Arlington/DM/project/encoder.pickle', 'rb'))
model = load_model('/Users/ssssshi/Desktop/Arlington/DM/project/model/CNN.h5')
y_pre = model.predict([test_padded_seqs])
y_result = invers_y(actual, y_pre, encoder, result_path)
```

    100%|██████████| 791452/791452 [01:06<00:00, 11943.24it/s]


    testtsest
    (791452, 10)
    [[2.9802756e-04 8.5681083e-04 3.2210606e-03 ... 3.1622094e-01
      1.5236647e-01 5.2785616e-02]
     [1.7472118e-04 5.0946727e-04 1.8128398e-03 ... 4.1266873e-01
      1.6228600e-01 4.1127238e-02]
     [1.5357362e-02 5.3490549e-02 6.0395591e-02 ... 6.0293365e-02
      2.2147166e-02 1.4878695e-02]
     ...
     [7.8841858e-04 2.0540361e-03 7.1855891e-03 ... 2.6817331e-01
      9.2804335e-02 2.5999894e-02]
     [4.5303148e-04 1.6571740e-03 6.4049466e-03 ... 1.3424359e-01
      2.4114052e-02 3.2463965e-03]
     [1.3051563e-03 3.4474202e-03 1.0559684e-02 ... 2.1396674e-01
      9.3526945e-02 2.4198100e-02]]
    the accuracy is:  29.13278379484795


<font face="微软雅黑" size=6>4 Challenge

<font face="微软雅黑" size=3>a. Too much data and the model runs slowly

<font face="微软雅黑" size=3>Resolution: Use some data to build a preliminary model, and then use all the data to evaluate the performance of the model.

<font face="微软雅黑" size=3>b.Score is continuous data, can't do forecast work.

<font face="微软雅黑" size=3>Resolution:The method of rounding is adopted to discretize the fractions, and finally it becomes a 10 classification model.

<font face="微软雅黑" size=6>5 Hyper parameter tuning

<font face="微软雅黑" size=3>a. convolution kernel size = 5

<font face="微软雅黑" size=3>b.the window of pooling = 5

<font face="微软雅黑" size=3>c.Word vector dimension = 200

<font face="微软雅黑" size=6>6 References 

<font face="微软雅黑" size=3>Mei-Rong W . Text Classification Algorithm Based on Convolution Neural Network[J]. Journal of Jiamusi University(Natural ence Edition), 2018.


```python

```
