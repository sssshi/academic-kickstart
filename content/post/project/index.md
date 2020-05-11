{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<font face=\"微软雅黑\" size=7>Using CNN to predict the rating of comment"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model principle"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##  Introduction "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As a kind of deep neural network, convolutional neural network (CNN) is most commonly used in computer vision applications.\n",
    "However, studies in recent years have found that not only in the visual aspect, but also in the text classification of CNN has a significant effect."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Word Embedding"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Word embedding as a textual representation serves the same purpose as one-hot encoding and integer encoding, but it has more advantages.Word embedding is to map each word through space and convert One hot representation into a Distributed representation, so that we can obtain a low-dimensional, dense word vector to represent each word. In this way, each word is a one-dimensional vector, and a sentence can be represented by several one-dimensional vectors, so we have a matrix to represent a sentence.Word2vec, as one of the mainstream methods of word embedding, is based on the statistical method to obtain the word vector."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## CNN"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Convolutional neural network is mainly composed of these layers: input layer, convolutional layer, Pooling layer and fully connected layer (the fully connected layer is the same as that in conventional neural network)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### convolutional layer"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The convolutional layer is the core layer for constructing the convolutional neural network. The convolution operation is actually the dot product multiplication of the convolution kernel matrix and a small matrix in the corresponding input layer. The convolution kernel slides to extract the features in the input layer according to the step length by means of weight sharing."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Pooling layer"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "NLP pool generally adopts the maximum pool, it will convolution layer vector of each channel to get maximum pool, get a scalar, so you can see simple convolution network can only be extracted to a sentence of whether there is a n - \"gramm, it doesn't get this n -\" gramm appear in statements, more can't extract to this \"gramm and the link between the 2 pet\" gramm dependencies. So there's going to be as many maxima scalars as there are in the convolution kernel, and then we're going to put those scalars together as a vector; The vectors obtained by the convolution kernel of all sizes are spliced again, so that a final one-dimensional vector is obtained, as shown in the figure above, at a glance. This allows you to transfer the final vector to the full connection layer or directly to the softmax layer for classification."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Fully connected layer"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The full connectivity layer serves to map the learned \"distributed feature representation\" into the sample tag space.It's essentially a linear transformation from one feature space to another."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data processing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data cleaning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<font face=\"微软雅黑\" size=3> In this step, I read the CSV file, deleted the data with empty comments, and sorted the training data and test data.\n",
    "    The get_label_sets function is used to read files and delete empty data.\n",
    "    The get_train_test function is used to sort the training data and test data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Length of original data:  13170073\n",
      "Length of data after preprocessing:  2638172\n",
      "Length of training data:  1846720\n",
      "Length of testing data:  791453\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from tqdm import tqdm\n",
    "from sklearn.utils import shuffle\n",
    "import math\n",
    "\n",
    "def get_label_sets():\n",
    "    rating = []\n",
    "    label_set = pd.read_csv('/Users/ssssshi/Desktop/Arlington/DM/project/boardgamegeek-reviews/bgg-13m-reviews.csv',encoding='utf-8',keep_default_na=False)\n",
    "    print(\"Length of original data: \",len(label_set))\n",
    "    label_set = label_set[label_set['comment'] != '']  # 删除\"缺陷内容\"为空的  (空格 \\u3000)\n",
    "    # print(len(label_set))\n",
    "    label_rating = label_set['rating']\n",
    "    for value in label_rating:\n",
    "        rating.append(math.ceil(value))\n",
    "    label_set['rating'] = rating\n",
    "    label_set = shuffle(label_set)\n",
    "    label_set = label_set.reset_index(drop=True)\n",
    "    label_set.to_csv('/Users/ssssshi/Desktop/Arlington/DM/project/data/data_process.csv')\n",
    "#     label_set = pd.read_csv('/Users/ssssshi/Desktop/Arlington/DM/project/data/data_process.csv',encoding='utf-8',keep_default_na=False)\n",
    "    return label_set\n",
    "\n",
    "def get_train_test(label_set):\n",
    "    len_train = int(len(label_set) * 0.7)\n",
    "    train_label_set = label_set[:len_train]\n",
    "    test_label_set = label_set[len_train-1:len(label_set)]\n",
    "    return train_label_set,test_label_set\n",
    "\n",
    "label_set = get_label_sets()\n",
    "print(\"Length of data after preprocessing: \",len(label_set))\n",
    "\n",
    "train_label_set,test_label_set = get_train_test(label_set)\n",
    "print(\"Length of training data: \",len(train_label_set))\n",
    "print(\"Length of testing data: \",len(test_label_set))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<font face=\"微软雅黑\" size=3>In this step, I remove the stop words and the non-english words in the comments.\n",
    "    The check_contain_english function is used to check whether the string is English or not.\n",
    "    The cut_sentence function is used to remove stopwords and remove the string that is not English."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Error loading stopwords: <urlopen error [SSL:\n",
      "[nltk_data]     CERTIFICATE_VERIFY_FAILED] certificate verify failed\n",
      "[nltk_data]     (_ssl.c:749)>\n"
     ]
    }
   ],
   "source": [
    "from nltk.corpus import stopwords\n",
    "import nltk\n",
    "\n",
    "nltk.download('stopwords')\n",
    "stop_words = set(stopwords.words('english'))\n",
    "\n",
    "def check_contain_english(check_str):  # 删除包含非中文ch的词语\n",
    "    for ch in check_str:\n",
    "        if (ch >= u'\\u0041' and ch<=u'\\u005a') or (ch >= u'\\u0061' and ch<=u'\\u007a'):\n",
    "            return True\n",
    "    return False\n",
    "\n",
    "def cut_sentence(sentences):\n",
    "    new_cut_sentences1 = []\n",
    "    for i in tqdm(range(len(sentences))):\n",
    "        word_list = \"\"\n",
    "#         print(sentences[i])\n",
    "        for word in sentences[i]:\n",
    "#             print(word)\n",
    "            flag = check_contain_english(word)  # 词语全是汉字\n",
    "            if flag == True and word not in stop_words:\n",
    "                word_list = word_list + word +\" \"\n",
    "        new_cut_sentences1.append(word_list)\n",
    "    return new_cut_sentences1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Contributions & Optimization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Word embedding"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<font face=\"微软雅黑\" size=3>In this step, I used word2vec to get the word vector and train the word vector model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 1846720/1846720 [03:20<00:00, 9217.54it/s]\n",
      "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/ipykernel_launcher.py:20: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Initialization done\n",
      "Complete the training\n",
      "Save success\n"
     ]
    }
   ],
   "source": [
    "from gensim.models import word2vec\n",
    "\n",
    "def word_model(sentences):\n",
    "    num_features = 200  # Word vector dimensionality\n",
    "    min_word_count = 10  # Minimum word count\n",
    "    num_workers = 16  # Number of threads to run in parallel\n",
    "    context = 10  # Context window size\n",
    "    downsampling = 1e-3  # Downsample setting for frequent words\n",
    "    print(\"Initialization done\")\n",
    "    model = word2vec.Word2Vec(sentences, workers=num_workers, \\\n",
    "                              size=num_features, min_count=min_word_count, \\\n",
    "                              window=context, sg=1, sample=downsampling)\n",
    "    print(\"Complete the training\")\n",
    "    model.init_sims(replace=True)\n",
    "    model.save(\"/Users/ssssshi/Desktop/Arlington/DM/project/model/word_model\")\n",
    "    print(\"Save success\")\n",
    "       \n",
    "train_sentences = train_label_set.comment  # \"缺陷描述\"\n",
    "cut_sentence_set = cut_sentence(train_sentences)\n",
    "train_label_set['cut_sentence'] = cut_sentence_set  # 在清洗后的原始数据加上一列, 内容为: 将清洗后的原始数据\"缺陷描述\"部分分词 (过滤了包含非汉字的词语)\n",
    "new_set = train_label_set[train_label_set.cut_sentence != ''].reset_index(drop=True) \n",
    "cut_sentence = list(new_set.cut_sentence) \n",
    "word_model(cut_sentence)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Building the CNN model and training it"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this step, I built the CNN model using keras lib and trained it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the shape of train: (1811724, 100)\n",
      "the train: [[ 6  4 10 ...  6 11 11]\n",
      " [ 0  0  0 ... 11  6  4]\n",
      " [ 0  0  0 ...  7  6  4]\n",
      " ...\n",
      " [ 3  6  7 ... 16  9 13]\n",
      " [ 0  0  0 ... 10 10  6]\n",
      " [ 0  0  0 ... 13  1 14]]\n",
      "the label: [[0. 0. 0. ... 1. 0. 0.]\n",
      " [0. 0. 0. ... 0. 0. 0.]\n",
      " [0. 0. 0. ... 0. 0. 1.]\n",
      " ...\n",
      " [0. 0. 0. ... 0. 0. 0.]\n",
      " [0. 0. 0. ... 1. 0. 0.]\n",
      " [0. 0. 0. ... 0. 0. 0.]]\n",
      "Epoch 1/20\n",
      "1811724/1811724 [==============================] - 461s 255us/step - loss: 1.9042 - accuracy: 0.2625\n",
      "Epoch 2/20\n",
      "1811724/1811724 [==============================] - 458s 253us/step - loss: 1.8476 - accuracy: 0.2790\n",
      "Epoch 3/20\n",
      "1811724/1811724 [==============================] - 458s 253us/step - loss: 1.8263 - accuracy: 0.2842\n",
      "Epoch 4/20\n",
      "1811724/1811724 [==============================] - 470s 259us/step - loss: 1.8145 - accuracy: 0.2874\n",
      "Epoch 5/20\n",
      "1811724/1811724 [==============================] - 472s 261us/step - loss: 1.8059 - accuracy: 0.2899\n",
      "Epoch 6/20\n",
      "1811724/1811724 [==============================] - 466s 257us/step - loss: 1.7999 - accuracy: 0.2913\n",
      "Epoch 7/20\n",
      "1811724/1811724 [==============================] - 469s 259us/step - loss: 1.7955 - accuracy: 0.2927\n",
      "Epoch 8/20\n",
      "1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7922 - accuracy: 0.2939\n",
      "Epoch 9/20\n",
      "1811724/1811724 [==============================] - 470s 259us/step - loss: 1.7892 - accuracy: 0.2942\n",
      "Epoch 10/20\n",
      "1811724/1811724 [==============================] - 470s 260us/step - loss: 1.7874 - accuracy: 0.2948\n",
      "Epoch 11/20\n",
      "1811724/1811724 [==============================] - 472s 260us/step - loss: 1.7852 - accuracy: 0.2955\n",
      "Epoch 12/20\n",
      "1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7833 - accuracy: 0.2959\n",
      "Epoch 13/20\n",
      "1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7819 - accuracy: 0.2961\n",
      "Epoch 14/20\n",
      "1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7806 - accuracy: 0.2965\n",
      "Epoch 15/20\n",
      "1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7792 - accuracy: 0.2969\n",
      "Epoch 16/20\n",
      "1811724/1811724 [==============================] - 469s 259us/step - loss: 1.7783 - accuracy: 0.2971\n",
      "Epoch 17/20\n",
      "1811724/1811724 [==============================] - 470s 259us/step - loss: 1.7774 - accuracy: 0.2974\n",
      "Epoch 18/20\n",
      "1811724/1811724 [==============================] - 470s 260us/step - loss: 1.7767 - accuracy: 0.2973\n",
      "Epoch 19/20\n",
      "1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7751 - accuracy: 0.2980\n",
      "Epoch 20/20\n",
      "1811724/1811724 [==============================] - 471s 260us/step - loss: 1.7749 - accuracy: 0.2979\n",
      "\n",
      "\n",
      "\n",
      "loss:  [1.9041693251199674, 1.8475539142681934, 1.826276781419597, 1.8144872036788675, 1.8059244947348765, 1.79994642072302, 1.7954695213357712, 1.7922353819333248, 1.789212237456284, 1.7873895611991106, 1.7851640862520515, 1.7833370288329953, 1.7818998512750652, 1.7805694353198025, 1.7791522975350968, 1.778344136639869, 1.7774241576255434, 1.7766880420620335, 1.7751168662869212, 1.7748565096822537]\n",
      "CNN success\n"
     ]
    }
   ],
   "source": [
    "import gensim\n",
    "import numpy as np\n",
    "import pickle\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from keras.utils import np_utils\n",
    "from keras.preprocessing.sequence import pad_sequences\n",
    "from keras import backend as K\n",
    "from keras.models import Model\n",
    "from keras.layers import *\n",
    "\n",
    "\n",
    "def make_embedding_matrix():\n",
    "    word_model = gensim.models.Word2Vec.load(\"/Users/ssssshi/Desktop/Arlington/DM/project/model/word_model\")\n",
    "    word2idx = {\"_PAD\": 0}  # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。\n",
    "    vocab_list = [(k, word_model.wv[k]) for k, v in word_model.wv.vocab.items()]  # 所有训练好的word2vec = List[('词语1',array([200维向量],float类型标记))] 8499个\n",
    "    # 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding\n",
    "    embeddings_matrix = np.zeros((len(word_model.wv.vocab.items()) + 2, word_model.vector_size))  # matrix = 8051 * 200维度  TODO:为什么+2,多两行 ?\n",
    "    for i in range(len(vocab_list)):\n",
    "        word = vocab_list[i][0]\n",
    "        word2idx[word] = i + 1  # word2id[词语]=word_id [1~8499]\n",
    "        embeddings_matrix[i + 1] = vocab_list[i][1]  # embeddings_matrix[word_id] = 词语vec\n",
    "    embeddings_matrix[len(vocab_list) + 1] = np.random.rand(1, 200)  # TODO: 什么作用 ? keras进行embedding的时候必须进行len(vocab)+1,因为Keras需要预留一个全零层， 所以+1\n",
    "    return embeddings_matrix, word2idx  # embeddings_matrix[word_id]=词语vec(=8501*200) 和 word2id[词语]=word_id [1~8499]\n",
    "\n",
    "\n",
    "def text_to_index_array(p_new_dic, p_sen):  # 文本转为索引数字模式\n",
    "    new_sentences = []\n",
    "    for sen in p_sen:   # sen = 每一个句子\n",
    "        new_sen = []  # ['表明','现象']->[231,2276], 每个句子'词语'转化为对应的'索引编号'\n",
    "        for word in sen.split():  # 句子里每一个'词语'->\n",
    "            try:\n",
    "                new_sen.append(p_new_dic[word])  # 单词转索引数字(映射表在word2idx中)\n",
    "            except:\n",
    "                new_sen.append(0)  # 索引字典里没有的词转为数字0  # TODO: 为什么选0?\n",
    "                #         for i in  range(max_len - len(new_sen)):\n",
    "                #             new_sen.append(0)\n",
    "        new_sentences.append(new_sen)  # 全部句子的索引形式\n",
    "    return np.array(new_sentences)\n",
    "\n",
    "\n",
    "def trans_value_y(y_all):  # 把4个类别变成one-hot形式\n",
    "    encoder = LabelEncoder()\n",
    "    #     y_dim = len(pd.Series(train_label1).value_counts())\n",
    "    encoded_Y1 = encoder.fit_transform(y_all)\n",
    "    # convert integers to dummy variables (one hot encoding)\n",
    "    y_train_label = np_utils.to_categorical(encoded_Y1)\n",
    "    return y_train_label, encoder\n",
    "\n",
    "# train_sentences = train_label_set.comment  # \"缺陷描述\"\n",
    "# cut_sentence_set = cut_sentence(train_sentences)\n",
    "# train_label_set['cut_sentence'] = train_cut_sentence  # 在清洗后的原始数据加上一列, 内容为: 将清洗后的原始数据\"缺陷描述\"部分分词 (过滤了包含非汉字的词语)\n",
    "# new_set = train_label_set[train_label_set.cut_sentence != ''].reset_index(drop=True)  # 删除分词后,结果cut_sentence为空的数据行\n",
    "tag = new_set.rating  # List\n",
    "cut_sentence = new_set.cut_sentence  # List\n",
    "label, encoder = trans_value_y(tag)  # 把4个类别变成one-hot形式. label=该条数据tag的onehot=len为7的list; encoder为tag与onehot对应关系\n",
    "\n",
    "embeddings_matrix, word2idx = make_embedding_matrix()  # embeddings_matrix[word_id]=词语vec(=8501*200) 和 word2id[词语]=word_id [1~8499]\n",
    "pickle.dump(embeddings_matrix, open('/Users/ssssshi/Desktop/Arlington/DM/project/embeddings_matrix.pickle', 'wb'))  # 给test.py的用\n",
    "pickle.dump(word2idx, open('/Users/ssssshi/Desktop/Arlington/DM/project/word2idx.pickle', 'wb'))\n",
    "pickle.dump(encoder, open('/Users/ssssshi/Desktop/Arlington/DM/project/encoder.pickle', 'wb'))\n",
    "\n",
    "def get_train_data(cut_sentence):\n",
    "    all_train_ids = text_to_index_array(word2idx, cut_sentence)  # 每句话一个List=List[List句子1(词语1的index,词2的index),...]\n",
    "    train_padded_seqs = pad_sequences(all_train_ids, maxlen=100)  # 每篇文章都扩充成100个词的index(不够的用index=0在前面占位)\n",
    "    # 我们需要重新整理数据集 TODO:为什么这样做\n",
    "    left_train_word_ids = [[len(word2idx)] + x[:-1] for x in all_train_ids]  # We shift the document to the right to obtain the left-side contexts. x=[1273, 7736, 3051, 1457]->[7736, 3051, 1457]+[固定的word2id长度8500]=[8500, 1273, 7736, 3051, 1457]\n",
    "    right_train_word_ids = [x[1:] + [len(word2idx)] for x in all_train_ids]  # We shift the document to the left to obtain the right-side contexts.\n",
    "    left_train_padded_seqs = pad_sequences(left_train_word_ids, maxlen=100)\n",
    "    right_train_padded_seqs = pad_sequences(right_train_word_ids, maxlen=100)\n",
    "    return train_padded_seqs,left_train_padded_seqs, right_train_padded_seqs\n",
    "\n",
    "train_padded_seqs, left_train_padded_seqs, right_train_padded_seqs = get_train_data(cut_sentence)\n",
    "\n",
    "# 词嵌入（使用预训练的词向量）\n",
    "# 神经网络的第一层，词向量层，本文使用了预训练词向量，可以把trainable那里设为False\n",
    "# 生成\n",
    "K.clear_session()  # 建模之前,清空'缓存'\n",
    "EMBEDDING_DIM = 200  # 词向量维度\n",
    "embedding_layer = Embedding(len(embeddings_matrix), # len(embeddings_matrix)=len(vocal)+1 keras进行embedding的时候必须进行len(vocab)+1\n",
    "                            EMBEDDING_DIM,  # 200\n",
    "                            weights=[embeddings_matrix],\n",
    "                            trainable=False)\n",
    "\n",
    "def create_CNN_model(train_padded_seqs, label):\n",
    "# https: // github.com / airalcorn2 / Recurrent - Convolutional - Neural - Network - Text - Classifier / blob / master / recurrent_convolutional_keras.py\n",
    "    # 参数\n",
    "    NUM_CLASSES = 10  # 模型3个分类类别\n",
    "    filter_size = [3,5]\n",
    "    # 模型输入,词向量\n",
    "    document = Input(shape=(100,), dtype=\"int32\")  # 每篇数据取100个词\n",
    "    # 构建词向量\n",
    "    doc_embedding = embedding_layer(document)\n",
    "    x = Conv1D(128, 5,activation='relu')(doc_embedding)#卷积层\n",
    "    x = MaxPooling1D(5)(x)#池化层，池化窗口为5\n",
    "    x = Flatten()(x)#把多维输入一维化\n",
    "    #全连接层\n",
    "    output = Dense(NUM_CLASSES, input_dim=128,activation=\"softmax\")(x)  # 等式(6)和(7)  softmax对应的输出为各个类别的概率,和为1;因为最后只是一个类别所以用softmax,否则对应多个类别的用sigmod. 因为输出类别为3类,softmax函数作用于pool_rnn，将其转换每个class的概率 TODO:可调参数\n",
    "    model = Model(inputs=document, outputs=output)  # Keras的函数式模型为Model，即广义的拥有输入和输出的模型，我们使用Model来初始化一个函数式模型\n",
    "    #\n",
    "    model.compile(loss='categorical_crossentropy',  # 多类的对数损失\n",
    "                  optimizer='adam',  # 目前认为adam比较好 TODO:optimizer可调为adadelta\n",
    "                  metrics=['accuracy'])\n",
    "\n",
    "    #训练模型\n",
    "    train_res = model.fit(train_padded_seqs, label,\n",
    "                batch_size=1024,  # TODO:2,4,8,16,32,64...每次用128条数据训练(整个数据集samples个数 = batch_size * n_batch)\n",
    "                epochs=20)  # 整体数据训练6轮\n",
    "                # 还可以加上 验证集 validation_data=(x_val, y_val))\n",
    "    loss = train_res.history[\"loss\"]  # 最后一次的效果,一般最好\n",
    "    print(\"\\n\\n\\nloss: \", loss)  # 打印每一次epoch的损失函数,看是否收敛(即,是否越来越小)'\n",
    "    return model\n",
    "\n",
    "# 训练模型\n",
    "print(\"the shape of train:\", train_padded_seqs.shape)\n",
    "print(\"the train:\", train_padded_seqs)\n",
    "print(\"the label:\", label)\n",
    "cnn_model = create_CNN_model(train_padded_seqs, label)\n",
    "print(\"CNN success\")\n",
    "cnn_model.save('/Users/ssssshi/Desktop/Arlington/DM/project/model/CNN.h5')\n",
    "\n",
    "#rnn_model = creat_RNN_model(train_padded_seqs,label)\n",
    "#print(\"RNN success\")\n",
    "#rnn_model.save('./RNN.h5')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Testing model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this step, the model is tested using a test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 791452/791452 [01:06<00:00, 11943.24it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "testtsest\n",
      "(791452, 10)\n",
      "[[2.9802756e-04 8.5681083e-04 3.2210606e-03 ... 3.1622094e-01\n",
      "  1.5236647e-01 5.2785616e-02]\n",
      " [1.7472118e-04 5.0946727e-04 1.8128398e-03 ... 4.1266873e-01\n",
      "  1.6228600e-01 4.1127238e-02]\n",
      " [1.5357362e-02 5.3490549e-02 6.0395591e-02 ... 6.0293365e-02\n",
      "  2.2147166e-02 1.4878695e-02]\n",
      " ...\n",
      " [7.8841858e-04 2.0540361e-03 7.1855891e-03 ... 2.6817331e-01\n",
      "  9.2804335e-02 2.5999894e-02]\n",
      " [4.5303148e-04 1.6571740e-03 6.4049466e-03 ... 1.3424359e-01\n",
      "  2.4114052e-02 3.2463965e-03]\n",
      " [1.3051563e-03 3.4474202e-03 1.0559684e-02 ... 2.1396674e-01\n",
      "  9.3526945e-02 2.4198100e-02]]\n",
      "the accuracy is:  29.13278379484795\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pickle\n",
    "from keras.models import Model, save_model, model_from_json, load_model\n",
    "import sys\n",
    "from keras.preprocessing.sequence import pad_sequences\n",
    "from keras.layers import Input,Dense,concatenate,Lambda,Conv1D, MaxPooling1D, Embedding,LSTM,Activation, AveragePooling1D,Dropout\n",
    "\n",
    "def text_to_index_array(p_new_dic, p_sen):  # 文本转为索引数字模式\n",
    "    max_len = 0\n",
    "    new_sentences = []\n",
    "    for sen in p_sen:\n",
    "        new_sen = []\n",
    "        for word in sen.split():\n",
    "            try:\n",
    "                new_sen.append(p_new_dic[word])  # 单词转索引数字\n",
    "            except:\n",
    "                new_sen.append(0)  # 索引字典里没有的词转为数字0\n",
    "        new_sentences.append(new_sen)\n",
    "\n",
    "    return np.array(new_sentences)\n",
    "\n",
    "def transform_sentence(cut_sentence, word2idx):\n",
    "    all_test_ids =text_to_index_array(word2idx,cut_sentence)\n",
    "    test_padded_seqs = pad_sequences(all_test_ids, maxlen=100)\n",
    "    left_left_word_ids = [[len(word2idx)] + x[:-1] for x in all_test_ids]\n",
    "    right_left_word_ids = [x[1:] + [len(word2idx)] for x in all_test_ids]\n",
    "    left_test_padded_seqs = pad_sequences(left_left_word_ids, maxlen=100)\n",
    "    right_test_padded_seqs = pad_sequences(right_left_word_ids, maxlen=100)\n",
    "    return test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs,\n",
    "\n",
    "def get_test_data(test_path):\n",
    "    label_set = pd.read_csv(test_path,encoding='utf-8',keep_default_na=False)\n",
    "    len_train = int(len(label_set) * 0.7)   \n",
    "    test_label_set = label_set[len_train:len(label_set)]\n",
    "    test_sentence = list(test_label_set.comment)\n",
    "#     print(test_sentence)\n",
    "    cut_sentence_set = cut_sentence(test_sentence)\n",
    "    actual = list(test_label_set.rating)\n",
    "    embeddings_matrix = pickle.load(open('/Users/ssssshi/Desktop/Arlington/DM/project/embeddings_matrix.pickle', 'rb'))\n",
    "    word2idx = pickle.load(open('/Users/ssssshi/Desktop/Arlington/DM/project/word2idx.pickle', 'rb'))\n",
    "    test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs = transform_sentence(cut_sentence_set, word2idx )\n",
    "    return test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs,actual\n",
    "\n",
    "def invers_y(actual, y_value,encoder, result_path):  #这个函数什么意思\n",
    "    list1 = []\n",
    "    print(y_value.shape)\n",
    "    print(y_value)\n",
    "    correct = 0\n",
    "    for i in range(y_value.shape[0]):\n",
    "        index = np.where(y_value[i] == np.max(y_value[i]))[0][0]\n",
    "        value = encoder.classes_[index]\n",
    "        if value == actual[i]:\n",
    "            correct += 1\n",
    "    score = correct / float(len(actual)) * 100.0\n",
    "    print(\"the accuracy is: \",score)\n",
    "    return score\n",
    "\n",
    "def accuracy_metric(actual, predicted):\n",
    "    correct = 0\n",
    "    for i in range(len(actual)):\n",
    "        if actual[i] == predicted[i]:\n",
    "            correct += 1\n",
    "    return correct / float(len(actual)) * 100.0\n",
    "\n",
    "test_path = '/Users/ssssshi/Desktop/Arlington/DM/project/data/data_process.csv'\n",
    "result_path = '/Users/ssssshi/Desktop/Arlington/DM/project/data/result.txt'\n",
    "test_padded_seqs, left_test_padded_seqs, right_test_padded_seqs, actual = get_test_data(test_path)\n",
    "print('testtsest')\n",
    "encoder = pickle.load(open('/Users/ssssshi/Desktop/Arlington/DM/project/encoder.pickle', 'rb'))\n",
    "model = load_model('/Users/ssssshi/Desktop/Arlington/DM/project/model/CNN.h5')\n",
    "y_pre = model.predict([test_padded_seqs])\n",
    "y_result = invers_y(actual, y_pre, encoder, result_path)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Challenge"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "a. Too much data and the model runs slowly"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Resolution: Use some data to build a preliminary model, and then use all the data to evaluate the performance of the model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "b.Score is continuous data, can't do forecast work."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Resolution:The method of rounding is adopted to discretize the fractions, and finally it becomes a 10 classification model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Hyper parameter tuning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "a. convolution kernel size = 5"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "b.the window of pooling = 5"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "c.Word vector dimension = 200"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# References "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Mei-Rong W . Text Classification Algorithm Based on Convolution Neural Network[J]. Journal of Jiamusi University(Natural ence Edition), 2018."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "py36",
   "language": "python",
   "name": "py36"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
