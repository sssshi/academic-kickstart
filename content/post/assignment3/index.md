
<font face="微软雅黑" size=5> a.Generate data pairs & build a index vocabulary


```python
import glob
from random import randrange
from nltk.corpus import stopwords
import nltk
import re
from math import log
import numpy as np

txt_negfile = glob.glob('/Users/ssssshi/Desktop/Arlington/DM/aclImdb/train/neg/*.txt')
txt_posfile = glob.glob('/Users/ssssshi/Desktop/Arlington/DM/aclImdb/train/pos/*.txt')

test_negfile = glob.glob('/Users/ssssshi/Desktop/Arlington/DM/aclImdb/test/neg/*.txt')
test_posfile = glob.glob('/Users/ssssshi/Desktop/Arlington/DM/aclImdb/test/pos/*.txt')

all_content = []
all_neg_content = []
all_pos_content = []
test_content = []


for filename in txt_negfile:
    with open(filename, 'r') as txt_filtxt:
        buf1 = txt_filtxt.readlines()
        for s in buf1:
            content = [s, "neg"]
            all_content.append(content)

for filename in txt_posfile:
    with open(filename, 'r') as txt_posfile:
        buf2 = txt_posfile.readlines()
        for s in buf2:
            content = [s, "pos"]
            all_content.append(content)

for filename in test_negfile:
    with open(filename, 'r') as txt_filtxt:
        buf1 = txt_filtxt.readlines()
        for s in buf1:
            content = [s, "neg"]
            test_content.append(content)

for filename in test_posfile:
    with open(filename, 'r') as txt_filtxt:
        buf1 = txt_filtxt.readlines()
        for s in buf1:
            content = [s, "pos"]
            test_content.append(content)


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub('\[[^]]*\]', '', string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"can\'t", "can not", string)
    string = re.sub(r"cannot", "can not ", string)
    string = re.sub(r"what\'s", "what is", string)
    string = re.sub(r"What\'s", "what is", string)
    string = re.sub(r"\'ve ", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"i\'m", "i am ", string)
    string = re.sub(r"I\'m", "i am ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d", " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r" e mail ", " email ", string)
    string = re.sub(r" e \- mail ", " email ", string)
    string = re.sub(r" e\-mail ", " email ", string)
    string = re.sub(r"&", " and ", string)
    string = re.sub(r"\|", " or ", string)
    string = re.sub(r"=", " equal ", string)
    string = re.sub(r"\+", " plus ", string)
    string = re.sub(r"\$", " dollar ", string)
    return string

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_content(all_content):
    for i in range(0, len(all_content)):
        clean_content = clean_str(all_content[i][0])
        words = clean_content.split()
        doc_words = []
        for word in words:
            if word not in stop_words:
                doc_words.append(word)
        doc_str = ' '.join(doc_words).strip()
        all_content[i] = [doc_str, all_content[i][1]]
    return all_content

all_content = clean_content(all_content)
test_content = clean_content(test_content)
    
def get_index(dataset):
    i = 0
    index = {}
    for d in dataset:
        d_content = d[0]
        d_content = d_content.split()
        for word in d_content:
            if word not in index:
                index[word] = i
                i += 1
    return index

index = get_index(all_content)

print("index is:", index)
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/ssssshi/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!




<font face="微软雅黑" size=5> b.conducting five fold cross validation to get dev dataset and caculate the probability using Laplace and evaluate algorithm


```python
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def evaluate_algorithm(dataset, n_folds):
    folds = cross_validation_split(dataset, n_folds)
    scores1 = list()
    scores_m = {}
    m = set(np.random.rand(10))
    m = list(m)
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        dev_set = list()
        for row in fold:
            row_copy = list(row)
            dev_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, dev_set)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores1.append(accuracy)
        for i in range(0, len(m)):
            if m[i] not in scores_m:
                scores_m[m[i]] = []
            predicted_m = algorithm1(train_set, dev_set, m[i])
            accuracy_m = accuracy_metric(actual, predicted_m)
            scores_m[m[i]].append(accuracy_m)
    return scores1, scores_m

def algorithm(train_set, dev_test_set):
    total_count = get_count(train_set)
    index = get_index(train_set)
    predictions = list()
    for row in dev_test_set:
        con_prob = con_probability(index, total_count, row)
        output = predict(con_prob)
        predictions.append(output)
    return predictions

def algorithm1(train_set, dev_test_set, m):
    total_count = get_count(train_set)
    index = get_index(train_set)
    predictions = list()
    for row in dev_test_set:
        con_prob = m_con_probability(index, total_count, row, m)
        output = predict(con_prob)
        predictions.append(output)
    return predictions

# input train set [],["","pos"]
def get_count(dataset):
    count = {}
    neg_count = {}
    pos_count = {}
    pos_total = 0
    neg_total = 0
    index = get_index(dataset)
    for d in dataset:
        train_content = d[0]
        train_content = train_content.split()
        train_content = list(set(train_content))
        if d[1] == 'pos':
            pos_total += 1
            for str in train_content:
                str = index[str]
                if str not in pos_count:
                    pos_count[str] = 1
                else:
                    pos_count[str] += 1
        else:
            neg_total += 1
            for str in train_content:
                str = index[str]
                if str not in neg_count:
                    neg_count[str] = 1
                else:
                    neg_count[str] += 1

    count['pos'] = pos_count
    count['neg'] = neg_count
    count['pos_total'] = pos_total
    count['neg_total'] = neg_total
    return count

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def prior_probability(index, total_count, row):
    prior = 1.0
    row_content = row[0]
    row_content = row_content.split()
    for j in range(0, len(row_content)):
        if row_content[j] not in index:
            prior = 0.0
        else:
            i = index[row_content[j]]
            if i in total_count['neg'] and i in total_count['pos']:
                i_count = total_count['neg'][index[row_content[j]]] + total_count['pos'][index[row_content[j]]]
                prior = prior * (i_count / (total_count['pos_total'] + total_count['neg_total']))
            else:
                prior = 0.0
    return prior


def con_probability(index, total_count, row):
    con_pro = {'pos': 1.0, 'neg': 1.0}
    row_content = row[0]
    row_content = row_content.split()
    for j in range(0, len(row_content)):
        if row_content[j] not in index:
            continue
        else:
            i = index[row_content[j]]
            if i not in total_count['neg'] and i in total_count['pos']:
                con_pro['pos'] = float((total_count['pos'][i] + 1) / (total_count['pos_total'] + 2)) * con_pro['pos']
                con_pro['neg'] = (1 / (total_count['neg_total'] + 2)) * con_pro['neg']
            elif i in total_count['neg'] and i not in total_count['pos']:
                con_pro['neg'] = float((total_count['neg'][i] + 1) / (total_count['neg_total'] + 2)) * con_pro['neg']
                con_pro['pos'] = (1 / (total_count['pos_total'] + 2)) * con_pro['pos']
            elif i in total_count['neg'] and i in total_count['pos']:
                con_pro['pos'] = float((total_count['pos'][i]) / (total_count['pos_total'])) * con_pro['pos']
                con_pro['neg'] = float((total_count['neg'][i]) / (total_count['neg_total'])) * con_pro['neg']
            elif i not in total_count['pos'] and i not in total_count['neg']:
                continue
    # print(con_pro['pos'])
    # print(con_pro['neg'])
    if con_pro['pos'] != 0.0:
        con_pro['pos'] = log(con_pro['pos'])
    if con_pro['neg'] != 0.0:
        con_pro['neg'] = log(con_pro['neg'])
    return con_pro

def predict(con_prob):
    if con_prob['pos'] >= con_prob['neg']:
        return 'pos'
    else:
        return 'neg'
    
def m_con_probability(index, total_count, row, m):
    con_pro = {'pos': 1.0, 'neg': 1.0}
    row_content = row[0]
    row_content = row_content.split()
    for j in range(0, len(row_content)):
        if row_content[j] not in index:
            continue
        else:
            i = index[row_content[j]]
            if i not in total_count['neg'] and i in total_count['pos']:
                con_pro['pos'] = float((total_count['pos'][i] + float(m)*0.5) / (total_count['pos_total'] + m)) * con_pro['pos']
                con_pro['neg'] = (float(m)*0.5 / (total_count['neg_total'] + m)) * con_pro['neg']
            elif i in total_count['neg'] and i not in total_count['pos']:
                con_pro['neg'] = float((total_count['neg'][i] + float(m)*0.5) / (total_count['neg_total'] + m)) * con_pro['neg']
                con_pro['pos'] = (float(m)*0.5 / (total_count['pos_total'] + m)) * con_pro['pos']
            elif i in total_count['neg'] and i in total_count['pos']:
                con_pro['pos'] = float((total_count['pos'][i]) / (total_count['pos_total'])) * con_pro['pos']
                con_pro['neg'] = float((total_count['neg'][i]) / (total_count['neg_total'])) * con_pro['neg']
            elif i not in total_count['pos'] and i not in total_count['neg']:
                continue
    # print(con_pro['pos'])
    # print(con_pro['neg'])
    if con_pro['pos'] != 0.0:
        con_pro['pos'] = log(con_pro['pos'])
    if con_pro['neg'] != 0.0:
        con_pro['neg'] = log(con_pro['neg'])
    return con_pro

total_count = get_count(all_content)
scores,scores_m = evaluate_algorithm(all_content,5)
print("the accuracy of algorithm using Laplace is:",scores)
print("the accuracy of algorithm using m estimate:", scores_m)
```

    the accuracy of algorithm using Laplace is: [79.42, 79.75999999999999, 79.34, 78.58000000000001, 79.44]
    the accuracy of algorithm using m estimate: {0.03531496822224778: [78.18, 78.53999999999999, 78.42, 77.24, 78.2], 0.6487382510392488: [79.10000000000001, 79.75999999999999, 79.42, 78.34, 79.46], 0.9801648133343199: [79.24, 79.84, 79.56, 78.4, 79.58], 0.22374901274725634: [78.64, 79.17999999999999, 79.06, 77.96, 79.24], 0.6714303811039863: [79.14, 79.75999999999999, 79.4, 78.4, 79.5], 0.6898912807183852: [79.17999999999999, 79.80000000000001, 79.4, 78.4, 79.5], 0.016293005636722424: [77.86, 77.86, 78.03999999999999, 76.7, 77.62], 0.3374801112580187: [78.84, 79.5, 79.17999999999999, 78.18, 79.36], 0.8693079186219261: [79.3, 79.74, 79.5, 78.42, 79.47999999999999], 0.4527237751290063: [78.96, 79.52, 79.34, 78.22, 79.47999999999999]}


<font face="微软雅黑" size=5> c.compare the effect and get top 10 words


```python


sort_distance_neg = sorted(total_count['neg'].items(),key=lambda kv:kv[1],reverse=True)
sort_distance_pos = sorted(total_count['pos'].items(),key=lambda kv:kv[1],reverse=True)

L_neg = sort_distance_neg[:10]
L_pos = sort_distance_pos[:10]
print("the top 10 words that predicts pos are:")
for i in range(0,len(L_pos)):
    for item,value in index.items():
        if value == L_pos[i][0]:
            print(item)


print("\nthe top 10 words that predicts neg are:")
for i in range(0,len(L_neg)):
    for item,value in index.items():
        if value == L_neg[i][0]:
            print(item)
            
score_laplace = np.sum(scores)/len(scores)
score_m_estimate = dict()

best_accuracy = 0.0
best_m = None
best_alg = dict()
for item, value in scores_m.items():
    score_tem = sum(value) / len(value)
    if score_tem > best_accuracy:
        best_accuracy = score_tem
        best_m  = item
score_m_estimate[best_m] = best_accuracy
# print(score_laplace)
# print(score_m_estimate)

if best_accuracy > score_laplace:
    best_alg[best_m] = best_accuracy
else:
    best_alg['laplace'] = score_laplace

print("\n\n")
for item,value in best_alg.items():
    if item != 'laplace':
        print("The best algorithm is m-estimate,the best m is: ", item)
        print("The best accuracy is: ", value)
    else:
        print("The best algorithm is m-estimate,the best accuracy is: " ,value)
    
    
```

    the top 10 words that predicts pos are:
    I
    's
    The
    br
    film
    movie
    one
    It
    This
    like
    
    the top 10 words that predicts neg are:
    I
    The
    's
    movie
    br
    film
    one
    like
    It
    This
    
    
    
    The best algorithm is m-estimate,the best m is:  0.9801648133343199
    The best accuracy is:  79.32399999999998


<font face="微软雅黑" size=5> d.using the test dataset


```python
test_set = []
actual = []
for row in test_content:
    row_copy = list(row)
    test_set.append(row_copy)
    row_copy[-1] = None
    actual.append(row[-1])


for item,value in best_alg.items():
    if item != 'laplace':
        predicted = algorithm1(all_content, test_set, value)
        accuracy = accuracy_metric(actual,predicted)       
    else:
        predicted = algorithm(all_content,test_set)
        accuracy = accuracy_metric(actual, predicted)
        
print("final accuracy is: ",accuracy)      
```

    final accuracy is:  77.804



```python

```