

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

    /kaggle/input/nlp-getting-started/test.csv
    /kaggle/input/nlp-getting-started/sample_submission.csv
    /kaggle/input/nlp-getting-started/train.csv


# 1 head files


```python
import pandas as pd
import jieba
import pickle
import gensim
import numpy as np
from tqdm import tqdm
import re
import string
import nltk
from nltk.corpus import stopwords

import os
import datetime
import numpy as np
import random
from scipy.sparse import hstack


from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection

```

# 2 Reading CSV files


```python
label_set = pd.read_csv('../input/nlp-getting-started/train.csv',encoding='gbk',keep_default_na=False)
label_set = label_set.reset_index(drop=True)
print(label_set.text)
print("Training data shape:",label_set.shape)
label_set.head()
```

    0       Our Deeds are the Reason of this #earthquake M...
    1                  Forest fire near La Ronge Sask. Canada
    2       All residents asked to 'shelter in place' are ...
    3       13,000 people receive #wildfires evacuation or...
    4       Just got sent this photo from Ruby #Alaska as ...
                                  ...                        
    7608    Two giant cranes holding a bridge collapse int...
    7609    @aria_ahrary @TheTawniest The out of control w...
    7610    M1.94 [01:04 UTC]?5km S of Volcano Hawaii. htt...
    7611    Police investigating after an e-bike collided ...
    7612    The Latest: More Homes Razed by Northern Calif...
    Name: text, Length: 7613, dtype: object
    Training data shape: (7613, 5)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td></td>
      <td></td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td></td>
      <td></td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td></td>
      <td></td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td></td>
      <td></td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td></td>
      <td></td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The traing data has 7613 observations and 5 features including the TARGET
```


```python
#Testing data
test_label_set = pd.read_csv('../input/nlp-getting-started/test.csv',encoding='gbk',keep_default_na=False)
tset_label_set = label_set.reset_index(drop=True)
print(test_label_set.text)
print("Training data shape:",test_label_set.shape)
test_label_set.head()
```

    0                      Just happened a terrible car crash
    1       Heard about #earthquake is different cities, s...
    2       there is a forest fire at spot pond, geese are...
    3                Apocalypse lighting. #Spokane #wildfires
    4           Typhoon Soudelor kills 28 in China and Taiwan
                                  ...                        
    3258    EARTHQUAKE SAFETY LOS ANGELES 聣脹脪 SAFETY FASTE...
    3259    Storm in RI worse than last hurricane. My city...
    3260    Green Line derailment in Chicago http://t.co/U...
    3261    MEG issues Hazardous Weather Outlook (HWO) htt...
    3262    #CityofCalgary has activated its Municipal Eme...
    Name: text, Length: 3263, dtype: object
    Training data shape: (3263, 4)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td></td>
      <td></td>
      <td>Just happened a terrible car crash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td></td>
      <td></td>
      <td>Heard about #earthquake is different cities, s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td></td>
      <td></td>
      <td>there is a forest fire at spot pond, geese are...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td></td>
      <td></td>
      <td>Apocalypse lighting. #Spokane #wildfires</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td></td>
      <td></td>
      <td>Typhoon Soudelor kills 28 in China and Taiwan</td>
    </tr>
  </tbody>
</table>
</div>



# 3 Data preprocessing

## 1 data cleaning


```python
#removing some symbols
#training data
def clean_text(text):
    sentences = text.lower()
    sentences = re.sub (r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', sentences, flags=re.MULTILINE)
    sentences = re.sub('[!#?,.:";//|''@-]','',sentences)
    return sentences
    
label_set['text'] = label_set['text'].apply(lambda x: clean_text(x))
test_label_set['text'] = test_label_set['text'].apply(lambda x:clean_text(x))

all_text = pd.concat([label_set,test_label_set])
all_text.head()
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:12: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=False'.
    
    To retain the current behavior and silence the warning, pass 'sort=True'.
    
      if sys.path[0] == '':





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>target</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td></td>
      <td></td>
      <td>1.0</td>
      <td>our deeds are the reason of this earthquake ma...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td></td>
      <td></td>
      <td>1.0</td>
      <td>forest fire near la ronge sask canada</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td></td>
      <td></td>
      <td>1.0</td>
      <td>all residents asked to 'shelter in place' are ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td></td>
      <td></td>
      <td>1.0</td>
      <td>13000 people receive wildfires evacuation orde...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td></td>
      <td></td>
      <td>1.0</td>
      <td>just got sent this photo from ruby alaska as s...</td>
    </tr>
  </tbody>
</table>
</div>



# 4 Transforming features to a vector


```python
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
train_features = tfidf.fit_transform(label_set['text'])
test_features = tfidf.transform(test_label_set["text"])
print(train_features.shape)
print(test_features.shape)
```

    (7613, 16953)
    (3263, 16953)


# 5 Building a Text Classification model

## Naives Bayes Classifier


```python
nb = MultinomialNB()
scores = model_selection.cross_val_score(nb,train_features,label_set["target"],cv=5,scoring="f1")
scores
```




    array([0.58090452, 0.57327189, 0.62686567, 0.60991581, 0.75163399])



## predict


```python
nb.fit(train_features,label_set['target'])
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
submission['target'] = nb.predict(test_features)

submission.to_csv("submission.csv",index = False)
```
