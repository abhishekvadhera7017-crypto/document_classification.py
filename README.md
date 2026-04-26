# Document Classification

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {
    'text':[
        'football match today',
        'government policy announced',
        'new smartphone launched',
        'team won the game',
        'election results declared',
        'latest tech trends'
    ],
    'category':['sports','politics','tech','sports','politics','tech']
}

df = pd.DataFrame(data)

vec = TfidfVectorizer()
X = vec.fit_transform(df['text'])
y = df['category']

model = LogisticRegression()
model.fit(X,y)

# Test
print(model.predict(vec.transform(['new election news'])))
