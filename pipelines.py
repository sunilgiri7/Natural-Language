import pandas as pd
import numpy
data = {'name': ['sunil', 'rishab', 'elon', 'mark', 'jeff'],
        'age': [21, 20, 45, 50, 60],
        'job': ['coorporate', 'trader', 'CEO-of-SpaceX', 'CEO-of-meta', 'CEO-of-amazon'],
        'gender': ['m','f','m','m','m']}
df = pd.DataFrame(data)

## Normally we can drop our column or any preprocessing with simple way like
# df= df.drop(['job'], axis=1)

# but we can do it through pipelines also
from sklearn.base import BaseEstimator, TransformerMixin
class ColDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
    def transform(self, X):
        return X.drop(['job'], axis=1)

drop = ColDropper()
df = drop.fit_transform(df)
print(df)