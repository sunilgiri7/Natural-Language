import pandas as pd
df= pd.read_csv("Data.csv", encoding = "ISO-8859-1")

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

data = train.iloc[:,2:27]
data.replace('[^a-zA-Z]',' ', regex=True, inplace=True)
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index

for index in new_Index:
    data[index] = data[index].str.lower()

' '.join(str(x) for x in data.iloc[1,0:25])
headlines = []
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
count = CountVectorizer(ngram_range=(2,2))
traindataset = count.fit_transform(headlines)

randomclassifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
randomclassifier.fit(traindataset, train['Label'])

test_transform = []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in data.iloc[row,2:27]))
test_dataset = count.transform(test_transform)
prediction = randomclassifier.predict(test_dataset)

from sklearn.metrics import accuracy_score
acc = accuracy_score(test['Label'], prediction)