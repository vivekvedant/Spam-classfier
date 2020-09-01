

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from wordcloud import WordCloud
import pickle
from sklearn import metrics
import numpy as np
import itertools
from sklearn.model_selection import train_test_split


sms_data = pd.read_csv('spam.csv',encoding = "ISO-8859-1")
df = pd.DataFrame(sms_data)



df.head()



df.drop(sms_data.iloc[:,2:],1,inplace = True)


df



df.rename(columns = {'v1':'label','v2':'message'},inplace = True)




df



df['label'] = df['label'].map({'ham': 0, 'spam': 1})


df


df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)



spam_words = ' '.join(list(df[df['label'] == 1]['message']))
spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()
spam_wc.to_file("spam.png")



ham_words = ' '.join(list(df[df['label'] == 0]['message']))
ham_wc = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()
ham_wc.to_file("ham.png")


ps = PorterStemmer()
wordnet=WordNetLemmatizer()
corpus = []
for i in range(len(df['message'])):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()



np.unique(X)


y = df['label']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)





def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png")


from sklearn.ensemble import RandomForestClassifier
class_weight = {0:1,1:200}
classifier = RandomForestClassifier(class_weight=class_weight)
classifier.fit(X_train,y_train)
random_predict = classifier.predict(X_test)
random_score = metrics.accuracy_score(y_test,random_predict)


print("accuracy:   %0.3f" % random_score)





classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['spam', 'ham'])



corpus_1 = []
test_msg = ["""You won 100 dollors"""]


test_vector = cv.transform(test_msg).toarray()
classifier.predict(test_vector)



#exporting model
pickle.dump(classifier, open("model.pkl", 'wb'))
#exporting vocabulary
pickle.dump(cv.vocabulary_, open("vocabulary.pkl", 'wb'))

