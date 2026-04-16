import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer #text to numerical representation
from sklearn.metrics import classification_report, confusion_matrix #accuracy alone is misleading, confusion matrix  and classification report
from sklearn.linear_model import LogisticRegression #target variable has 2 class
from sklearn.metrics import accuracy_score 
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv('ml/mail_data.csv')
print(df)
data = df.where(pd.notnull(df), '') #replaces null values with empty strings
data.head() #top 5(or more if u specify)
data.info() #data info
data.shape #columns + rows

X = data['text']

Y = data['spam']
print(Y)
print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 3) #80% training, 20% testing

print(X.shape)
print(X_train.shape)
print(X_test.shape)

print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True) #converts raw email text into numbers so that model understands

X_train_features = feature_extraction.fit_transform(X_train) #fit learns the vocab (what words exist, how frequent); #transform converts emails into numeric vectors based on vocab
X_test_features = feature_extraction.transform(X_test) #transform test data learnt from training


Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print (X_train)

model = LogisticRegression()
model.fit(X_train_features, Y_train)

models = {
    "Logistic Regression": LogisticRegression(),
    "Multinomial NB": MultinomialNB(),
    "Bernoulli NB": BernoulliNB()
}
for name, m in models.items():
    m.fit(X_train_features, Y_train)
    preds = m.predict(X_test_features)
    print(f"\n--- {name} ---")
    print(classification_report(Y_test, preds))
    print(X_train_features) #numerical features


model.fit(X_train_features, Y_train) #trains the model + sees features labels (learns patterns)

prediction_on_training_data = model.predict(X_train_features) #actually predicts
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data) #% of emails guessed correctly

print ('accuracy on training data: ', accuracy_on_training_data)
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print('accuracy on test data: ', accuracy_on_test_data)

input_your_mail = [""]
input_data_features = feature_extraction.transform(input_your_mail)

prediction=model.predict(input_data_features)
print(prediction)

if(prediction[0]==1):
    print('Ham mail')

else:
    print('Spam mail')

print(classification_report(Y_test, prediction_on_test_data))


pickle.dump(model, open('ml/spam_model.pkl', 'wb'))
pickle.dump(feature_extraction, open('ml/vectorizer.pkl', 'wb'))


scores = cross_val_score(model, X_train_features, Y_train, cv=5)
print("individual fold scores:", scores)
print("average accuracy:", scores.mean())


ConfusionMatrixDisplay.from_predictions(Y_test, prediction_on_test_data)
plt.title("logistic regression - confusion matrix")
plt.show()


spam_words = ' '.join(data[data['spam'] == 0]['text'])
ham_words = ' '.join(data[data['spam'] == 1]['text'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

WordCloud(background_color='black').generate(spam_words)
axes[0].imshow(WordCloud(background_color='black').generate(spam_words))
axes[0].set_title('Spam Words')
axes[0].axis('off')

axes[1].imshow(WordCloud(background_color='white').generate(ham_words))
axes[1].set_title('Ham Words')
axes[1].axis('off')

plt.show()


feature_names = feature_extraction.get_feature_names_out()
coefs = model.coef_[0]

top_spam = sorted(zip(coefs, feature_names))[:15]
top_ham = sorted(zip(coefs, feature_names))[-15:]

print("Top spam words:")
for coef, word in top_spam:
    print(f"  {word}: {coef:.3f}")

print("\nTop ham words:")
for coef, word in top_ham:
    print(f"  {word}: {coef:.3f}")