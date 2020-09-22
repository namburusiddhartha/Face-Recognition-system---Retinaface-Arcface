from numpy import load
from numpy import expand_dims
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

def get_pred(embs):
    emba= in_encoder.transform(embs)
    predict = model.predict(emba)
    predict = predict + 1 
    return predict

# load dict of arrays
dict_data = load('dataembed.npz')
data = dict_data['arr_0']
train = data[:,0:512]
test = data[:,512]
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.1, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(X_train)
testX = in_encoder.transform(X_test)
out_encoder = LabelEncoder()
out_encoder.fit(y_train)
trainy = out_encoder.transform(y_train)
testy = out_encoder.transform(y_test)
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
samples = expand_dims(train[55,0:512], axis=0)
X = get_pred(samples)
Y = test[55]
print(X,Y)

