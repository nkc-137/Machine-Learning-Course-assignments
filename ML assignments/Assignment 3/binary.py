import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv('30_train_features.csv')

for i in data.index:
    if data['OS'].iloc[i] <= 300:
        data['OS'].iloc[i] = 0
#     elif data['OS'].iloc[i] <= 450 and data['OS'].iloc[i]>300:
#         data['OS'].iloc[i] = 1
#     else:
#         data['OS'].iloc[i] = 2
    else:
        data['OS'].iloc[i] = 1


test_data = pd.read_csv('30_test_features.csv')
for i in test_data.index:
    if test_data['OS'].iloc[i] <= 300:
        test_data['OS'].iloc[i] = 0
#     elif test_data['OS'].iloc[i] <= 450 and test_data['OS'].iloc[i]>300:
#         test_data['OS'].iloc[i] = 1
#     else:
#         test_data['OS'].iloc[i] = 2
    else:
        test_data['OS'].iloc[i] = 1

X_train = data.drop(['OS'],axis=1)
y_train = data['OS']

X_test = test_data.drop(['OS'],axis=1)
y_test = test_data['OS']

num_labels = 2
one_hot_test = (np.arange(num_labels) == np.array(y_test)[:,None]).astype(np.float32)
one_hot_y = (np.arange(num_labels) == np.array(y_train)[:,None]).astype(np.float32)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

X_train = X_train.as_matrix()
X_test = X_test.as_matrix()

from sklearn import preprocessing
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

                                            """TENSORFLOW part"""
feature_size = X_train.shape[1]
beta = 0.01

tf_train = tf.constant(X_train)
tf_label = tf.constant(one_hot_y)

tf_test = tf.constant(X_test)
tf_test_label = tf.constant(one_hot_test)

W = tf.Variable(tf.random_normal(shape=(feature_size,num_labels),dtype=tf.float64,seed=45))
b = tf.Variable(tf.zeros(shape=(num_labels),dtype=tf.float64))

z = tf.matmul(tf_train,W) + b
yhat = tf.nn.softmax(z)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yhat,labels=tf_label))
regularizer = tf.nn.l2_loss(W)
loss = tf.reduce_mean(loss + beta*regularizer)

optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

test_pred = tf.nn.softmax(tf.matmul(tf_test,W) + b)
prediction = tf.nn.softmax(z)

epochs = 10000
costs = []
test_losses = []

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        sess.run(optimizer)

        cost = sess.run(loss)
        costs.append(cost)

        pred = sess.run(prediction)

        if (i%1000 == 0):
            print('Accuracy = ',accuracy(pred,one_hot_y))
    test_prediction = sess.run(test_pred)
    print('Test Accuracy = ', accuracy(test_prediction,one_hot_test))


                                         """RECALL AND PRECISION"""
TP = 0
TN = 0
FP = 0
FN = 0
for i in range(len(y_test)):
    if (y_test[i] == 1):
        if (final_labels[i] == 1):
            TP += 1
        else:
            FN += 1
    else:
        if (final_labels[i] == 1):
            FP += 1
        else:
            TN += 1
print(TP + TN + FP + FN)
sensitivity = TP/(TP + FN)
specificity = TN/(TN + FP)
print('Sensitivity = ',sensitivity)
print('Specificity = ',specificity)
