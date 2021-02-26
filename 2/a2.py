import math
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import bonnerlib2D as bl2d

with open('dataA2Q1.pickle','rb') as file:
    dataTrain,dataTest = pickle.load(file)

############################ QUESTION 1 ##################################
print("\nQuestion 1:")
def fit_plot(dataTrain,dataTest,K):
    # Need the min and max of for linspace 
    xMin = min(dataTrain[0])
    xMax = max(dataTrain[0])

    # z = ψ(x) = [1,sin(x),sin(2x), ...sin(kx), cos(x), cos(2x), ..., cos(kx)]
    # Feature matrix of the data sets and for the linspace
    k = np.arange(1,K+1)
    f_Train = np.array([dataTrain[0]]).reshape(-1,1)
    fm_Train = np.concatenate((np.ones((f_Train.shape[0],1)), np.sin(f_Train*k), np.cos(f_Train*k)), axis=1)

    f_Test = np.array([dataTest[0]]).reshape(-1,1)
    fm_Test = np.concatenate((np.ones((f_Test.shape[0],1)), np.sin(f_Test*k), np.cos(f_Test*k)), axis=1)

    xList = np.linspace(xMin,xMax, 1000)
    xList = np.array([xList]).reshape(-1,1)
    fm_xList = np.concatenate((np.ones((xList.shape[0],1)), np.sin(xList*k), np.cos(xList*k)), axis=1)

    # Linear least squares for the feature matrix
    w = np.linalg.lstsq(fm_Train,dataTrain[1], rcond=None)[0]

    # Training and Test error
    y_train = np.matmul(fm_Train,w.T)
    error_train = np.mean((dataTrain[1] - y_train)**2)
    y_test = np.matmul(fm_Test,w.T)
    error_test = np.mean((dataTest[1] - y_test)**2)
    y_xList = np.matmul(fm_xList,w.T)

    # Print statements
    print("Value of K: " + str(K))
    print("Training error: " +  str(error_train))
    print("Test error: ", str(error_test))
    print("Weight Vector: \n", w)

    # Plotting the graph based on K
    plt.scatter(dataTrain[0],dataTrain[1], s=20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(min(dataTrain[1])-5, max (dataTrain[1]) + 5)
    plt.plot(xList,y_xList, c="red")

print("\nQuestion 1(a):")
plt.figure()
plt.title("Question 1(a): the fitted function (K=4)")
fit_plot(dataTrain, dataTest,4)

print("\nQuestion 1(b):")
plt.figure()
plt.title("Question 1(b): the fitted function (K=3)")
fit_plot(dataTrain, dataTest,3)

print("\nQuestion 1(c):")
plt.figure()
plt.title("Question 1(c): the fitted function (K=9)")
fit_plot(dataTrain, dataTest,9)

print("\nQuestion 1(d):")
plt.figure()
plt.title("Question 1(d): the fitted function (K=12)")
fit_plot(dataTrain, dataTest,12)

print("\nQuestion 1(e):")
fig=plt.figure(figsize=(4,3))
plt.suptitle('Question 1(e): fitted functions for many values of K.')
for i in range(1, 13):
    plt.subplot(4,3,i)
    fit_plot(dataTrain,dataTest,i)

print("\nQuestion 1(f):")

print("\nI Don't Know")

# def validatationfold(X, n):
#     w = X[(n-1)*5:((n-1)*5) + 5]
#     print(w)

# print(dataTrain[0])
# validatationfold(dataTrain[0], 1)

############################ QUESTION 2 ##################################
print("\nQuestion 2:")
with open('dataA2Q2.pickle','rb') as file:
    dataTrain,dataTest = pickle.load(file)
    Xtrain,Ttrain = dataTrain
    Xtest,Ttest = dataTest

def plot_data(X,T):
    col = ["red" if x == 0 else "blue" if x == 1 else "green" for x in T]
    plt.scatter(X[:,0],X[:,1], s=2, c=col)
    plt.xlim(min(X[:,0]) - 0.1, max(X[:,0]) + 0.1)
    plt.ylim(min(X[:,1]) - 0.1,max(X[:,1]) + 0.1)

# plot_data(Xtrain, Ttrain)

print("\nQuestion 2(b):")
import sklearn.linear_model as lin
clf = lin.LogisticRegression(multi_class='multinomial', solver='lbfgs') # create a classification object, clf
clf.fit(Xtrain,Ttrain) # learn a logistic-regression classifier

accuracy1 = clf.score(Xtrain, Ttrain)

# Logistic Regression accuracy calculator
def accuracyLR(clf,X,T):
    w = clf.coef_ 
    w0 = clf.intercept_
    z = np.matmul(X,w.T) + w0
    y = np.exp(z.T) / np.sum(np.exp(z), axis=1)
    accuracy2 = np.mean(np.argmax(y, axis=0) == T)
    return accuracy2

print("Value of Logistic Regression accuracy1: " + str(accuracy1))
print("Value of Logistic Regression accuracy2: " + str(accuracyLR(clf,Xtrain,Ttrain)))
print("Difference of accuracy2 and accuracy1 for Logistic Regression: " + str(abs(accuracyLR(clf,Xtrain,Ttrain) - accuracy1)))

plt.figure()
plt.suptitle("Question 2(b): decision boundaries for logistic regression.")
plot_data(Xtrain, Ttrain)
bl2d.boundaries(clf)

print("\nQuestion 2(c):")
import sklearn.discriminant_analysis as dis
import scipy.stats as ss
clf2 = dis.QuadraticDiscriminantAnalysis(store_covariance=True)
clf2.fit(Xtrain,Ttrain)

accuracy1 = clf2.score(Xtrain,Ttrain)

# Quadratic Discriminative Analysis Calculator
def accuracyQDA(clf,X,T):
    d = []
    for i in range(len(clf.means_)):
        d.append(ss.multivariate_normal.pdf(X, mean=clf.means_[i], cov=clf.covariance_[i]) * clf.priors_[i])

    accuracy2 = np.mean(np.argmax(d, axis=0) == T)
    return accuracy2


print("Value of Discriminative Analysis accuracy1: " + str(accuracy1))
print("Value of Discriminative Analysis accuracy2: " + str(accuracyQDA(clf2,Xtrain,Ttrain)))
print("Difference of accuracy2 and accuracy1 for Discriminative Analysis: " + str(abs(accuracyQDA(clf2,Xtrain,Ttrain) - accuracy1)))

plt.figure()
plt.suptitle("Question 2(c): decision boundaries for quadratic discriminant analysis.")
plot_data(Xtrain, Ttrain)
bl2d.boundaries(clf2)

print("\nQuestion 2(d):")
import sklearn.naive_bayes as nb
clf_nbayes = nb.GaussianNB()
clf_nbayes.fit(Xtrain,Ttrain)

accuracy1 = clf_nbayes.score(Xtrain,Ttrain)

# Naive bayes accuracy calculator
def accuracyNB(clf,X,T):
    X = np.reshape(X,[X.shape[0], 1, X.shape[1]])
    y = np.prod(np.exp(-(X-clf.theta_)**2 / 2*(clf.sigma_**2)) / (np.sqrt(2*np.pi) * clf.sigma_))
    accuracy2 = np.mean(np.argmax(y, axis=0) == T)
    return accuracy2

print("Value of Gaussian Naive Bayes accuracy1: " + str(accuracy1))
print("Value of Gaussian Naive Bayes accuracy2: " + str(accuracyNB(clf_nbayes,Xtrain,Ttrain)))
print("Difference of accuracy2 and accuracy1 for Gaussian Naive Bayes: " + str(abs(accuracyNB(clf_nbayes,Xtrain,Ttrain) - accuracy1)))

plt.figure()
plt.suptitle("Question 2(d): decision boundaries for Gaussian naive Bayes.")
plot_data(Xtrain, Ttrain)
bl2d.boundaries(clf_nbayes)

########################### QUESTION 3 ##################################
print("\nQuestion 3:")
with open('dataA2Q2.pickle','rb') as file:
    dataTrain,dataTest = pickle.load(file)
    Xtrain,Ttrain = dataTrain
    Xtest,Ttest = dataTest

np.random.seed(0)

import sklearn.neural_network as nn

print("\nQuestion 3(b):")

# This creates a mpl classifier based on k hidden layers
def mplclf_units(k):
    mplclf = nn.MLPClassifier(hidden_layer_sizes=(k,), activation='logistic', solver='sgd', max_iter=1000, learning_rate_init=0.01, tol=10**(-6))
    mplclf.fit(Xtrain,Ttrain)
    accuracy = mplclf.score(Xtest,Ttest)
    print("Value of Neural Network accuracy for " + str(k) + " hidden unit(s): " + str(accuracy))
    return mplclf

mplclf1 = mplclf_units(1)

plt.figure()
plt.suptitle("Question 3(b): Neural net with 1 hidden unit.")
plot_data(Xtrain, Ttrain)
bl2d.boundaries(mplclf1)

print("\nQuestion 3(c):")
mplclf2 = mplclf_units(2)

plt.figure()
plt.suptitle("Question 3(c): Neural net with 2 hidden units.")
plot_data(Xtrain, Ttrain)
bl2d.boundaries(mplclf2)

print("\nQuestion 3(d):")
mplclf9 = mplclf_units(9)

plt.figure()
plt.suptitle("Question 3(d): Neural net with 9 hidden units.")
plot_data(Xtrain, Ttrain)
bl2d.boundaries(mplclf9)

print("\nQuestion 3(e):")

# This function changes that max_iterations that the mpl classifier at 7 hidden layers
def mplclf_iter_7(maxiter):
    mplclf = nn.MLPClassifier(hidden_layer_sizes=(7,), activation='logistic', solver='sgd', max_iter=maxiter, learning_rate_init=0.01, tol=10**(-6))
    mplclf.fit(Xtrain,Ttrain)
    accuracy = mplclf.score(Xtest,Ttest)
    print("Value of Neural Network accuracy for 7 hidden unit(s): " + str(accuracy))
    return mplclf

fig=plt.figure(figsize=(3,3))
plt.suptitle('Question 3(e): different numbers of epochs.')
for i in range(1, 10):
    plt.subplot(3,3,i)
    mpl_clff = mplclf_iter_7(2**(i+1))
    plot_data(Xtrain, Ttrain)
    bl2d.boundaries(mpl_clff)

print("\nQuestion 3(f):")

fig=plt.figure(figsize=(3,3))
plt.suptitle('Question 3(f): different initial weights')
for i in range(1, 10):
    np.random.seed(i)
    plt.subplot(3,3,i)
    mplclf5 = mplclf_units(5)
    plot_data(Xtrain, Ttrain)
    bl2d.boundaries(mplclf5)

print("\nQuestion 3(g):")
np.random.seed(0)
mplclf9 = mplclf_units(9)
accuracy1 = mplclf9.score(Xtrain,Ttrain)

# Neural Network accuracy calculator
def accuracyNN(clf,X,T):
    lay = np.matmul(X,clf.coefs_[0]) + clf.intercepts_[0]
    hlay = 1/(1+np.exp(-lay))
    lay2 = np.matmul(hlay, clf.coefs_[1]) + clf.intercepts_[1]
    y = np.exp(lay2.T) / np.sum(np.exp(lay2), axis=1)
    accuracy2 = np.mean(np.argmax(y, axis=0) == T)
    return accuracy2

print("Value of Neural Network accuracy1 for 9 hidden unit(s): " + str(accuracy1))
print("Value of Neural Network accuracy2 for 9 hidden unit(s): " + str(accuracyNN(mplclf9,Xtrain,Ttrain)))
print("Difference of accuracy2 and accuracy1 for Neural Network for 9 hidden unit(s): " + str(abs(accuracyNN(mplclf9,Xtrain,Ttrain) - accuracy1)))

print("\nQuestion 3(h):")

print("\nI Don't Know")

# def ceNN(clf,X,T):
#     CE1 = clf.predict_log_proba(X)
#     CE2 = clf.predict_log_proba(X)

#     return 0, 0

# CE1, CE2 = ceNN(mplclf9,Xtrain,Ttrain)

# print("Value of Neural Network CE1 for 9 hidden unit(s): " + str(CE1))
# print("Value of Neural Network CE2 for 9 hidden unit(s): " + str(CE2))
# print("Difference of CE2 and CE1 for Neural Network for 9 hidden unit(s): " + str(abs(CE2 - CE1)))

########################### QUESTION 5 ##################################
print("\nQuestion 5(a):")

with open('mnistTVT.pickle','rb') as f:
    Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)

# Reduced data set for digits 5 and 6
reduced_Ttrain = Ttrain[np.logical_or(Ttrain == 5, Ttrain == 6)]
reduced_Xtrain = Xtrain[np.logical_or(Ttrain == 5, Ttrain == 6)]
reduced_Tval = Tval[np.logical_or(Tval == 5, Tval == 6)]
reduced_Xval = Xval[np.logical_or(Tval == 5, Tval == 6)]
reduced_Ttest = Ttest[np.logical_or(Ttest == 5, Ttest == 6)]
reduced_Xtest = Xtest[np.logical_or(Ttest == 5, Ttest == 6)]

# Change the Target set to 1's and 0's
reduced_Ttrain = np.where(reduced_Ttrain==5, 1, reduced_Ttrain)
reduced_Ttrain = np.where(reduced_Ttrain==6, 0, reduced_Ttrain)
reduced_Ttest = np.where(reduced_Ttest==5, 1, reduced_Ttest)
reduced_Ttest = np.where(reduced_Ttest==6, 0, reduced_Ttest)

# Reduced data set for digits 4 and 5
reduced_Ttrain2 = Ttrain[np.logical_or(Ttrain == 4, Ttrain == 5)]
reduced_Xtrain2 = Xtrain[np.logical_or(Ttrain == 4, Ttrain == 5)]
reduced_Tval2 = Tval[np.logical_or(Tval == 4, Tval == 5)]
reduced_Xval2 = Xval[np.logical_or(Tval == 4, Tval == 5)]
reduced_Ttest2 = Ttest[np.logical_or(Ttest == 4, Ttest == 5)]
reduced_Xtest2 = Xtest[np.logical_or(Ttest == 4, Ttest == 5)]

# Change the Target set to 1's and 0's
reduced_Ttrain2 = np.where(reduced_Ttrain2==4, 1, reduced_Ttrain2)
reduced_Ttrain2 = np.where(reduced_Ttrain2==5, 0, reduced_Ttrain2)
reduced_Ttest2 = np.where(reduced_Ttest2==4, 1, reduced_Ttest2)
reduced_Ttest2 = np.where(reduced_Ttest2==5, 0, reduced_Ttest2)

print("\nQuestion 5(b):")
print("\nI Don't Know")

# mplclf = nn.MLPClassifier(hidden_layer_sizes=(2,), activation='tanh', solver='sgd', max_iter=1000, learning_rate_init=0.01, tol=10**(-6))

# def evaluateNN(clf,X,T):
#     return 0

print("\nQuestion 5(c):")
print("\nI Don't Know")

print("\nQuestion 5(d):")
print("\nI Don't Know")

print("\nQuestion 5(e):")
print("\nI Don't Know")

print("\nQuestion 5(f):")
print("\nI Don't Know")

print("\nQuestion 5(g):")
print("\nI Don't Know")

print("\nQuestion 5(h):")
print("\nI Don't Know")

plt.show()
