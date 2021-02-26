import math
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import scipy as sc
from sklearn.utils.testing import ignore_warnings

with open('mnistTVT.pickle','rb') as f:
    Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)

Xtrain = Xtrain[:5000]

Xtrain = Xtrain.astype(np.float64)
Xval = Xval.astype(np.float64)
Xtest = Xtest.astype(np.float64)

############################ QUESTION 1 ##################################
print("\nQuestion 1:")

from sklearn.decomposition import PCA

print("\nQuestion 1(a):")
def npca(n):
    pca = PCA(n_components=n)
    pca.fit(Xtrain)
    reduced_test = pca.transform(Xtest)
    projected_test = pca.inverse_transform(reduced_test)

    fig=plt.figure(figsize=(5,5))
    plt.suptitle("Question 1(c): MNIST test data projected onto " + str(n) + " dimensions")
    for i in range(1, 26):
        s = projected_test[i].reshape(28,28)
        fig.add_subplot(5,5,i)
        plt.axis('off')
        plt.imshow(s, cmap='Greys', interpolation='nearest')


npca(30)
print("\nQuestion 1(b):")
npca(3)
print("\nQuestion 1(c):")
npca(300)


print("\nQuestion 1(d):")
def myPCA(X,K):
    xmean = np.mean(X, axis=0)
    xcov = np.matmul((X-xmean.T).T, (X-xmean.T)) / X.shape[0]
    eig2 = sc.linalg.eigh(xcov, eigvals=(xcov.shape[0]-K, xcov.shape[0]-1))
    U = eig2[1]
    z = np.matmul(U.T,(X-xmean.T).T).T
    newx = xmean + np.matmul(z,U.T)
    return newx

# test = np.array([[90,60,90],[90,90,30],[60,60,60],[60,60,90],[30,30,30]])

print("\nQuestion 1(f):")

myXtrainP = myPCA(Xtrain, 100)
fig=plt.figure(figsize=(5,5))
plt.suptitle("Question 1(f): MNIST data projected onto 100 dimensions (mine)")
for i in range(1, 26):
    s = myXtrainP[i].reshape(28,28)
    fig.add_subplot(5,5,i)
    plt.axis('off')
    plt.imshow(s, cmap='Greys', interpolation='nearest')

pca = PCA(n_components=100, svd_solver='full')
pca.fit(Xtrain)
XtrainP = pca.transform(Xtrain)
XtrainP = pca.inverse_transform(XtrainP)

fig=plt.figure(figsize=(5,5))
plt.suptitle("Question 1(f): MNIST data projected onto 100 dimensions (sklearn).")
for i in range(1, 26):
    s = XtrainP[i].reshape(int(math.sqrt(XtrainP[i].shape[0])),int(math.sqrt(XtrainP[i].shape[0])))
    fig.add_subplot(5,5,i)
    plt.axis('off')
    plt.imshow(s, cmap='Greys', interpolation='nearest')

def RMS(X):
    return np.sqrt(np.mean(np.square(X)) / X.shape[0])

print(RMS(XtrainP - myXtrainP))


############################ QUESTION 2 ##################################
print("\nQuestion 2:")

small_Xtrain = Xtrain[:200]
small_Ttrain = Ttrain[:200]

sl_Xtrain = Xtrain[:300]
sl_Ttrain = Ttrain[:300]

import sklearn.discriminant_analysis as dis

print("\nQuestion 2(a):")
clf = dis.QuadraticDiscriminantAnalysis()
ignore_warnings(clf.fit)(small_Xtrain, small_Ttrain)

print("Accuracy of QDA on small training set:", clf.score(small_Xtrain,small_Ttrain))
print("Accuracy of QDA on full test set:", clf.score(Xtest,Ttest))

print("\nQuestion 2(b):")

def QDAclassifiers():
    val_acc = []
    train_acc = []
    k = [] 
    for i in range(0, 21):
        clf2 = dis.QuadraticDiscriminantAnalysis(reg_param=2**(-i))
        ignore_warnings(clf2.fit)(small_Xtrain, small_Ttrain)
        val_acc.append(clf2.score(Xval,Tval))
        train_acc.append(clf2.score(small_Xtrain,small_Ttrain))
        k.append(2**(-i))

    max_val_acc = max(val_acc)        
    max_index = val_acc.index(max_val_acc)
    print("Maximum Validation accuracy:", max_val_acc)
    print("Corresponding training accuracy:", train_acc[max_index])
    print("Regularization parameter:", max_index)

    plt.figure()
    plt.suptitle("Question 2(b): Training and Validation Accuracy for Regularized QDA")
    plt.xlabel("Regularization parameter")
    plt.ylabel("Accuracy")
    plt.semilogx(k,train_acc, color='blue')
    plt.semilogx(k,val_acc, color='red')

QDAclassifiers()

print("\nQuestion 2(d):")
def train2d(K,X,T):
    pca = PCA(n_components=K, svd_solver='full')
    pca.fit(X)
    reduced_x = pca.transform(X)
    clf = dis.QuadraticDiscriminantAnalysis()
    ignore_warnings(clf.fit)(reduced_x, T)
    acc = clf.score(reduced_x, T)
    return pca, clf, acc

def test2d(pca,qda,X,T):
    reduced_x = pca.transform(X)
    acc = qda.score(reduced_x,T)
    return acc

def qdaclass():
    val_acc = []
    train_acc = []
    k = [] 
    for i in range(1, 51):
        pca, clf, tacc = train2d(i, small_Xtrain, small_Ttrain)
        vacc = test2d(pca, clf, Xval, Tval)
        train_acc.append(tacc)
        val_acc.append(vacc)
        k.append(i)

    max_val_acc = max(val_acc)        
    max_index = k[val_acc.index(max_val_acc)]
    print("Maximum Validation accuracy:", max_val_acc)
    print("Corresponding training accuracy:", train_acc[max_index])
    print("Value of K:", max_index)

    plt.figure()
    plt.suptitle("Question 2(d): Training and Validation Accuracy for PCA + QDA")
    plt.xlabel("Reduced dimension")
    plt.ylabel("Accuracy")
    plt.plot(k,train_acc, color='blue')
    plt.plot(k,val_acc, color='red')

qdaclass()

print("\nQuestion 2(f):")

def overfitqdareduction():
    max_acc = []
    max_ind = []
    max_tacc = []
    for i in range(1,51):
        curr_acc = []
        curr_ind = []
        curr_tacc = []
        pca = PCA(n_components=i, svd_solver='full')
        pca.fit(small_Xtrain)
        reduced_x = pca.transform(small_Xtrain)
        for j in range(0,21):
            clf = dis.QuadraticDiscriminantAnalysis(reg_param=2**(-j))
            ignore_warnings(clf.fit)(reduced_x, small_Ttrain)
            s = pca.transform(Xval)
            acc = clf.score(s, Tval)
            t = pca.transform(small_Xtrain)
            tacc = clf.score(t, small_Ttrain)
            curr_acc.append(acc)
            curr_ind.append((i,2**(-j)))
            curr_tacc.append(tacc)

        max_acc.append(max(curr_acc))
        max_ind.append(curr_ind[curr_acc.index(max(curr_acc))])
        max_tacc.append(curr_tacc[curr_acc.index(max(curr_acc))])

    max_max_acc = max(max_acc)
    max_max_ind = max_ind[max_acc.index(max_max_acc)]
    max_max_tacc = max_tacc[max_acc.index(max_max_acc)]

    print("Max Training accuracy over all combinations:", max_max_acc)
    print("Corresponding training accuracy:", max_max_tacc)
    print("Corresponding reg_param:", max_max_ind[1])
    print("Corresponding value of K:", max_max_ind[0])

    plt.figure()
    plt.suptitle("Question 2(f): Maximum validation accuracy for QDA")
    plt.xlabel("Reduced dimension")
    plt.ylabel("maximum accuracy")
    plt.plot(max_acc)

overfitqdareduction()

############################ QUESTION 3 ##################################
print("\nQuestion 3:")

import sklearn.utils as sku

print("\nQuestion 3(a):")

def myBootstrap(X,T):
    X, T = sku.resample(X, T)
    ocurrance = np.unique(T, return_counts=True)[1]
    while ((len(ocurrance) != 10) or (0 in ocurrance) or (1 in ocurrance) or (2 in ocurrance)):
        X, T = sku.resample(X, T)
        ocurrance = np.unique(T, return_counts=True)[1]
    
    return X, T

print("\nQuestion 3(b):")

def qdaBSMNIST50():
    pm = []
    clf = dis.QuadraticDiscriminantAnalysis(reg_param=0.004)
    ignore_warnings(clf.fit)(small_Xtrain, small_Ttrain)
    valacc = clf.score(Xval, Tval)
    print("Validation accuracy of the base classifier:", valacc)
    for i in range(50):
        X, T = myBootstrap(small_Xtrain, small_Ttrain)
        clf = dis.QuadraticDiscriminantAnalysis(reg_param=0.004)
        ignore_warnings(clf.fit)(X,T)
        probmatrix = clf.predict_proba(Xval)
        pm.append(probmatrix)

    av_ac = np.mean(pm, axis=0) 
    sm = np.exp(av_ac.T) / np.sum(np.exp(av_ac), axis=1)
    pred = np.argmax(sm, axis=0)
    av = np.mean(pred == Tval)
    print("Validation accuracy of the bagged classifier of 50 resamples:", av)

qdaBSMNIST50()

print("\nQuestion 3(c):")

def qdaBSMNIST500():
    pm = []
    acc = []
    clf = dis.QuadraticDiscriminantAnalysis(reg_param=0.004)
    ignore_warnings(clf.fit)(small_Xtrain, small_Ttrain)
    for i in range(500):
        X, T = myBootstrap(small_Xtrain, small_Ttrain)
        clf = dis.QuadraticDiscriminantAnalysis(reg_param=0.004)
        ignore_warnings(clf.fit)(X,T)
        probmatrix = clf.predict_proba(Xval)
        pm.append(probmatrix)

        av_ac = np.mean(pm, axis=0) 
        sm = np.exp(av_ac.T) / np.sum(np.exp(av_ac), axis=1)
        pred = np.argmax(sm, axis=0)
        av = np.mean(pred == Tval)
        acc.append(av)

    plt.figure()
    plt.suptitle("Question 3(c): Validation accuracy")
    plt.xlabel("Number of bootstrap samples")
    plt.ylabel("Accuracy")
    plt.plot(acc)

    plt.figure()
    plt.suptitle("Question 3(c): Validation accuracy (log scale)")
    plt.xlabel("Number of bootstrap samples")
    plt.ylabel("Accuracy")
    plt.semilogx(acc)

qdaBSMNIST500()

print("\nQuestion 3(d):")

def train3d(K,R,X,T):
    pca = PCA(n_components=int(K), svd_solver='full')
    pca.fit(X)
    reduced_x = pca.transform(X)
    clf = dis.QuadraticDiscriminantAnalysis(reg_param=R)
    ignore_warnings(clf.fit)(reduced_x, T)
    return pca, clf
    
def proba3d(pca,qda,X):
    redx = pca.transform(X)
    pm = qda.predict_proba(redx)
    return pm

print("\nQuestion 3(e):")

def myBag(K,R):
    pca, clf = train3d(K,R,small_Xtrain, small_Ttrain)
    redxval = pca.transform(Xval)
    vacc = clf.score(redxval, Tval)

    pmm = []
    for i in range(200):
        X,T = myBootstrap(small_Xtrain, small_Ttrain)
        pca2, clf2 = train3d(K,R,X,T)
        pm = proba3d(pca2, clf2, Xval)
        pmm.append(pm)

    av_ac = np.mean(pmm, axis=0) 
    sm = np.exp(av_ac.T) / np.sum(np.exp(av_ac), axis=1)
    pred = np.argmax(sm, axis=0)
    av = np.mean(pred == Tval)
    return vacc, av

print("\nQuestion 3(f):")
vacc, av = myBag(100, 0.01)
print("Validation accuracy of the 100 classifier:", vacc)
print("Validation accuracy of the bagged classifier of 200 resamples:", av)

print("\nQuestion 3(g):") 

baseacc = []
bagacc = []

for i in range(50):
    K = np.random.uniform(1,11)
    R = np.random.uniform(0.2,1.0)
    base,bag = myBag(K,R)
    baseacc.append(base)
    bagacc.append(bag)

plt.figure()
plt.suptitle("Question 3(g): Bagged v.s. base validation accuracy")
plt.xlabel("Base validation accuracy")
plt.ylabel("Bagged validation accuracy")
plt.xlim(0,1)
plt.ylim(0,1)
plt.scatter(baseacc, bagacc, c="blue")
plt.plot([0,1], [0,1], c="red")

print("\nQuestion 3(h):") 

baseacc = []
bagacc = []

for i in range(1):
    K = np.random.uniform(50,201)
    R = np.random.uniform(0,0.05)
    base,bag = myBag(K,R)
    baseacc.append(base)
    bagacc.append(bag)

plt.figure()
plt.suptitle("Question 3(h): Bagged v.s. base validation accuracy")
plt.xlabel("Base validation accuracy")
plt.ylabel("Bagged validation accuracy")
plt.ylim(0,1)
plt.scatter(baseacc, bagacc, c="blue")
plt.axhline(max(bagacc), color="red")
print("Max bagged validation accuracy", max(bagacc))

############################ QUESTION 4 ##################################
print("\nQuestion 4:")

with open('dataA2Q2.pickle','rb') as file:
    dataTrain,dataTest = pickle.load(file)
    Xtrain,Ttrain = dataTrain
    Xtest,Ttest = dataTest

print("\nQuestion 4(a):")

def plot_clusters(X,R,Mu):
    R = R[:, np.argsort(np.sum(R, axis=0))]
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=R, s=5)
    plt.scatter(Mu[:, 0],Mu[:, 1], c="black")

print("\nQuestion 4(b):")

import sklearn.cluster as clu
clf = clu.KMeans(n_clusters=3)
clf.fit(Xtrain)
q = np.empty((0,3), int)
for i in clf.labels_:
    if(i == 0):
        q = np.append(q, np.array([[1,0,0]]), axis=0)
    elif (i == 1):
        q = np.append(q, np.array([[0,1,0]]), axis=0) 
    else: 
        q = np.append(q, np.array([[0,0,1]]), axis=0)
    
plot_clusters(Xtrain,q, clf.cluster_centers_)
plt.suptitle("Question 4(b): K means")
trainacc = clf.score(Xtrain, Ttrain)
testacc = clf.score(Xtest, Ttest)
print("Training accuracy:",trainacc)
print("Test accuracy:",testacc)

print("\nQuestion 4(c):")

print("I Don't Know")

print("\nQuestion 4(d):")

print("I Don't Know")

print("\nQuestion 4(e):")

print("I Don't Know")

print("\nQuestion 4(f):")

print("I Don't Know")

print("\nQuestion 4(g):")

print("I Don't Know")

print("\nQuestion 4(h):")

print("I Don't Know")

print("\nQuestion 4(i):")

print("I Don't Know")

print("\nQuestion 4(j):")

print("I Don't Know")

plt.show()

