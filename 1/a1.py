import math
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import bonnerlib3 as bl3d

np.random.seed(3)

print("\nQuestion 1(a):")
B = np.random.rand(4,5)
print(B)

print("\nQuestion 1(b):")
y = np.random.rand(4,1)
print(y)

print("\nQuestion 1(c):")
C = B.reshape(2,10)
print(C)

print("\nQuestion 1(d):")
D = B - y
print(D)

print("\nQuestion 1(e):")
z = y.reshape(4)
print(z)

print("\nQuestion 1(f):")
B[:,3] = z
print(B)

print("\nQuestion 1(g):")
D[:,0] = B[:,2] + z
print(D)

print("\nQuestion 1(h):")
print(B[:3])

print("\nQuestion 1(i):")
print(np.array([B[:,1],B[:,3]]).T)

print("\nQuestion 1(j):")
print(np.log(B))

print("\nQuestion 1(k):")
print(np.sum(B))

print("\nQuestion 1(l):")
print(np.array([np.max(B[:,0]), np.max(B[:,1]), np.max(B[:,2]), np.max(B[:,3]), np.max(B[:,4])] ))

print("\nQuestion 1(m):")
print(max(np.sum(B[0]), np.sum(B[1]), np.sum(B[2]), np.sum(B[3])))

print("\nQuestion 1(n):")
print(np.matmul(B.T,D))

print("\nQuestion 1(o):")
print(np.matmul(np.matmul(y.T,D),np.matmul(D.T,y)))
print(" ")

# print("\nQuestion 2(a):")

def matrix_poly(A):
    n = A.shape[0]
    B = np.zeros_like(A, dtype=np.float)
    C = np.zeros_like(A, dtype=np.float)
    # B = (A + A * A)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                B[i][j] += A[i][k] * A[k][j]
    # C = A + A * (A + A * A)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C

# print("\nQuestion 2(b):")

def timing(N):
    A = np.random.rand(N,N)
    # Timer for B1
    start_timer = time.time()
    B1 = matrix_poly(A)
    end_timer = time.time() - start_timer
    print("B1 execution time at N = " + str(N) + " is " + str(end_timer) + " seconds")
    # Timer for B2
    start_timer2 = time.time()
    B2 = np.matmul(A, np.matmul(A,A))
    end_timer2 = time.time() - start_timer2
    print("B2 execution time at N = " + str(N) + " is " + str(end_timer2) + " seconds")
    print("Magnitude of the difference matrix: " + str(np.sum(abs(B1 - B2))))
    print("The number of floating-point multiplications is: " + str(2*(N**3)))

print("\nQuestion 2(c):")
# timing(100)
# timing(300)
# timing(1000)

# print("\nQuestion 3(a):")

with open('dataA1Q3.pickle','rb') as f:
    dataTrain,dataTest = pickle.load(f)

def least_squares(x,t):
    # X has two columns: the first column is all 1's and 
    # the second column consists of the input values
    x_one = np.ones(x.shape[0])
    X = np.array([x_one, x])
    # w = (X^TX)^-1 X^Tt
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X,X.T)),X),t)
    return w

# print("\nQuestion 3(b):")

def plot_data(x,t):
    w = least_squares(x,t)
    plt.title("Question 3(b): the fitted line")
    plt.scatter(x,t)
    slope, intercept = np.polyfit(x,t,1)
    plt.plot(x, x * slope + intercept, 'r')
    return w[0], w[1]

# plot_data(dataTrain[0],dataTrain[1])

# print("\nQuestion 3(c):")

def error(a,b,X,T):
    loss = (T - ((a * X) + b))**2
    return np.mean(loss)

print("\nQuestion 3(d):")

plot_a, plot_b = plot_data(dataTrain[0],dataTrain[1])
print("Value of a for the fitted line is: " + str(plot_a))
print("Value of b for the fitted line is: " + str(plot_b))
print("Training error is: " + str(error(plot_a,plot_b, dataTrain[0], dataTrain[1])))
print("Test error is: " + str(error(plot_a,plot_b, dataTest[0], dataTest[1])))

print("\nQuestion 4(a):")

with open('dataA1Q4v2.pickle','rb') as f:
    Xtrain,Ttrain,Xtest,Ttest = pickle.load(f)

import sklearn.linear_model as lin
clf = lin.LogisticRegression() # create a classification object, clf
clf.fit(Xtrain,Ttrain) # learn a logistic-regression classifier
w = clf.coef_[0] # weight vector
w0 = clf.intercept_[0] # bias term

print("Weight Vector: " + str(w))
print("Bias Term: " + str(w0))

print("\nQuestion 4(b):")

# Method 1
accuracy1 = clf.score(Xtest,Ttest)
# Method 2
z = np.matmul(Xtest,w.T) + w0
print(z)
y = 1 / (1 + np.exp(-z))
pred = y >= 0.5
accuracy2 = np.sum(pred == Ttest)/len(Ttest)

print("Value of accuracy1: " + str(accuracy1))
print("Value of accuracy2: " + str(accuracy2))
print("Difference of accuracy2 and accuracy1: " + str(accuracy2 - accuracy1))

# print("\nQuestion 4(c):")

# bl3d.plot_db(Xtrain, Ttrain, w, w0, 30, 5)
# plt.suptitle('Question 4(c): Training data and decision boundary')

# print("\nQuestion 4(d):")

# bl3d.plot_db(Xtrain, Ttrain, w, w0, 30, 20)
# plt.suptitle('Question 4(d): Training data and decision boundary')

print("\nQuestion 5:")

def gd_logreg(lrate):
    np.random.seed(3)
    acc_train = []  # Accuracy for training data set
    acc_test = []   # Accuracy for training data set
    ce_train = []   # Average Cross entropy for training data set
    ce_test = []    # Average Cross entropy for test data set

    iterations = 0
    iterations_list = []
    entropy_diff = 1
    prev_entropy_diff = 1

    weight_vector = np.random.randn(4) / 1000
    ones = np.ones((2000, 1))

    # Vectorized Training data set
    oneXtrain = np.insert(Xtrain, 0, ones.T, axis=1)
    # Vectorized Test data set
    oneXtest = np.insert(Xtest, 0, ones.T, axis=1)

    while(entropy_diff >= 10**(-10)): 
        # logistic regression on train data
        z_train = np.matmul(oneXtrain,weight_vector)
        y_train = 1 / (1 + np.exp(-z_train))
        # logistic regression on test data
        z_test = np.matmul(oneXtest,weight_vector)
        y_test = 1 / (1 + np.exp(-z_test))

        weight_vector = weight_vector - ((lrate / oneXtrain.shape[0]) * np.dot(oneXtrain.T, (y_train - Ttrain)))
        
        # Cross entropy and accuracy for Training data set
        entropy_train = Ttrain * np.logaddexp(0, -z_train) + (1-Ttrain) * np.logaddexp(0, z_train)
        ce_train.append(np.mean(entropy_train))
        pred_train = y_train >= 0.5
        accuracy_train = np.sum(pred_train == Ttrain)/len(Ttrain)
        acc_train.append(accuracy_train)
        # Cross entropy and accuracy for Test data set
        entropy_test = Ttest * np.logaddexp(0, -z_test) + (1-Ttest) * np.logaddexp(0, z_test)
        ce_test.append(np.mean(entropy_test))
        pred_test = y_test >= 0.5
        accuracy_test = np.sum(pred_test == Ttest)/len(Ttest)
        acc_test.append(accuracy_test)

        entropy_diff = prev_entropy_diff - np.mean(entropy_train)
        prev_entropy_diff = np.mean(entropy_train)

        iterations+=1
        iterations_list.append(iterations)


    print("The final weight vector is: " + str(weight_vector))
    print("The number of iterations is: " + str(iterations))
    print("The learning rate is: " + str(lrate))
    print("The weight vector in Question 4 is: " + str(w))
    print("The bias term  in Question 4 is: " + str(w0))

    plt.figure()
    plt.suptitle('Question 5: Training and test loss v.s. iterations.')
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    plt.plot(iterations_list, ce_train, color='blue')
    plt.plot(iterations_list, ce_test, color='red')
    
    plt.figure()
    plt.suptitle('Question 5: Training and test loss v.s. iterations (logscale)')
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    plt.semilogx(ce_train, color='blue')
    plt.semilogx(ce_test, color='red')

    plt.figure()
    plt.suptitle('Question 5: Training and test accuracy v.s. iterations (log scale).')
    plt.xlabel("Iteration number")
    plt.ylabel("Accuracy")
    plt.semilogx(acc_train, color='blue')
    plt.semilogx(acc_test, color='red')

    plt.figure()
    plt.suptitle('Question 5: last 100 training cross entropies.')
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    plt.plot(iterations_list[-100:], ce_train[-100:], color='blue')

    plt.figure()
    plt.suptitle('Question 5: test loss from iteration 50 on (log scale)')
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    plt.semilogx(ce_test[51:], color='red')

    bl3d.plot_db(Xtrain, Ttrain, weight_vector[1:], weight_vector[0], 30, 5)
    plt.suptitle('Question 5: Training data and decision boundary')
    plt.show()

# gd_logreg(1)

print("\nQuestion 6(abc):")

from sklearn.neighbors import KNeighborsClassifier

with open('mnistTVT.pickle','rb') as f:
    Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)

# Reduced data set for digits 5 and 6
reduced_Ttrain = Ttrain[np.logical_or(Ttrain == 5, Ttrain == 6)]
reduced_Xtrain = Xtrain[np.logical_or(Ttrain == 5, Ttrain == 6)]
reduced_Tval = Tval[np.logical_or(Tval == 5, Tval == 6)]
reduced_Xval = Xval[np.logical_or(Tval == 5, Tval == 6)]
reduced_Ttest = Ttest[np.logical_or(Ttest == 5, Ttest == 6)]
reduced_Xtest = Xtest[np.logical_or(Ttest == 5, Ttest == 6)]

# Smaller version of Training data consisting of 2000 elements
smaller_Ttrain = reduced_Ttrain[:2000]
smaller_Xtrain = reduced_Xtrain[:2000]

# Printing a 4x4 grid of the 5's and 6's
fig=plt.figure(figsize=(4,4))
plt.suptitle('Question 6(b): 16 MNIST training images.')
for i in range(1, 17):
    s = reduced_Xtrain[i].reshape(28,28)
    fig.add_subplot(4,4,i)
    plt.axis('off')
    plt.imshow(s, cmap='Greys', interpolation='nearest')


def knn56class():
    k = []          # Odd Nieghbours    
    acc_val = []    # Accuracies of Validation data
    acc_train = []  # Accuracies of Training data
    for i in range(1,20):
        if(i % 2 == 1): # check odd
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(reduced_Xtrain, reduced_Ttrain)
            # Accuracy of Validation data
            sc = knn.score(reduced_Xval, reduced_Tval)
            # Accuracy of Smaller Training data
            sc2 = knn.score(smaller_Xtrain, smaller_Ttrain)
            k.append(i)
            acc_val.append(sc)
            acc_train.append(sc2)

    plt.figure()
    plt.suptitle('Question 6(c): Training and Validation Accuracy for KNN, digits 5 and 6.')
    plt.xlabel("Number of Neighbours, K.")
    plt.ylabel("Accuracy.")
    plt.plot(k, acc_val,color='red')
    plt.plot(k,acc_train,color='blue')
    plt.show()

    # Finding Best K and finding accuracy of best k
    best_k = k[acc_val.index(max(acc_val))]
    knn_bestk = KNeighborsClassifier(best_k)
    knn_bestk.fit(reduced_Xtest,reduced_Ttest)
    sc_bestk = knn_bestk.score(reduced_Xtest, reduced_Ttest)
    sc2_bestk = knn_bestk.score(reduced_Xval, reduced_Tval)

    print("Best value of K: " + str(best_k))
    print("Accuracy of best K for reduced Test data: " + str(sc_bestk))
    print("Accuracy of best K for reduced Validation data: " + str(sc2_bestk))

print("\nQuestion 6(d):")

# Reduced data set for digits 4 and 7
reduced_Ttrain2 = Ttrain[np.logical_or(Ttrain == 4, Ttrain == 7)]
reduced_Xtrain2 = Xtrain[np.logical_or(Ttrain == 4, Ttrain == 7)]
reduced_Tval2 = Tval[np.logical_or(Tval == 4, Tval == 7)]
reduced_Xval2 = Xval[np.logical_or(Tval == 4, Tval == 7)]
reduced_Ttest2 = Ttest[np.logical_or(Ttest == 4, Ttest == 7)]
reduced_Xtest2 = Xtest[np.logical_or(Ttest == 4, Ttest == 7)]

# Smaller version of Training data consisting of 2000 elements
smaller_Ttrain2 = reduced_Ttrain2[:2000]
smaller_Xtrain2 = reduced_Xtrain2[:2000]

# Printing a 4x4 grid of the 4's and 7's
fig=plt.figure(figsize=(4,4))
plt.suptitle('Question 6(d): 16 MNIST training images.')
for i in range(1, 17):
    s = reduced_Xtrain2[i].reshape(28,28)
    fig.add_subplot(4,4,i)
    plt.axis('off')
    plt.imshow(s, cmap='Greys', interpolation='nearest')

def knn47class():
    k = []          # Odd Nieghbours  
    acc_val = []    # Accuracies of Validation data
    acc_train = []  # Accuracies of Training data
    for i in range(1,20):
        if(i % 2 == 1):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(reduced_Xtrain2, reduced_Ttrain2)
            # Accuracy of Validation data
            sc = knn.score(reduced_Xval2, reduced_Tval2)
            # Accuracy of Smaller Training data
            sc2 = knn.score(smaller_Xtrain2, smaller_Ttrain2)
            k.append(i)
            acc_val.append(sc)
            acc_train.append(sc2)

    plt.figure()
    plt.suptitle('Question 6(d): Training and Validation Accuracy for KNN, digits 4 and 7.')
    plt.xlabel("Number of Neighbours, K.")
    plt.ylabel("Accuracy.")
    plt.plot(k, acc_val,color='red')
    plt.plot(k,acc_train,color='blue')
    plt.show()

    # Finding Best K and finding accuracy of best k
    best_k = k[acc_val.index(max(acc_val))]
    knn_bestk = KNeighborsClassifier(best_k)
    knn_bestk.fit(reduced_Xtest2,reduced_Ttest2)
    sc_bestk = knn_bestk.score(reduced_Xtest2, reduced_Ttest2)
    sc2_bestk = knn_bestk.score(reduced_Xval2, reduced_Tval2)

    print("Best value of K: " + str(best_k))
    print("Accuracy of best K for reduced Test data: " + str(sc_bestk))
    print("Accuracy of best K for reduced Validation data: " + str(sc2_bestk))

# knn56class()
# knn47class()

