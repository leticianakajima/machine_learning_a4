import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils
import pdb

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w, self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

class logRegL2(logReg):
    # L2 Regularized Logistic Regression
    def __init__(self, lammy=1.0, verbose=1, maxEvals=400):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        # pdb.set_trace()
        f = np.sum(np.log(1. + np.exp(-yXw)))
        f += 0.5 * self.lammy * np.sum(w ** 2)

        # Calculate the gradient value
        res = (- y / (1. + np.exp(yXw)))
        g = X.T.dot(res) + self.lammy * w

        return f, g

class logRegL1(logReg):
    # L1 Regularized Logistic Regression
    def __init__(self, lammy=1.0, maxEvals=400, verbose=1):
        self.verbose = verbose
        self.L1_lambda = lammy
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        # pdb.set_trace()
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = (- y / (1. + np.exp(yXw)))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.L1_lambda,
                                      self.maxEvals, X, y, verbose=self.verbose)

class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, l0_lammy=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = l0_lammy
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                               np.zeros(len(ind)),
                                               self.maxEvals,
                                               X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        # pdb.set_trace()

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i}


                w, loss = minimize(list(selected_new))
                loss += self.L0_lambda * len(selected_new)

                if (loss < minLoss):
                    minLoss = loss
                    bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))

class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)



class logLinearClassifier(logReg):
    def fit(self,X, y):
        n, d = X.shape
        k = np.unique(y).size
        print(k)

        self.W = np.zeros((k,d))

        #want to copy each instance to one column of matrix W
        for i in range(0,k):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            print(ytmp)
            (wtmp, f) = findMin.findMin(self.funObj, np.zeros(d), self.maxEvals, X, ytmp, verbose=self.verbose)
            print(wtmp.size)
            print(k)

            self.W[i] = wtmp

    def predict(self, X):
        return np.argmax(X @ self.W.T, axis=1)

class softmaxClassifier(logLinearClassifier):
    def funObj(self, w, X, y):
        n, d = X.shape
        k = np.unique(y).size

        W = np.reshape(w, (k, d))

        f = 0
        for i in range (n):
            val1 = np.sum(np.exp(W@X[i,:]))

            f += -W[y[i],:].dot(X[i,:]) + np.log(val1)

        g = np.zeros((k,d))
        for c in range(k):
            for j in range(d):
                for i in range(n):
                    Pdenom =  np.sum(np.exp(W@X[i,:]))

                    PVal = np.exp(np.dot(W[c,:], X[i,:]))/Pdenom

                    g[c,j] += X[i,j] * (PVal - (y[i] == c))

        return f, g.flatten()

    def fit(self,X, y):
        n, d = X.shape
        k = np.unique(y).size

        self.W = np.zeros(d*k)
        self.w = self.W

        utils.check_gradient(self, X, y)
        (self.W, f) = findMin.findMin(self.funObj, self.W, self.maxEvals, X, y, verbose=self.verbose)


        self.W = np.reshape(self.W, (k,d))

    def predict(self, X):
        return np.argmax(X @ self.W.T, axis=1)
