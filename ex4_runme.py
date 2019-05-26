"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author: Lior Paz
Date: May, 2019

"""
import numpy as np
from ex4_tools import DecisionStump, decision_boundaries, generate_data
import matplotlib.pyplot as plt
from adaboost import AdaBoost
from perceptron import Perceptron
from sklearn.svm import SVC

mean = [0, 0]
cov = np.eye(2)
m_num = [5, 10, 15, 25, 70]


def Q4():
    for m in m_num:
        class_check = True
        X = None
        y = None
        while class_check:
            X = np.random.multivariate_normal(mean, cov, m)
            y = np.sign(np.array([0.3, -0.5]) @ X.T + 0.1)
            if np.abs(np.sum(y)) == m:
                class_check = True
            else:
                class_check = False
        plt.scatter(X.T[0], X.T[1], c=y)
        pts = np.linspace(-3.5, 3.5, 1000)
        plt.plot(pts, 0.6 * pts + 0.2, label='True')
        perceptron = Perceptron()
        perceptron.fit(X, y)
        w_perceptron = perceptron.w
        plt.plot(pts, -(w_perceptron[0] * pts + w_perceptron[2]) / w_perceptron[1], label='Perceptron')
        clf = SVC(C=1e10, kernel='linear')
        clf.fit(X, y)
        w_clf = clf.coef_[0]
        plt.plot(pts, -(w_clf[0] * pts + clf.intercept_[0]) / w_clf[1], label='SVM')
        plt.legend()
        plt.title('Hyperplane Classification for %i Samples' % m)
        plt.show()


def Q5():
    svm_err_m = np.array([])
    perceptron_err_m = np.array([])
    for m in m_num:
        svm_err = 0
        perceptron_err = 0
        for k in range(500):
            class_check = True
            X_train = None
            y_train = None
            while class_check:
                X_train = np.random.multivariate_normal(mean, cov, m)
                y_train = np.sign(np.array([0.3, -0.5]) @ X_train.T + 0.1)
                if np.abs(np.sum(y_train)) == m:
                    class_check = True
                else:
                    class_check = False
            X_test = np.random.multivariate_normal(mean, cov, 10000)
            y_test = np.sign(np.array([0.3, -0.5]) @ X_test.T + 0.1)
            perceptron = Perceptron()
            perceptron.fit(X_train, y_train)
            clf = SVC(C=1e10, kernel='linear')
            clf.fit(X_train, y_train)
            y_test_svm = clf.predict(X_test)
            y_test_perception = np.apply_along_axis(perceptron.predict, 1, X_test)
            svm_err += 1 - np.argwhere((y_test + y_test_svm) == 0).size / y_test.size
            perceptron_err += 1 - np.argwhere((y_test + y_test_perception) == 0).size / y_test.size
        svm_err_m = np.append(svm_err_m, svm_err / 500)
        perceptron_err_m = np.append(perceptron_err_m, perceptron_err / 500)
    plt.plot(np.array(m_num), perceptron_err_m, label='Perceptron')
    plt.plot(np.array(m_num), svm_err_m, label='SVM')
    plt.legend()
    plt.title('SVM against Perceptron with Different Train Set Size ')
    plt.xlabel('m')
    plt.ylabel('accuracy')
    plt.show()


def Q_adaboost(noise_ratio):
    X_train, y_train = generate_data(5000, noise_ratio)
    classifier = AdaBoost(DecisionStump, 500)
    classifier.train(X_train, y_train)
    X_test, y_test = generate_data(200, noise_ratio)
    vals = np.arange(1, 501)
    plt.plot(vals, [classifier.error(X_train, y_train, t) for t in vals], label='Training Error', lw=1, alpha=0.6)
    plt.plot(vals, [classifier.error(X_test, y_test, t) for t in vals], label='Test Error', lw=1, alpha=0.6)
    plt.legend()
    plt.title(f'Adaboost Training & Test Error according to T, noise={noise_ratio}')
    plt.show()
    boosts = [5, 10, 50, 100, 200, 500]
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        decision_boundaries(classifier, X_test, y_test, boosts[i])
        plt.title(f'T={boosts[i]}, noise={noise_ratio}')
    plt.show()
    test_errors = [classifier.error(X_test, y_test, t) for t in vals]
    min_t = np.argmin(test_errors)
    min_err = test_errors[min_t]
    # print(min_t, min_err)
    decision_boundaries(classifier, X_train, y_train, min_t)
    plt.title(f'min test_err {min_err} T={min_t} noise {noise_ratio}')
    plt.show()
    decision_boundaries(classifier, X_train, y_train, 499, classifier.D_of_last_iteration)
    plt.title(f'un-normalized weighed sample T=500, noise={noise_ratio}')
    plt.show()
    decision_boundaries(classifier, X_train, y_train, 499,
                        classifier.D_of_last_iteration / np.max(classifier.D_of_last_iteration) * 100)
    plt.title(f'normalized weighed sample T=500, noise={noise_ratio}')
    plt.show()


def Q8_11():
    Q_adaboost(0)


def Q12():
    Q_adaboost(0.01)
    Q_adaboost(0.4)


if __name__ == '__main__':
    Q4()
    Q5()
    Q8_11()
    Q12()
