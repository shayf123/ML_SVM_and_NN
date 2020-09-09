#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""


# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    linear_kernel = svm.SVC(kernel="linear", C=1000)
    linear_kernel.fit(X_train, y_train)
    create_plot(X_train, y_train, linear_kernel)
    plt.title("linear_kernel")
    # plt.show()

    quardatic_kernel = svm.SVC(kernel="poly", degree=2, C=1000)
    quardatic_kernel.fit(X_train, y_train)
    create_plot(X_train, y_train, quardatic_kernel)
    plt.title("quardatic_kernel")
    # plt.show()

    rbf_kernel = svm.SVC(kernel="rbf", C=1000)
    rbf_kernel.fit(X_train, y_train)
    create_plot(X_train, y_train, rbf_kernel)
    plt.title("rbf_kernel")
    # plt.show()
    return np.array([linear_kernel.n_support_, quardatic_kernel.n_support_, rbf_kernel.n_support_])


# X, Y, W, Z = get_points()
# train_three_kernels(X, Y, W, Z)


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C_lst = [10 ** i for i in range(-5, 6)]
    train_accuracy = []
    validation_accuracy = []
    for C in C_lst:
        linear_kernel = svm.SVC(kernel="linear", C=C)
        linear_kernel.fit(X_train, y_train)
        train_accuracy.append(linear_kernel.score(X_train, y_train))
        validation_accuracy.append(linear_kernel.score(X_val, y_val))
        # create_plot(X_val, y_val, linear_kernel)
        # plt.title("C = " + str(C))
        # plt.show()

    plt.plot(C_lst, train_accuracy, label='train_accuracy')
    plt.plot(C_lst, validation_accuracy, label="validation_accuracy")
    plt.legend()
    plt.xscale("log")
    plt.title("accuracy on train and validation")
    # plt.show()
    return np.array(validation_accuracy)


# X, Y, W, Z = get_points()
# linear_accuracy_per_C(X, Y, W, Z)


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    G_lst = [10 ** i for i in range(-5, 6)]
    train_accuracy = []
    validation_accuracy = []
    for g in G_lst:
        rbf_kernel = svm.SVC(kernel="rbf", C=10, gamma=g)
        rbf_kernel.fit(X_train, y_train)
        train_accuracy.append(rbf_kernel.score(X_train, y_train))
        validation_accuracy.append(rbf_kernel.score(X_val, y_val))
        create_plot(X_val, y_val, rbf_kernel)
        plt.title("gamma = " + str(g))
        # plt.show()

    plt.plot(G_lst, train_accuracy, label='train_accuracy')
    plt.plot(G_lst, validation_accuracy, label="validation_accuracy")
    plt.legend()
    plt.xscale("log")
    plt.title("accuracy on train and validation")
    # plt.show()

    return np.array(validation_accuracy)

# X, Y, W, Z = get_points()
# rbf_accuracy_per_gamma(X, Y, W, Z)
