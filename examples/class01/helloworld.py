import numpy as np
import time
from sklearn.datasets import make_classification

X, y = make_classification(1000, n_features=20, n_informative=2, n_redundant=2, n_classes=2, random_state=0)

from pandas import DataFrame

df = DataFrame(np.hstack((X, y[:, None])), columns=list(range(20)) + ["class"])

import matplotlib.pyplot as plt
import seaborn as sns
#
# _ = sns.pairplot(df[:50], vars=[8, 11, 12, 14, 19], hue="class", size=2.5)
# plt.show()

# plt.figure(figsize=(12, 10))
# _ = sns.corrplot(df, annot=False)
# plt.show()

from sklearn.svm import LinearSVC
from sklearn.learning_curve import learning_curve
#绘制学习曲线，以确定模型的状况
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    """

    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on")
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()


# 少样本的情况情况下绘出学习曲线
# plot_learning_curve(LinearSVC(C=0.1), "LinearSVC(C=0.1)",
#                     X[:, [11, 14]], y, ylim=(0.8, 1.01),
#                     train_sizes=np.linspace(.05, 0.2, 5))


# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import SelectKBest, f_classif
# # SelectKBest(f_classif, k=2) 会根据Anova F-value选出 最好的k=2个特征
# plot_learning_curve(Pipeline([("fs", SelectKBest(f_classif, k=2)), # select two features
#                                ("svc", LinearSVC(C=10.0))]), "SelectKBest(f_classif, k=2) + LinearSVC(C=10.0)", X, y, ylim=(0.8, 1.0), train_sizes=np.linspace(.05, 0.2, 5))

# from sklearn.grid_search import GridSearchCV
# estm=GridSearchCV(LinearSVC(),param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]})
#
# plot_learning_curve(estm, "LinearSVC(C=AUTO)",
#                     X, y, ylim=(0.8, 1.0),
#                     train_sizes=np.linspace(.05, 0.2, 5))
# print ("Chosen parameter on 100 datapoints: %s" % estm.fit(X[:500], y[:500]).best_params_)

# plot_learning_curve(LinearSVC(C=0.1, penalty='l1', dual=False), "LinearSVC(C=0.1, penalty='l1')", X, y, ylim=(0.8, 1.0), train_sizes=np.linspace(.05, 0.2, 5))
# estm = LinearSVC(C=0.1, penalty='l1', dual=False)
# estm.fit(X[:450], y[:450])  # 用450个点来训练
# print ("Coefficients learned: %s" % estm.coef_)
# print ("Non-zero coefficients: %s" % np.nonzero(estm.coef_)[1])

#构造一份环形数据
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, random_state=2)
#绘出学习曲线
# plot_learning_curve(LinearSVC(C=0.25),"LinearSVC(C=0.25)",X, y, ylim=(0.5, 1.0),train_sizes=np.linspace(.1, 1.0, 5))

# f = DataFrame(np.hstack((X, y[:, None])), columns = list(range(2)) + ["class"])
# _ = sns.pairplot(df, vars=[0, 1], hue="class", size=3.5)
# plt.show()

# 加入原始特征的平方项作为新特征
# X_extra = np.hstack((X, X[:, [0]]**2 + X[:, [1]]**2))
#
# plot_learning_curve(LinearSVC(C=0.25), "LinearSVC(C=0.25) + distance feature", X_extra, y, ylim=(0.5, 1.0), train_sizes=np.linspace(.1, 1.0, 5))

# from sklearn.svm import SVC
# # note: we use the original X without the extra feature
# plot_learning_curve(SVC(C=2.5, kernel="rbf", gamma=1.0), "SVC(C=2.5, kernel='rbf', gamma=1.0)",X, y, ylim=(0.5, 1.0), train_sizes=np.linspace(.1, 1.0, 5))

#生成大样本，高纬度特征数据
# X, y = make_classification(200000, n_features=200, n_informative=25, n_redundant=0, n_classes=10, class_sep=2, random_state=0)
#
# #用SGDClassifier做训练，并画出batch在训练前后的得分差
# from sklearn.linear_model import SGDClassifier
# est = SGDClassifier(penalty="l2", alpha=0.001)
# progressive_validation_score = []
# train_score = []
# for datapoint in range(0, 199000, 1000):
#     X_batch = X[datapoint:datapoint+1000]
#     y_batch = y[datapoint:datapoint+1000]
#     if datapoint > 0:
#         progressive_validation_score.append(est.score(X_batch, y_batch))
#     est.partial_fit(X_batch, y_batch, classes=range(10))
#     if datapoint > 0:
#         train_score.append(est.score(X_batch, y_batch))
#
# plt.plot(train_score, label="train score")
# plt.plot(progressive_validation_score, label="progressive validation score")
# plt.xlabel("Mini-batch")
# plt.ylabel("Score")
# plt.legend(loc='best')
# plt.show()


#直接从sklearn中load数据集
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
print ("Dataset consist of %d samples with %d features each" % (n_samples, n_features))

# 绘制数字示意图
# n_img_per_row = 20
# img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
# for i in range(n_img_per_row):
#     ix = 10 * i + 1
#     for j in range(n_img_per_row):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))
#
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# _ = plt.title('A selection from the 8*8=64-dimensional digits dataset')
# plt.show()


#import所需的package
from sklearn import (manifold, decomposition, random_projection)
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)

#定义绘图函数
from matplotlib import offsetbox
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 12})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
        plt.show()

# from sklearn import (manifold, decomposition, random_projection)
# #TruncatedSVD 是 PCA的一种实现
# X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
# #记录时间
# start_time = time.time()
#
# plot_embedding(X_pca,"Principal Components projection of the digits (time: %.3fs)" % (time.time() - start_time))

from sklearn import (manifold, decomposition, random_projection)
#降维
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
start_time = time.time()
X_tsne = tsne.fit_transform(X)
#绘图
plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time: %.3fs)" % (time.time() - start_time))
