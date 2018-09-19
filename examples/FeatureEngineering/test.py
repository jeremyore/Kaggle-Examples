from sklearn.datasets import load_iris

# 导入IRIS数据集
iris = load_iris()

# 特征矩阵
iris.data

# 目标向量
iris.target

from sklearn.preprocessing import StandardScaler

# 标准化，返回值为标准化后的数据
StandardScaler().fit_transform(iris.data)
