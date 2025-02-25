import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit



def perform_loo_cv(X, Y, nComp):
    """执行留一交叉验证"""
    loo = LeaveOneOut()
    yCalculatedLoo = np.zeros(nComp)
    yCalculatedLooCopy = np.zeros(nComp)
    residuals = np.zeros(nComp)
    absResidual = np.zeros(nComp)
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        yCalculatedLoo[test_idx] = pred
        yCalculatedLooCopy[test_idx] = pred
        residuals[test_idx] = pred - y_test
        absResidual[test_idx] = np.abs(residuals[test_idx])
    return yCalculatedLoo, yCalculatedLooCopy, residuals, absResidual



def perform_lmo_cv(x_modeling, y_modeling, m_value):
    # 初始化变量
    total_samples = len(y_modeling)
    yCalculatedLmo = np.zeros(total_samples)  # 存储 LMO 的预测值
    yCalculatedLmoCopy = np.zeros(total_samples)  # 副本（可以替换为其他方法预测值）
    residuals_lmo = np.zeros(total_samples)  # 残差
    absResidual_lmo = np.zeros(total_samples)  # 绝对残差

    # 创建 ShuffleSplit 进行分组
    rs = ShuffleSplit(n_splits=10, test_size=m_value / total_samples, random_state=42)

    for modeling_idx, valid_idx in rs.split(x_modeling):
        # 分割训练集和测试集
        X_modeling, X_valid = x_modeling[modeling_idx], x_modeling[valid_idx]
        Y_modeling, Y_valid = y_modeling[modeling_idx], y_modeling[valid_idx]

        # 构建模型并训练
        model = LinearRegression()
        model.fit(X_modeling, Y_modeling)

        # 对测试集进行预测
        Y_pred = model.predict(X_valid)

        # 保存预测值及残差信息
        yCalculatedLmo[valid_idx] = Y_pred
        yCalculatedLmoCopy[valid_idx] = Y_pred  # 这里是副本
        residuals_lmo[valid_idx] = Y_pred - Y_valid
        absResidual_lmo[valid_idx] = np.abs(residuals_lmo[valid_idx])

    # 返回结果
    return yCalculatedLmo, yCalculatedLmoCopy, residuals_lmo, absResidual_lmo
