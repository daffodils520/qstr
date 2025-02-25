import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np




# 修复低方差处理函数
def remove_constant_features(data, constant_threshold=0.95, variance_threshold=0.0):
    """
    删除接近常量的特征和方差小于给定阈值的特征，支持 Pandas DataFrame 和 NumPy 数组。

    参数：
        data (pd.DataFrame 或 np.ndarray): 输入特征矩阵。
        constant_threshold (float): 常量特征的阈值，表示该特征值相同的比例超过该值时视为常量。
        variance_threshold (float): 方差阈值，表示方差小于此值的特征将被删除。

    返回：
        newMatrix (pd.DataFrame 或 np.ndarray): 去除接近常量和低方差特征后的数据。
        newMatrixT (pd.DataFrame 或 np.ndarray): 标准化后的矩阵（可能转置）。
        removed_descriptors (list): 被移除的描述符的名称或索引。
    """
    if isinstance(data, pd.DataFrame):  # 处理 Pandas DataFrame
        # 计算每个特征的最大值和最小值
        max_vals = data.max(axis=0)
        min_vals = data.min(axis=0)

        # 标准化
        nMatrix = (data - min_vals) / (max_vals - min_vals)

        # 计算每个特征的方差
        variances = nMatrix.var(axis=0)

        # 计算接近常量的特征（即大部分值都相同的特征）
        near_constant_mask = data.apply(lambda col: (col == col.iloc[0]).sum() / len(col) > constant_threshold)

        # 根据方差阈值删除特征
        low_variance_mask = variances <= variance_threshold

        # 组合两个条件的掩码：接近常量或方差小于阈值的特征
        final_mask = near_constant_mask | low_variance_mask

        # 输出接近常量或低方差的特征数
        print(f"接近常量或低方差的特征数: {np.sum(final_mask)}")

        # 返回去除接近常量或低方差特征后的数据
        newMatrix = data.loc[:, ~final_mask]
        newMatrixT = nMatrix.loc[:, ~final_mask]  # 可能的矩阵转置

    elif isinstance(data, np.ndarray):  # 处理 NumPy 数组
        # 计算每个特征的最大值和最小值
        max_vals = np.max(data, axis=0)
        min_vals = np.min(data, axis=0)

        # 标准化
        nMatrix = (data - min_vals) / (max_vals - min_vals)

        # 计算每个特征的方差
        variances = np.var(nMatrix, axis=0)

        # 计算接近常量的特征（即大部分值都相同的特征）
        near_constant_mask = np.apply_along_axis(lambda col: np.sum(col == col[0]) / len(col) > constant_threshold, axis=0, arr=data)

        # 根据方差阈值删除特征
        low_variance_mask = variances <= variance_threshold

        # 组合两个条件的掩码：接近常量或方差小于阈值的特征
        final_mask = near_constant_mask | low_variance_mask

        # 输出接近常量或低方差的特征数
        print(f"接近常量或低方差的特征数: {np.sum(final_mask)}")

        # 返回去除接近常量或低方差特征后的数据
        newMatrix = data[:, ~final_mask]
        newMatrixT = nMatrix[:, ~final_mask]  # 可能的矩阵转置

    return newMatrix


def remove_high_correlation_features(data, correlation_cutoff=0.99):
    # 计算特征的相关系数矩阵
    corr_matrix = data.corr().abs()

    # 提取上三角矩阵
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # 找到需要删除的特征
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_cutoff)]

    # 保留需要的特征
    to_keep = [col for col in data.columns if col not in to_drop]

    # 输出删除特征的数量
    print(f"删除了 {len(to_drop)} 个特征")

    # 返回处理后的数据集（仅返回 X，不返回 removed_descriptors）
    return data[to_keep]



def preprocess_data(file_path):

    # 加载数据集
    data = pd.read_csv(file_path)
    print(f"数据加载完成，数据维度: {data.shape}")

    # 提取特征变量
    print("提取特征变量...")
    X = data.iloc[:, 1:-1]  # 假设第一列为索引列，最后一列为目标变量
    y = data.iloc[:, -1].to_numpy()  # 目标变量
    print(f"特征矩阵维度: {X.shape}")

    # 数据预处理
    print("开始预处理...")
    X = remove_constant_features(X, constant_threshold=0.95)
    X = remove_high_correlation_features(X, correlation_cutoff=0.99)

    print(f"预处理完成，处理后特征矩阵维度: {X.shape}")

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    X_modeling_pd = pd.DataFrame(X_normalized, columns=X.columns)
    X_modeling = X_modeling_pd.to_numpy()

    return X, y, X_modeling, X_modeling_pd

def preprocess_external_data(file_path):
    # 加载数据集
    data = pd.read_csv(file_path)
    # 提取特征变量
    X = data.iloc[:, 1:]

    return X

