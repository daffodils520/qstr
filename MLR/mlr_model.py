from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from small_modeler.MLR.LOO import perform_loo_cv, perform_lmo_cv



def initialize_variables(Y, X):
    """初始化基本变量"""
    nComp = len(Y)  # 样本数
    nDesc = X.shape[1]  # 特征数
    nDescAct = nDesc + 1  # 实际特征数(含常数项)
    # 初始化结果变量
    results = {
        'r2PredLoo': 0.0,
        'SDEPLoo': 0.0,
        'scaleavgRm2Loo': 0.0,
        'scaledeltaRm2Loo': 0.0,
        'MAE100Train': 0.0,
        'SD100Train': 0.0,
        'MAE95Train': 0.0,
        'SD95Train': 0.0,
        'rangeTrain': 0.0,
        'fitnessScore': 0.0,
        'predictionQualityTrain': ""
    }
    return nComp, results

def calculate_mlr_metrics(X, Y):
    """计算多元线性回归的所有统计量"""
    # 基础数据初始化
    n_comp = len(Y)  # 样本数
    n_desc = X.shape[1]  # 特征数
    # 模型拟合
    model = LinearRegression()
    model.fit(X, Y)
    # 预测值计算
    y_calculated = model.predict(X)
    # 残差计算
    residuals = y_calculated - Y
    # 基础统计量计算
    r2 = r2_score(Y, y_calculated)
    r2_adjusted = 1 - (1 - r2) * (n_comp - 1) / (n_comp - n_desc - 1)
    press = np.sum(residuals ** 2)

    see = np.sqrt(press / (n_comp - n_desc - 1))
    mae100 = mean_absolute_error(Y, y_calculated)
    sd_y_calculated = np.std(y_calculated)
    range_train = np.max(Y) - np.min(Y)
    # 高级统计量计算

    sd100 = np.std(np.abs(Y - y_calculated))
    #constant_coeffi_values = np.concatenate(([model.intercept_], model.coef_))
    # F值计算
    ss_reg = np.sum((y_calculated - np.mean(Y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    f_value = (ss_reg / n_desc) / (ss_res / (n_comp - n_desc - 1))
    try:
        # 计算MSE
        mse = np.sum(residuals ** 2) / (n_comp - n_desc - 1)
        # 添加截距列
        X_with_intercept = np.column_stack([np.ones(n_comp), X])
        # 计算X'X
        XtX = X_with_intercept.T.dot(X_with_intercept)
        # 使用伪逆代替普通逆矩阵
        var_covar_matrix = mse * np.linalg.pinv(XtX)
        # 计算标准误差
        se_constant_coef = np.sqrt(np.abs(np.diag(var_covar_matrix)))
    except:
        # 如果计算失败，设置一个默认值
        se_constant_coef = np.zeros(n_desc + 1)
        print("警告：无法计算系数标准误差，已设置为0")
    # 返回结果字典
    results = {
        'r2': r2,
        'r2_adjusted': r2_adjusted,
        'f_value': f_value,
        'see': see,
        'press': press,
        'mae100': mae100,
        'sd_y_calculated': sd_y_calculated,
        'sd100': sd100,
        'constant': model.intercept_,
        'coefficients': model.coef_,
        'se_coefficients': se_constant_coef,
        'y_predicted': y_calculated,
        'residuals': residuals,
        'range_train': range_train
    }
    return results

def calculate_mlr_metrics_test(X, Y, X_test, test_y):
    """计算多元线性回归的所有统计量"""
    # 基础数据初始化
    n_comp = len(Y)  # 样本数
    n_desc = X.shape[1]  # 特征数
    # 模型拟合
    model = LinearRegression()
    model.fit(X, Y)
    # 预测值计算
    y_calculated = model.predict(X_test)
    # 残差计算
    residuals = y_calculated - test_y
    # 基础统计量计算
    r2 = r2_score(test_y, y_calculated)
    r2_adjusted = 1 - (1 - r2) * (n_comp - 1) / (n_comp - n_desc - 1)
    press = np.sum(residuals ** 2)

    see = np.sqrt(press / (n_comp - n_desc - 1))
    mae100 = mean_absolute_error(test_y, y_calculated)
    sd_y_calculated = np.std(y_calculated)
    range_train = np.max(test_y) - np.min(test_y)
    # 高级统计量计算

    sd100 = np.std(np.abs(test_y - y_calculated))
    #constant_coeffi_values = np.concatenate(([model.intercept_], model.coef_))
    # F值计算
    ss_reg = np.sum((y_calculated - np.mean(test_y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    f_value = (ss_reg / n_desc) / (ss_res / (n_comp - n_desc - 1))
    try:
        # 计算MSE
        mse = np.sum(residuals ** 2) / (n_comp - n_desc - 1)
        # 添加截距列
        X_with_intercept = np.column_stack([np.ones(n_comp), X])
        # 计算X'X
        XtX = X_with_intercept.T.dot(X_with_intercept)
        # 使用伪逆代替普通逆矩阵
        var_covar_matrix = mse * np.linalg.pinv(XtX)
        # 计算标准误差
        se_constant_coef = np.sqrt(np.abs(np.diag(var_covar_matrix)))
    except:
        # 如果计算失败，设置一个默认值
        se_constant_coef = np.zeros(n_desc + 1)
        print("警告：无法计算系数标准误差，已设置为0")
    # 返回结果字典
    results = {
        'r2': r2,
        'r2_adjusted': r2_adjusted,
        'f_value': f_value,
        'see': see,
        'press': press,
        'mae100': mae100,
        'sd_y_calculated': sd_y_calculated,
        'sd100': sd100,
        'constant': model.intercept_,
        'coefficients': model.coef_,
        'se_coefficients': se_constant_coef,
        'y_predicted': y_calculated,
        'residuals': residuals,
        'range_train': range_train
    }
    return results

def external_data_predicted(X, Y, X_modeling):


    # 模型拟合
    model = LinearRegression()
    model.fit(X_modeling, Y)
    # 预测值计算
    y_calculated = model.predict(X)

    return y_calculated

def calculate_mlr_metrics_modeling(X, Y, X_modeling, y):
    """计算多元线性回归的所有统计量"""
    # 基础数据初始化
    n_comp = len(Y)  # 样本数
    n_desc = X.shape[1]  # 特征数
    # 模型拟合
    model = LinearRegression()
    model.fit(X, Y)
    # 预测值计算
    y_calculated = model.predict(X_modeling)
    # 残差计算
    residuals = y_calculated - y
    # 基础统计量计算
    r2 = r2_score(y, y_calculated)
    r2_adjusted = 1 - (1 - r2) * (n_comp - 1) / (n_comp - n_desc - 1)
    press = np.sum(residuals ** 2)

    see = np.sqrt(press / (n_comp - n_desc - 1))
    mae100 = mean_absolute_error(y, y_calculated)
    sd_y_calculated = np.std(y_calculated)
    range_train = np.max(y) - np.min(y)
    # 高级统计量计算

    sd100 = np.std(np.abs(y - y_calculated))
    constant_coeffi_values = np.concatenate(([model.intercept_], model.coef_))
    # F值计算
    ss_reg = np.sum((y_calculated - np.mean(y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    f_value = (ss_reg / n_desc) / (ss_res / (n_comp - n_desc - 1))
    try:
        # 计算MSE
        mse = np.sum(residuals ** 2) / (n_comp - n_desc - 1)
        # 添加截距列
        X_with_intercept = np.column_stack([np.ones(n_comp), X])
        # 计算X'X
        XtX = X_with_intercept.T.dot(X_with_intercept)
        # 使用伪逆代替普通逆矩阵
        var_covar_matrix = mse * np.linalg.pinv(XtX)
        # 计算标准误差
        se_constant_coef = np.sqrt(np.abs(np.diag(var_covar_matrix)))
    except:
        # 如果计算失败，设置一个默认值
        se_constant_coef = np.zeros(n_desc + 1)
        print("警告：无法计算系数标准误差，已设置为0")
    # 返回结果字典
    results = {
        'r2': r2,
        'r2_adjusted': r2_adjusted,
        'f_value': f_value,
        'see': see,
        'press': press,
        'mae100': mae100,
        'sd_y_calculated': sd_y_calculated,
        'sd100': sd100,
        'constant': model.intercept_,
        'coefficients': model.coef_,
        'se_coefficients': se_constant_coef,
        'y_predicted': y_calculated,
        'residuals': residuals,
        'range_train': range_train
    }
    return results


def calculate_basic_stats(Y, residuals, absResidual):
    """计算基本统计量"""
    MAE100Train = np.mean(absResidual)
    SD100Train = np.std(absResidual)
    PRESS = np.sum(residuals ** 2)
    TSS = np.sum((Y - np.mean(Y)) ** 2)
    r2PredLoo = 1 - PRESS/TSS
    SDEPLoo = np.sqrt(PRESS/len(Y))
    return MAE100Train, SD100Train, r2PredLoo, SDEPLoo

def calculate_basic_stats_lmo(Y, residuals_lmo, absResidual_lmo):
    """计算基本统计量"""
    MAE100modeling = np.mean(absResidual_lmo)
    SD100modeling = np.std(absResidual_lmo)
    PRESS = np.sum(residuals_lmo ** 2)
    TSS = np.sum((Y - np.mean(Y)) ** 2)
    r2PredLmo = 1 - PRESS/TSS
    SDEPLmo = np.sqrt(PRESS/len(Y))
    return MAE100modeling, SD100modeling, r2PredLmo, SDEPLmo

def calculate_scaled_metrics(Y, yCalculatedLoo):
    """计算缩放指标"""
    y_max = np.max(Y)
    y_min = np.min(Y)
    rangeTrain = y_max - y_min
    scaleYObs = (Y - y_min) / (y_max - y_min)
    scaleYPred = (yCalculatedLoo - y_min) / (y_max - y_min)
    scaleavgYObs = np.mean(scaleYObs)
    scaleavgYPred = np.mean(scaleYPred)
    scaleRYPredYObs = np.sum((scaleYPred - scaleavgYPred) * (scaleYObs - scaleavgYObs))
    scaleyPredYBar2 = np.sum((scaleYPred - scaleavgYPred) ** 2)
    scaleyObsYBar2 = np.sum((scaleYObs - scaleavgYObs) ** 2)
    scaler2 = (scaleRYPredYObs / np.sqrt(scaleyPredYBar2 * scaleyObsYBar2)) ** 2
    return rangeTrain, scaler2


def calculate_r2(y_true, y_pred):
    """计算R²"""
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_scaled_rm2_metrics(Y, yCalculatedLoo):
    """计算scaleavgRm2Loo和scaledeltaRm2Loo"""
    # 数据缩放
    y_max = np.max(Y)
    y_min = np.min(Y)
    scaleYObs = (Y - y_min) / (y_max - y_min)
    scaleYPred = (yCalculatedLoo - y_min) / (y_max - y_min)
    # 计算r2和r2'
    r2 = calculate_r2(scaleYObs, scaleYPred)
    r2_prime = calculate_r2(scaleYPred, scaleYObs)
    # 计算avgRm2和deltaRm2
    scaleavgRm2Loo = (r2 + r2_prime) / 2
    scaledeltaRm2Loo = abs(r2 - r2_prime)
    return scaleavgRm2Loo, scaledeltaRm2Loo

def calculate_scaled_rm2_metrics_lmo(Y, yCalculatedLmo):
    """计算scaleavgRm2Loo和scaledeltaRm2Loo"""
    # 数据缩放
    y_max = np.max(Y)
    y_min = np.min(Y)
    scaleYObs = (Y - y_min) / (y_max - y_min)
    scaleYPred = (yCalculatedLmo - y_min) / (y_max - y_min)
    # 计算r2和r2'
    r2 = calculate_r2(scaleYObs, scaleYPred)
    r2_prime = calculate_r2(scaleYPred, scaleYObs)
    # 计算avgRm2和deltaRm2
    scaleavgRm2Lmo = (r2 + r2_prime) / 2
    scaledeltaRm2Lmo = abs(r2 - r2_prime)
    return scaleavgRm2Lmo, scaledeltaRm2Lmo


def calculate_95_percentile_stats(absResidual, Y, yCalculatedLoo, nComp):
    """计算95%统计量"""
    sorted_indices = np.argsort(absResidual)
    nComp95 = int(np.ceil(0.95 * nComp))
    abs_residual_95 = absResidual[sorted_indices[:nComp95]]
    Y_95 = Y[sorted_indices[:nComp95]]
    yCalculatedLoo_95 = yCalculatedLoo[sorted_indices[:nComp95]]
    MAE95Train = np.mean(abs_residual_95)
    SD95Train = np.std(np.abs(Y_95 - yCalculatedLoo_95))
    return MAE95Train, SD95Train

def calculate_95_percentile_stats_modelilng(absResidual_lmo, Y, yCalculatedLmo, nComp):
    """计算95%统计量"""
    sorted_indices = np.argsort(absResidual_lmo)
    nComp95 = int(np.ceil(0.95 * nComp))
    abs_residual_95 = absResidual_lmo[sorted_indices[:nComp95]]
    Y_95 = Y[sorted_indices[:nComp95]]
    yCalculatedLmo_95 = yCalculatedLmo[sorted_indices[:nComp95]]
    MAE95modeling = np.mean(abs_residual_95)
    SD95modeling = np.std(np.abs(Y_95 - yCalculatedLmo_95))
    return MAE95modeling, SD95modeling


def evaluate_prediction_quality(MAE95Train, SD95Train, rangeTrain):
    """评估预测质量"""
    condition1Good = 0.1 * rangeTrain
    condition1Moderate = 0.15 * rangeTrain
    condition2Good = 0.2 * rangeTrain
    condition2Moderate = 0.25 * rangeTrain
    if (MAE95Train <= condition1Good and MAE95Train + 3.0 * SD95Train <= condition2Good):
        predictionQualityTrain = "GOOD"
    elif (MAE95Train > condition1Moderate or MAE95Train + 3.0 * SD95Train > condition2Moderate):
        predictionQualityTrain = "BAD"
    else:
        predictionQualityTrain = "MODERATE"
    fitnessScore = ((0.15 * rangeTrain - MAE95Train) / (0.15 * rangeTrain) +
                    (0.25 * rangeTrain - MAE95Train + 3.0 * SD95Train) / (0.25 * rangeTrain))
    return predictionQualityTrain, fitnessScore





def calculate_model_metrics(X, Y):
    """主函数：计算所有模型指标"""
    # 初始化变量
    nComp, results = initialize_variables(Y, X)

    # 执行LOO交叉验证
    yCalculatedLoo, yCalculatedLooCopy, residuals, absResidual = perform_loo_cv(X, Y, nComp)

    # 计算基本统计量
    MAE100Train, SD100Train, r2PredLoo, SDEPLoo = calculate_basic_stats(Y, residuals, absResidual)
    MAE100modeling, SD100modeling, r2PredLmo, SDEPLmo = calculate_basic_stats(Y, residuals, absResidual)

    # 计算缩放指标
    rangeTrain, scaler2 = calculate_scaled_metrics(Y, yCalculatedLoo)

    # 计算scaleavgRm2Loo和scaledeltaRm2Loo
    scaleavgRm2Loo, scaledeltaRm2Loo = calculate_scaled_rm2_metrics(Y, yCalculatedLoo)

    # 计算95%统计量
    MAE95Train, SD95Train = calculate_95_percentile_stats(absResidual, Y, yCalculatedLoo, nComp)

    # 评估预测质量
    predictionQualityTrain, fitnessScore = evaluate_prediction_quality(MAE95Train, SD95Train, rangeTrain)

    # 整合所有结果
    results = {
        'Q2_LOO': r2PredLoo,
        'SDEP': SDEPLoo,
        'MAE100': MAE100Train,
        'SD100': SD100Train,
        'MAE95': MAE95Train,
        'SD95': SD95Train,
        'range_train': rangeTrain,
        'fitness_score': fitnessScore,
        'prediction_quality': predictionQualityTrain,
        'y_predicted': yCalculatedLooCopy,
        'scaled_avg_Rm2': scaleavgRm2Loo,
        'scaled_delta_Rm2': scaledeltaRm2Loo,
        'SDEP_LMO': SDEPLmo,
    }
    return results

def calculate_model_metrics_modeling(x_modeling, y, m_value = 3):
    """主函数：计算所有模型指标"""
    # 初始化变量
    nComp, results = initialize_variables(y, x_modeling)
    # 执行LOO交叉验证
    yCalculatedLmo, yCalculatedLmoCopy, residuals_lmo, absResidual_lmo = perform_lmo_cv(x_modeling, y, m_value)

    # 计算基本统计量
    MAE100modeling, SD100modeling, r2PredLmo, SDEPLmo = calculate_basic_stats(y, residuals_lmo, absResidual_lmo)
    # 计算缩放指标
    rangemodeling, scaler2 = calculate_scaled_metrics(y, yCalculatedLmo)
    # 计算scaleavgRm2Loo和scaledeltaRm2Loo
    scaleavgRm2Lmo, scaledeltaRm2Lmo = calculate_scaled_rm2_metrics_lmo(y, yCalculatedLmo)

    # 计算95%统计量
    MAE95modeling, SD95modeling = calculate_95_percentile_stats_modelilng(absResidual_lmo, y, yCalculatedLmo, nComp)
    # 评估预测质量
    predictionQualitymodeling, fitnessScore = evaluate_prediction_quality(MAE95modeling, SD95modeling, rangemodeling)
    # 整合所有结果
    results = {
        'Q2_LMO': r2PredLmo,
        'SDEP': SDEPLmo,
        'MAE100': MAE100modeling,
        'SD100': SD100modeling,
        'MAE95': MAE95modeling,
        'SD95': SD95modeling,
        'range_train': rangemodeling,
        'fitness_score': fitnessScore,
        'prediction_quality': predictionQualitymodeling,
        'y_predicted': yCalculatedLmoCopy,
        'scaled_avg_Rm2': scaleavgRm2Lmo,
        'scaled_delta_Rm2': scaledeltaRm2Lmo,
        'SDEP_LMO': SDEPLmo
    }
    return results



def calculate_fitness_metrics(X, Y):
    """计算所有适应度指标"""
    # 1. 执行留一交叉验证 (LOO)
    loo_results = calculate_model_metrics(X, Y)
   # loo_results_modeling = calculate_model_metrics(X_modeling, y, X, Y, m_value)
    # 2. 从LOO计算结果中获取指标
    Q2 = loo_results['Q2_LOO']
    SDEP = loo_results['SDEP']
    scaleAvgRm2LOO = loo_results['scaled_avg_Rm2']
   # scaleAvgRm2LOO_modeling = loo_results_modeling['scaled_avg_Rm2_modeling']
    scaleDeltaRm2LOO = loo_results['scaled_delta_Rm2']
    MAE95Train = loo_results['MAE95']
    SD95Train = loo_results['SD95']
    # 3. 执行多元线性回归 (MLR)
    mlr_results = calculate_mlr_metrics(X, Y)
    R2 = mlr_results['r2']
    r2Adjusted = mlr_results['r2_adjusted']
    SEE = mlr_results['see']
    Constant = mlr_results['constant']
    coeffValues = mlr_results['coefficients']
    SECoef = mlr_results['se_coefficients']
    PRESS = mlr_results['press']
    rangeTrain = mlr_results['range_train']
    # 4. 计算fitness得分
    model = LinearRegression()
    model.fit(X, Y)
    predictions = model.predict(X)
    mse = np.mean((predictions - Y) ** 2)
    fitness2 = -mse
    # 6. 整合结果
    results = {
        'PRESS': PRESS,
        'fitness2': fitness2,
        'Q2': Q2,
        'SDEP': SDEP,
        'R2': R2,
        'r2Adjusted': r2Adjusted,
        'SEE': SEE,
        'Constant': Constant,
        'coeffValues': coeffValues,
        'SECoef': SECoef,
        'scaleAvgRm2LOO': scaleAvgRm2LOO,
        'scaleDeltaRm2LOO': scaleDeltaRm2LOO,
        'MAE95Train': MAE95Train,
        'SD95Train': SD95Train,
        'rangeTrain': rangeTrain,
    }
    return results



