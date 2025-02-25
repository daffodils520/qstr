import numpy as np
import random
from tqdm import tqdm
from small_modeler.MLR.mlr_model import external_data_predicted, calculate_fitness_metrics, calculate_model_metrics, calculate_model_metrics_modeling,calculate_mlr_metrics_test, calculate_mlr_metrics_modeling
from small_modeler.MLR.XV import XV



def rand_int(min, max):
    return random.randint(min, max - 1)


# 适应度检查，检查是否收敛
def fitness_check_up(fitness_check, n_step):
    count = 0
    for i in range(n_step, n_step - 10, -1):
        if fitness_check[i] - fitness_check[i - 1] < 0.001:
            count += 1
    return count == 10


# 选择父代进行繁殖
def select_parents_for_reproduction(fitness, chromosomes, n_chromosome_selected, equation_length):
    selected_chromosomes = [[0] * equation_length for _ in range(n_chromosome_selected)]
    index_selected = bubble_sort_and_select_best(fitness, chromosomes, n_chromosome_selected)
    for i in range(n_chromosome_selected):
        for j in range(equation_length):
            selected_chromosomes[i][j] = chromosomes[index_selected[i]][j]
    return selected_chromosomes


# 交叉操作
def cross_over(selected_chromosomes, initial_n_chromosome, equation_length, n_chromosome_selected):
    crossover_chromosomes = [[0] * equation_length for _ in range(initial_n_chromosome)]
    for i in range(initial_n_chromosome):
        random1, random2 = 0, 0
        while random1 == random2:
            random1 = rand_int(0, n_chromosome_selected)
            random2 = rand_int(0, n_chromosome_selected)

        chromosome1 = selected_chromosomes[random1]
        chromosome2 = selected_chromosomes[random2]

        for j in range(equation_length):
            if j % 2 == 0:
                crossover_chromosomes[i][j] = chromosome1[j]
            else:
                crossover_chromosomes[i][j] = chromosome2[j]
    return crossover_chromosomes


# 检查冗余基因
def check_redundancy(value, all_values, index, equation_length):
    for i in range(equation_length):
        if value == all_values[index][i]:
            return False
    return True

# 排序并选择最好的染色体
def bubble_sort_and_select_best(fitness, chromosomes, n_chromosome_selected):
    index = list(range(len(fitness)))
    for i in range(len(fitness)):
        for j in range(1, len(fitness) - i):
            if fitness[j - 1] < fitness[j]:
                fitness[j - 1], fitness[j] = fitness[j], fitness[j - 1]
                index[j - 1], index[j] = index[j], index[j - 1]
    return index[:n_chromosome_selected]


# 染色体修复函数
def repair_chromosome(chromosome, n_features):
    rng = np.random.default_rng(42)
    unique_genes = list(set(chromosome))
    while len(unique_genes) < len(chromosome):
        unique_genes.append(rng.integers(0, n_features))
        unique_genes = list(set(unique_genes))
    return np.array(unique_genes[:len(chromosome)])

def normalize_data(train_matrix, test_matrix):
    """
    根据训练集的统计量归一化训练集和测试集
    """
    min_vals = np.min(train_matrix, axis=0)
    max_vals = np.max(train_matrix, axis=0)
    range_vals = max_vals - min_vals + 1e-8
    train_matrix_norm = (train_matrix - min_vals) / range_vals
    test_matrix_norm = (test_matrix - min_vals) / range_vals
    return train_matrix_norm, test_matrix_norm


def run_ga_on_combinations(combinations, X, y, initial_n_chromosome, n_chromosome_selected,
                           n_generations, equation_length, crossover_probability, mutation_probability,
                           X_modeling, X_modeling_pd, external_X):


    rng = np.random.default_rng(42)

    best_chromosomes_for_combos = []
    best_results_for_combos = []
    best_model_metrics = []
    results_to_save = []
    all_samples = X.shape[0]

    # 迭代每个组合
    for combo_index, combo in tqdm(enumerate(combinations), total=len(combinations), desc="Running GA on Combinations"):


        # 当前组合的测试集和训练集索引
        test_indices = list(combo)
        train_indices = [i for i in range(all_samples) if i not in test_indices]

        # 分割训练集和测试集
        X_numpy = X.to_numpy()
        combo_train_data = X_numpy[train_indices, :]
        combo_train_y = y[train_indices]
        combo_test_data = X_numpy[test_indices, :]
        combo_test_y = y[test_indices]

        # 数据归一化（基于训练集统计量）
        combo_train_data, combo_test_data = normalize_data(combo_train_data, combo_test_data)


        # 遗传算法初始化
        n_features = combo_train_data.shape[1]  # 预处理后特征数量
        chromosomes = np.array([rng.choice(n_features, size=equation_length, replace=False)
                                for _ in range(initial_n_chromosome)])


        # 初始化适应度
        fitness = [calculate_fitness_metrics(combo_train_data[:, chromo], combo_train_y) for chromo in chromosomes]
        fitness2 = [f['fitness2'] for f in fitness]
        best_fitness = -np.inf
        best_chromosome = None
        best_results = None
        best_model_metric = None
        best_test_metrics = None  # 全局列表存储所有组合的测试集结果
        best_result_modeling = None
        best_model_metrics_modeling = None
        best_model_metric_modeling_lmo = None

        # 遗传算法主循环
        for generation in range(n_generations):


            # 选择父代
            sorted_indices = np.argsort(fitness2)[::-1]

            # 选择适应度最高的 n_chromosome_selected 个染色体
            selected_chromosomes = chromosomes[sorted_indices[:n_chromosome_selected]]


            # 交叉操作
            crossover_chromosomes = np.zeros_like(chromosomes)  # 初始化交叉后的染色体
            for i in range(initial_n_chromosome):
                if rng.random() < crossover_probability:
                    random1, random2 = rng.integers(0, n_chromosome_selected, 2)  # 两个随机索引
                    chromosome1 = selected_chromosomes[random1]
                    chromosome2 = selected_chromosomes[random2]

                    crossover_mask = rng.random(equation_length) > 0.5
                    crossover_chromosomes[i] = np.where(crossover_mask, chromosome1, chromosome2)
                else:
                    crossover_chromosomes[i] = selected_chromosomes[i]


            # 变异操作
            mutated_chromosomes = crossover_chromosomes.copy()
            for i in range(initial_n_chromosome):
                if rng.random() < mutation_probability:
                    mutation_index = rng.integers(0, equation_length)
                    random_descriptor = rng.integers(0, n_features)
                    mutated_chromosomes[i][mutation_index] = random_descriptor

            # 修复染色体
            chromosomes = np.array([repair_chromosome(chromo, n_features) for chromo in mutated_chromosomes])

            # 重新计算适应度
            fitness = [calculate_fitness_metrics(combo_train_data[:, chromo], combo_train_y) for chromo in chromosomes]
            fitness2 = [f['fitness2'] for f in fitness]

            # 更新最佳染色体
            current_best_fitness = max(fitness2)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = chromosomes[np.argmax(fitness2)]
                best_result = fitness[np.argmax(fitness2)]
                best_model_metric = calculate_model_metrics(combo_train_data[:, best_chromosome], combo_train_y)
                best_model_metric_modeling = calculate_model_metrics(X_modeling[:, best_chromosome], y)
                best_model_metric_modeling_lmo = calculate_model_metrics_modeling(X_modeling[:, best_chromosome], y, m_value=3)

            # 检查是否收敛
            if fitness_check_up(fitness2, generation):
                print(f"Converged at generation {generation}")
                break


        best_chromosomes_for_combos.append(best_chromosome)
        best_results_for_combos.append(best_result)
        best_model_metrics.append(best_model_metric)

        best_chromosome_column_names = X.columns[best_chromosome].tolist()
        X_train_selected = combo_train_data[:, best_chromosome]
        X_test_selected = combo_test_data[:, best_chromosome]
        best_test_metrics = calculate_mlr_metrics_test(X_train_selected, combo_train_y, X_test_selected, combo_test_y)
        best_modeling_metrics = calculate_mlr_metrics_modeling(X_modeling[:, best_chromosome], y,
                                                               X_modeling[:, best_chromosome], y)
        y_predicted_test = best_test_metrics['y_predicted']
        y_predicted_modeling = best_modeling_metrics['y_predicted']
        best_test_metrics_test = XV(combo_train_y, combo_test_y, y_predicted_test)
        best_modeling_metrics_modeling = XV(X_modeling[:, best_chromosome], y, y_predicted_modeling)
        best_result_modeling = calculate_fitness_metrics(X_modeling[:, best_chromosome], y)
        print(best_chromosome)

        if external_X is not None:
            # 计算外部数据集的模型评估指标
            X_modeling_selected = X_modeling_pd.iloc[:, best_chromosome]
            external_y_predicted = external_data_predicted(external_X[best_chromosome_column_names], y, X_modeling_selected)
        else:
            # 如果没有外部数据集，设置默认值或跳过相关处理
            external_y_predicted = None

        print("最佳染色体对应的列名:", best_chromosome_column_names)
        result_entry = {
            "组合编号": combo_index + 1,
            "最佳染色体": best_chromosome,
            "最佳染色体对应的列名": best_chromosome_column_names,
            "R2(train)": best_result['R2'],
            "Q2_LOO(train)": best_model_metric['Q2_LOO'],
            "AvgRm^2LOO(train)": best_model_metric['scaled_avg_Rm2'],
            "MAE95(train)": best_result['MAE95Train'],
            "Q2F1(test)": best_test_metrics_test['Q2F1'],
            "Q2F2(test)": best_test_metrics_test['Q2F2'],
            "Q2F3(test)": best_test_metrics_test['Q2F3'],
            "AvgRm^2(test)": best_test_metrics_test['scaleAvgRm2Test'],
            "MAE95(test)": best_test_metrics_test['MAE95Test'],
            "R2(modeling)": best_result_modeling['R2'],
            "Q2_LOO(modeling)": best_model_metric_modeling['Q2_LOO'],
            "Q2_LMO(modeling)": best_model_metric_modeling_lmo['Q2_LMO'],
            "SDEP(modeling)": best_model_metric_modeling_lmo['SDEP'],
            "AvgRm^2LOO(modeling)": best_model_metric_modeling['scaled_avg_Rm2'],
            "MAE95(modeling)": best_modeling_metrics_modeling['MAE95Test'],
            "回归系数": best_modeling_metrics['coefficients'],
            "预测值": best_model_metric_modeling['y_predicted'],
        }
        if external_X is not None:
            result_entry["外部数据预测值"] = external_y_predicted
        # 将结果字典添加到结果列表中
        results_to_save.append(result_entry)

        print(best_test_metrics_test['Q2F1'])
        print(best_test_metrics_test['Q2F2'])
        print(best_test_metrics_test['Q2F3'])
        print(external_y_predicted)
    return results_to_save
