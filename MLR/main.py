from small_modeler.MLR.preprocessing import preprocess_data, preprocess_external_data
from small_modeler.MLR.genetic_algorithm import run_ga_on_combinations
import itertools

def main(file_path, output_path, combination_size, ga_params, external_dataset_path=None):
    if not file_path or not combination_size or not ga_params:
        raise ValueError("file_path, combination_size, and ga_params are required to run main().")

    #  加载和预处理数据
    X, y, X_modeling, X_modeling_pd = preprocess_data(file_path)
    if external_dataset_path:
        try:
            external_X = preprocess_external_data(external_dataset_path)
            # 根据需要，可以对外部数据集执行更多操作
        except Exception as e:
            raise ValueError(f"Error processing external dataset: {str(e)}")

    #  定义数据组合方式
    combination_size = int(combination_size)
    all_samples = X.shape[0]
    max_combinations = 1540
    combinations = list(itertools.combinations(range(all_samples), combination_size))
    combinations = combinations[:max_combinations]

    # 3. 遍历每个训练/测试集组合
    results_to_save = []

    # 3.1 运行遗传算法，筛选最佳特征
    # 提取 ga_params 中的参数
    initial_n_chromosome = ga_params.get('initial_n_chromosome', 100)  # 设置默认值
    n_chromosome_selected = ga_params.get('n_chromosome_selected', 30)
    n_generations = ga_params.get('n_generations', 100)
    equation_length = ga_params.get('equation_length', 4)
    crossover_probability = ga_params.get('crossover_probability', 1)
    mutation_probability = ga_params.get('mutation_probability', 0.3)


    result = run_ga_on_combinations(combinations, X, y, initial_n_chromosome, n_chromosome_selected,
                           n_generations, equation_length, crossover_probability, mutation_probability,
                            X_modeling, X_modeling_pd, external_X if external_dataset_path else None)
    results_to_save.append(result)

    return results_to_save