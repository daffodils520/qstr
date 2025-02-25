from flask import Flask, render_template, request, jsonify
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from tkinter import Tk, filedialog
from flask_cors import CORS
import logging
from small_modeler.MLR.main import main
from flask import session
import pandas as pd



app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量
GA_PARAMS = {}

@app.route('/')
def index():
    """
    加载前端界面
    """
    return render_template('index.html')

@app.route('/settings')
def settings():
    """
    渲染遗传算法设置页面
    """
    return render_template('settings.html')

@app.route('/save_settings', methods=['POST'])
def save_settings():
    """
    保存 GA 参数
    """
    try:
        global GA_PARAMS

        # 保存 GA 参数
        GA_PARAMS = {
            "iterations": int(request.json.get('iterations', 100)),
            "equation_length": int(request.json.get('eq_length', 4)),
            "crossover_probability": float(request.json.get('crossover_prob', 1.0)),
            "mutation_probability": float(request.json.get('mutation_prob', 0.3)),
            "initial_equations": int(request.json.get('initial_eq', 100)),
            "number of equationsselected in each generation": int(request.json.get('n_eq_selected_each_gene', 30)),
        }
        return jsonify({"message": "Settings saved successfully!"})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 400


@app.route('/submit_combination_size', methods=['POST'])
def submit_combination_size():
    global combination_size
    if request.is_json:
        data = request.get_json()
        combination_size = data.get('combination_size')
    else:
        return jsonify({"message": "Invalid content type. Expected JSON."}), 415

    if not combination_size:
        return jsonify({"message": "combination_size is required"}), 400

    session['combination_size'] = combination_size
    return jsonify({"message": "Combination size received successfully!", "combination_size": combination_size}), 200

@app.route('/run', methods=['POST'])
def run_model():
    try:
        # 获取上传的文件
        file = request.files.get('file')

        if not file or not allowed_file(file.filename):
            return jsonify({"message": "Invalid file format. Please upload a CSV file."}), 400
        # 保存上传的文件
        filename = secure_filename(file.filename)
        dataset_path = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(dataset_path)

        # 获取外部数据集文件（可选）
        external_file = request.files.get('external_file')  # 外部数据集文件
        external_dataset_path = None  # 初始化路径为 None
        if external_file and allowed_file(external_file.filename):
            external_filename = secure_filename(external_file.filename)
            external_dataset_path = Path(app.config['UPLOAD_FOLDER']) / external_filename
            external_file.save(external_dataset_path)



        # 获取输出文件夹路径
        output_folder = request.form.get('output_folder')
        if not output_folder:
            return jsonify({"message": "No output folder selected."}), 400

        output_path = Path(output_folder).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取 combination_size
        combination_size = session.get('combination_size')
        if not combination_size:
            return jsonify({"message": "Combination size is required"}), 400

        # 运行主模型逻辑
        if external_dataset_path:
            results_to_save = main(dataset_path, output_path, combination_size, GA_PARAMS, external_dataset_path)
        else:
            results_to_save = main(dataset_path, output_path, combination_size, GA_PARAMS)

        results_df = pd.DataFrame(results_to_save)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Results saved to {output_path}")

        results_file = output_path / "results.csv"


        return jsonify({
            "message": "Model ran successfully!",
            "data": {
                "results_file": str(results_file),
                "dataset_path": str(dataset_path),
                "external_dataset_path": str(external_dataset_path),
                "output_folder": str(output_path),
            }
        })
    except Exception as model_error:
        return jsonify({"message": f"Error during model run: {str(model_error)}"}), 500
    except Exception as e:
        return jsonify({"message": f"Error occurred: {str(e)}"}), 500


@app.route("/select_folder", methods=["POST"])
def select_folder():
    try:
        # 使用 Tkinter 打开文件夹选择对话框
        root = Tk()
        root.withdraw()  # 隐藏主窗口
        root.attributes('-topmost', True)  # 确保窗口在最前
        folder_path = filedialog.askdirectory(title="Select Output Folder")  # 打开文件夹选择对话框
        root.destroy()
        print(f"Selected folder path: {folder_path}")
        if folder_path:
            return jsonify({"folderPath": folder_path})
        else:
            return jsonify({"folderPath": None})  # 用户取消选择
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def allowed_file(filename):
    """
    检查文件类型
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run(debug=True)


