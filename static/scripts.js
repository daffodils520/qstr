<script>
// 获取模态窗口和按钮
const settingsButton = document.getElementById("settingsButton");
const modal = document.getElementById("geneticAlgorithmSettings");
const saveButton = document.getElementById("saveSettings");
const closeButton = document.getElementById("closeSettings");

// 显示设置窗口
settingsButton.addEventListener("click", function() {
    modal.style.display = "block";
});

// 关闭设置窗口
closeButton.addEventListener("click", function() {
    modal.style.display = "none";
});

// 保存参数设置
saveButton.addEventListener("click", function() {
    const settingsData = {
        iterations: document.getElementById("iterations").value,                  // 保持不变
        eq_length: document.getElementById("equation_length").value,              // 修改键名为 'eq_length'
        crossover_prob: document.getElementById("crossover_probability").value,  // 修改键名为 'crossover_prob'
        mutation_prob: document.getElementById("mutation_probability").value,   // 修改键名为 'mutation_prob'
        initial_eq: document.getElementById("initial_equations").value,          // 修改键名为 'initial_eq'
        n_eq_selected_each_gene: document.getElementById("number of equationsselected in each generation").value,  // 新参数不受影响
    };

    // 发送数据到后端
    fetch("/save_settings", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(settingsData)
    }).then(response => response.json())
      .then(data => {
          alert("Settings saved successfully!");
          console.log(data);
          modal.style.display = "none";
      }).catch(error => {
          console.error("Error:", error);
      });
});
document.getElementById('submitButton').addEventListener('click', () => {
    const url = '/submit_combination_size'; // 后端的目标 API 路径
    const formData = new FormData();
    formData.append('combination_size', document.getElementById('combinationSizeInput').value);

    fetch(url, {
        method: 'POST',
        credentials: 'include', // 确保携带 Cookie
        body: formData,
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.text();
    })
    .then(data => console.log(data))
    .catch(error => console.error('Error:', error));
});

</script>
