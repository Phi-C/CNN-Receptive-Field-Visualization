import streamlit as st
import json
import pickle
import torch

# 定义模型和对应的层列表
with open("meta_info.pkl", "rb") as f:
    meta_info = pickle.load(f)
    
MODEL_LAYERS = meta_info["model_layers_dict"]
HW_DICT = meta_info["hw_dict"]
input_height, input_width = meta_info["input_hw"]


# TODO: 实现自定义的 mapping 函数
def mapping(row, col, feat_width, index_dict):
    feat_idx = row * feat_width + col
    if feat_idx not in index_dict:
        print(f"feat_idx {feat_idx} not found in index_dict")  # 调试信息
        return []
    rf_tensor = index_dict[feat_idx]
    indices = torch.nonzero(rf_tensor).tolist()
    return [f"cell2_{i}_{j}" for i, j in indices]


# Streamlit 界面
st.title("Receptive Field Visualization")
model_name = st.selectbox("请选择模型名称", options=list(MODEL_LAYERS.keys()))

if model_name:
    layer_list = MODEL_LAYERS[model_name]
    layer_name = st.selectbox("请选择层名称", options=layer_list)
else:
    layer_name = None

if st.button("Visualize Receptive Field"):
    if model_name and layer_name:
        feat_height, feat_width = HW_DICT[layer_name]
        with open(f"{model_name}/{layer_name}.pkl", "rb") as f:
            index_dict = pickle.load(f)

        # 将所有 HTML 和 JavaScript 放在同一个 st.components.v1.html 调用中
        html_content = """
        <div id="coord-display" style="font-size: 16px; font-weight: bold; margin-bottom: 20px;">坐标: </div>
        <div style="display: flex;">
            <div style="margin-right: 20px;">
                <p>第一个方格：</p>
                <table style="border-collapse: collapse; margin-bottom: 20px;">
        """

        # 生成第一个表格
        for i in range(feat_height):
            html_content += "<tr>"
            for j in range(feat_width):
                html_content += f"<td id='cell1_{i}_{j}' style='border: 1px solid black; width: 20px; height: 20px; text-align: center;' onmouseover='highlightCells({i}, {j})'></td>"
            html_content += "</tr>"
        html_content += "</table></div>"

        # 生成第二个表格
        html_content += """
            <div>
                <p>第二个方格：</p>
                <table id="table2" style="border-collapse: collapse;">
        """
        for i in range(input_height):
            html_content += "<tr>"
            for j in range(input_width):
                html_content += f"<td id='cell2_{i}_{j}' style='border: 1px solid black; width: 20px; height: 20px; text-align: center;'></td>"
            html_content += "</tr>"
        html_content += "</table></div></div>"

        # 将 mapping 函数的逻辑转换为 JavaScript 可用的格式
        mapping_js = {
            f"{i}_{j}": mapping(i, j, feat_width, index_dict)
            for i in range(feat_height)
            for j in range(feat_width)
        }

        # 添加 JavaScript 代码
        html_content += f"""
        <script>
        // 定义映射关系
        const mapping = {json.dumps(mapping_js)};

        // 存储当前高亮的单元格
        let currentlyHighlightedCells = [];

        // 高亮第二个表格的单元格
        function highlightCells(row, col) {{
            // 显示当前格子的坐标
            const coordDisplay = document.getElementById("coord-display");
            if (coordDisplay) {{
                coordDisplay.innerText = `坐标: (${{row}}, ${{col}})`;
            }}

            // 清除之前高亮的单元格
            currentlyHighlightedCells.forEach(cellId => {{
                const cell = document.getElementById(cellId);
                if (cell) {{
                    cell.style.backgroundColor = '';
                }}
            }});
            currentlyHighlightedCells = []; // 清空当前高亮的单元格列表

            // 获取需要高亮的单元格 id
            const cellsToHighlight = mapping[`${{row}}_${{col}}`];
            if (cellsToHighlight) {{
                cellsToHighlight.forEach(cellId => {{
                    const cell = document.getElementById(cellId);
                    if (cell) {{
                        cell.style.backgroundColor = 'yellow';
                        currentlyHighlightedCells.push(cellId); // 记录当前高亮的单元格
                    }}
                }});
            }}
        }}
        </script>
        """

        # 注入 HTML 和 JavaScript
        st.components.v1.html(html_content, height=600)
    else:
        st.warning("请选择模型名称和层名称")
