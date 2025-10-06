from flask import Flask, request, jsonify
from inference_showo import ShowoModel
import os

app = Flask(__name__)

# 初始化推理类
showo_model = ShowoModel(
    config="outputs/show-o-tuning-stage2-512x512/config_infer.yaml",
    max_new_tokens=1000,
    temperature=0.8,
    top_k=1,
    load_from_showo=False,
)


@app.route('/mmu_infer', methods=['POST'])
def mmu_infer():
    try:
        data = request.get_json()

        # 验证输入
        if 'image_path' not in data or 'question' not in data:
            return jsonify({'error': 'Missing required fields'}), 400

        image_path = data['image_path']
        question = data['question']

        # 验证图片存在
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404

        # 执行推理
        response = showo_model.mmu_infer(image_path, question)
        print(response)
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/t2i_infer', methods=['POST'])
def t2i_infer():
    try:
        data = request.get_json()

        # 验证输入
        if 'prompts' not in data:
            return jsonify({'error': 'Missing prompts field'}), 400

        prompts = data['prompts']

        # 验证prompts是列表
        if not isinstance(prompts, list):
            return jsonify({'error': 'Prompts must be a list'}), 400

        # 执行推理
        image_paths = showo_model.t2i_infer(prompts)

        return jsonify({
            'image_paths': image_paths
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
