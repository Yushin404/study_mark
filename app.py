from flask import Flask, request, jsonify
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from mark_reader_core import detect_and_warp, get_fixed_positions, analyze_boxes

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        # 画像をPIL → NumPy(BGR) に変換
        image = Image.open(file.stream).convert("RGB")
        img_np = np.array(image)[:, :, ::-1]  # RGB → BGR

        # 読み取り処理
        warped = detect_and_warp(img_np)
        boxes = get_fixed_positions()
        result, debug = analyze_boxes(warped, boxes)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
