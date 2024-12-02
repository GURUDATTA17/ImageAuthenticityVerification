from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from utils.test_bihpf import predict_bihpf
from utils.test_capsnet import predict_capsnet
from utils.test_keras import predict_keras  # Import for Keras model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    is_allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    print(f"File allowed: {is_allowed} for filename: {filename}")
    return is_allowed


@app.route('/')
def index():
    print("Serving index page")
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received a file upload request")
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        print("No file selected for upload")
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file at {filepath}")
        file.save(filepath)

        try:
            print("Predicting using BiHPF, CapsNet, and Keras models...")
            # Get predictions from all models
            bihpf_result, bihpf_confidence = predict_bihpf(filepath)
            print(f"BiHPF Prediction: {bihpf_result}, Confidence: {bihpf_confidence}")

            capsnet_result, capsnet_confidence = predict_capsnet(filepath)
            print(f"CapsNet Prediction: {capsnet_result}, Confidence: {capsnet_confidence}")

            keras_result, keras_confidence = predict_keras(filepath)
            print(f"Keras Prediction: {keras_result}, Confidence: {keras_confidence}")

            # Determine final result based on highest confidence
            results = [
                ('BiHPF', bihpf_result, bihpf_confidence),
                ('CapsNet', capsnet_result, capsnet_confidence),
                ('Keras', keras_result, keras_confidence)
            ]
            winning_method, final_result, final_confidence = max(results, key=lambda x: x[2])

            print(f"Final Result: {final_result}, Confidence: {final_confidence}, Method: {winning_method}")
            return jsonify({
                'filepath': '/' + filepath,
                'bihpf_result': bihpf_result,
                'bihpf_confidence': f"{bihpf_confidence:.2f}",
                'capsnet_result': capsnet_result,
                'capsnet_confidence': f"{capsnet_confidence:.2f}",
                'keras_result': keras_result,
                'keras_confidence': f"{keras_confidence:.2f}",
                'final_result': final_result,
                'final_confidence': f"{final_confidence:.2f}",
                'winning_method': winning_method
            })
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': str(e)}), 500

    print("Invalid file type submitted")
    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    print("Starting Flask application...")
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
