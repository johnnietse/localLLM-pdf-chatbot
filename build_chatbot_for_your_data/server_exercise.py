import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import worker  # Import the worker module

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process-message', methods=['POST'])
def process_message_route():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing message in request'}), 400

    try:
        user_message = data['message']
        bot_response = worker.process_message(user_message)
        return jsonify({'response': bot_response})
    except Exception as e:
        app.logger.error(f"Error processing message: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/process-document', methods=['POST'])
def process_document_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 415

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        worker.process_document(file_path)
        return jsonify({'message': 'Document processed and ready for queries'}), 200
    except Exception as e:
        app.logger.error(f"Document processing error: {str(e)}")
        return jsonify({'error': 'Document processing failed'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)  # Disable debug in production