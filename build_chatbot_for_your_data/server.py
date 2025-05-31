import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import worker

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json['userMessage']
    bot_response = worker.process_prompt(user_message)
    return jsonify({"botResponse": bot_response})


@app.route('/process-document', methods=['POST'])
def process_document_route():
    if 'file' not in request.files:
        return jsonify({
            "botResponse": "File not uploaded correctly. Please try again."
        }), 400

    file = request.files['file']
    file_path = file.filename
    file.save(file_path)

    try:
        worker.process_document(file_path)
        return jsonify({
            "botResponse": "PDF analyzed! You can now ask questions about the document."
        })
    except Exception as e:
        return jsonify({
            "botResponse": f"Error processing document: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')