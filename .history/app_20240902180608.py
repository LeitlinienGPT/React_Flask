from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
import threading
import json
import time

from langchain_logic import prompt_queue, sse_event_queue, qa_chain, StreamHandler

app = Flask(__name__, static_folder='./my-app/build', static_url_path='/')
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Enable CORS for all routes and allow requests from React frontend

@app.route('/')
def serve_react_app():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/process', methods=['POST'])
def process():
    prompt = request.json.get('question')
    if prompt:
        prompt_queue.put(prompt)
        return {'status': 'success'}, 200
    return {'status': 'failed', 'message': 'No prompt provided.'}, 400

def send_sse_data():
    global qa_chain, prompt_queue, sse_event_queue, response_thread
    while True:
        if not prompt_queue.empty():
            if response_thread and response_thread.is_alive():
                continue

            prompt = prompt_queue.get()

            response_thread = threading.Thread(target=qa_chain.run, args=(prompt,), kwargs={'callbacks': [StreamHandler()]})
            response_thread.start()

        while not sse_event_queue.empty():
            sse_event = sse_event_queue.get()
            yield f"data: {json.dumps(sse_event)}\n\n"

        time.sleep(1)

@app.route('/stream', methods=['GET'])
def stream():
    def event_stream():
        return send_sse_data()

    return Response(event_stream(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8080)
