from flask import Flask, request, jsonify
import json
import numpy as np
from parvus import ParvusCompressor

app = Flask(__name__)
system = ParvusCompressor()

@app.route('/upload', methods=['POST'])
def upload_data():
    data = request.json.get('data')
    system.compress(np.array(data))
    return jsonify({"status": "Dataset compressed successfully."})

@app.route('/query', methods=['POST'])
def query_data():
    query = request.json.get('query')
    indices, distances = system.query(query, is_text_query=True)
    return jsonify({"indices": indices.tolist(), "distances": distances.tolist()})

@app.route('/decompress', methods=['POST'])
def decompress_data():
    indices = request.json.get('indices')
    decompressed = system.decompress(indices)
    return jsonify({"decompressed": decompressed.tolist()})

if __name__ == "__main__":
    app.run(port=5000)