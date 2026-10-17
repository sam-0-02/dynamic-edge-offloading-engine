import time
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Lightweight endpoint for measuring network Round-Trip Time (RTT)."""
    return jsonify({"status": "ok", "timestamp": time.time()}), 200


@app.route("/offload", methods=["POST"])
def offload_task():
    """Executes offloaded synthetic computational task (matrix multiplication).

    Payload format:
        {
            "matrix_size": int (e.g., 500)
        }
    """
    start_time = time.time()
    data = request.get_json()

    if not data or "matrix_size" not in data:
        return jsonify({"error": "Invalid payload, 'matrix_size' required."}), 400

    size = int(data["matrix_size"])

    # Generate synthetic N x N matrices and perform dot product / inference task
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)
    result = np.dot(matrix_a, matrix_b)

    server_compute_time = time.time() - start_time

    return (
        jsonify(
            {
                "status": "success",
                "matrix_size": size,
                "result_shape": list(result.shape),
                "server_compute_time": server_compute_time,
            }
        ),
        200,
    )


if __name__ == "__main__":
    # Host on 0.0.0.0 to listen on local interface
    app.run(host="127.0.0.1", port=5000, debug=False)