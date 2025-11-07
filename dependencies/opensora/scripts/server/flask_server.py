from flask import Flask, jsonify, request
from scripts.server.client import send_to_server
import torch
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET"])
def test():
    return jsonify({"message": "Test Successful!!!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    obs = data.get("obs", [])
    action = data.get("action", [])
    obs = torch.tensor(np.array(obs), dtype=torch.bfloat16)
    action = torch.tensor(np.array(action), dtype=torch.bfloat16)
    # result = send_to_server(obs, action)

    # 这里你可以添加实际的处理逻辑，比如模型推理
    result = {
        "received_obs_shape": str(len(obs)),
        "received_action_shape": str(len(action)),
        "status": "Prediction done."
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="::", port=9642, debug=False)


# from flask import Flask, jsonify

# if __name__ == "__main__":
#     app = Flask(__name__)
#     @app.route("/", methods=["GET"])
#     def test():
#         return jsonify("Test Successful!!!")
#     app.run(host="::", port=10346, debug=False)
 