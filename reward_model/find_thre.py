import json
import numpy as np

with open("results/final_results_thre_0.50.json", "r") as f:
    results = json.load(f)

THRESH_MIN = 0.0
THRESH_MAX = 1.0
THRESH_STEPS = 30

thresholds = np.linspace(THRESH_MIN, THRESH_MAX, THRESH_STEPS)

for thre in thresholds:
    y_true = []
    y_pred = []
    for result in results:
        video_path = result["video_path"]
        true_label = video_path.split("_")[-1].split(".")[0] == '1'

        pred_label = False
        for info in result["results"]:
            pred_prob = info["prob"][1]
            if pred_prob >= thre:
                pred_label = True

        y_true.append(true_label)
        y_pred.append(pred_label)

    # 转换成 numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算指标
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    acc = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"Thre={thre:.2f}, acc={acc:.2f}, recall={recall:.2f}, precision={precision:.2f}, f1={f1:.2f}")
