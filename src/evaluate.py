import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import print_metrics


def evaluate_model(y_true, y_pred, y_prob=None, model_name=None):
    print_metrics(y_true, y_pred, y_prob, model_name)

    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name or ''}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
    except Exception:
        pass
