import evaluate
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers.trainer_utils import PredictionOutput
from seqeval.metrics import classification_report
from src.data import TAG_NAMES

LABELS = ["O"] + TAG_NAMES


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # we need to remove -100 labels for real evaluation
    y_true = [[LABELS[l] for l in label if l != -100] for label in labels]
    y_pred = [
        [LABELS[p] for p, l in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)
    ]

    metric = evaluate.load("seqeval")
    all_metrics = metric.compute(predictions=y_pred, references=y_true)

    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


def report(eval_results: PredictionOutput):
    predictions = eval_results.predictions.argmax(-1)
    labels = eval_results.label_ids

    # we need to remove -100 labels for real evaluation
    y_true = [[LABELS[l] for l in label if l != -100] for label in labels]
    y_pred = [
        [LABELS[p] for p, l in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)
    ]

    return classification_report(y_true, y_pred)


def save_confusion_matrix(eval_results: PredictionOutput):
    predictions = eval_results.predictions.argmax(-1)
    labels = eval_results.label_ids

    # we need to remove -100 labels for real evaluation
    y_true = [l for label in labels for l in label if l != -100]
    y_pred = [p for prediction, label in zip(predictions, labels) for p, l in zip(prediction, label) if l != -100]

    fig, ax = plt.subplots(1, 1, layout="constrained")

    _ = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=range(len(LABELS)), ax=ax, cmap="Blues", colorbar=False
    )
    fig.suptitle("Normalized confusion matrix")
    fig.savefig("cm.jpeg")
