from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class MetricsCalculator:
    def compute(self, prediction):
        true_labels = prediction.label_ids
        predicted_labels = prediction.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average="macro", zero_division=1
        )
        accuracy = accuracy_score(true_labels, predicted_labels)

        return {
            "Accuracy": accuracy,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
        }
