import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate_model(model, dataloader, device, class_names=None):
    """
    Avalia um modelo de classificação em um dataloader.

    Retorna um dicionário com:
    - accuracy
    - classification_report (dict)
    - confusion_matrix (ndarray)
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )

    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm
    }
