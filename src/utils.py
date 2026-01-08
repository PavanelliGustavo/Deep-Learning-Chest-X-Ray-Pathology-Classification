import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
from contextlib import nullcontext

def get_device():
    """Retorna o dispositivo disponível (CUDA, MPS ou CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Para Macs com chip M1/M2/M3
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class EarlyStopping:
    """Para o treinamento se a perda de validação não melhorar."""
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.status = f"Melhoria encontrada, contador resetado."
        else:
            self.counter += 1
            self.status = f"Sem melhoria por {self.counter} épocas."
            if self.counter >= self.patience:
                self.status = f"Early stopping ativado."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Modelo salvo em {path}')

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f'Modelo carregado de {path}')

# Funções de Mixed Precision (AMP)
def _amp_available(device):
    return device.type == 'cuda' and hasattr(torch.cuda, 'amp')

def get_grad_scaler(device):
    if _amp_available(device):
        return torch.cuda.amp.GradScaler(enabled=True)
    return None

def get_autocast(device):
    if _amp_available(device):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()

def plot_training_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Treino')
    plt.plot(epochs, history['val_loss'], label='Validação')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Treino')
    plt.plot(epochs, history['val_acc'], label='Validação')
    plt.title('Acurácia')
    plt.legend()

    return fig
