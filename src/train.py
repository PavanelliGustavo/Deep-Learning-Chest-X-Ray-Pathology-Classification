import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import EarlyStopping, get_grad_scaler, get_autocast


def train_engine(model, train_loader, val_loader, device, epochs=100, lr=1e-4, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    scaler = get_grad_scaler(device)
    autocast_ctx = get_autocast(device)
    early_stopper = EarlyStopping(patience=patience)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    model.to(device)
    
    print(f"Iniciando treinamento em: {device}")

    for epoch in range(epochs):
        # --- Treino ---
        model.train()
        run_loss, run_corrects, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed Precision
            with autocast_ctx:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            run_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, 1)
            run_corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)
            
        epoch_loss = run_loss / total
        epoch_acc = run_corrects / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # --- Validação ---
        model.eval()
        val_loss_acc, val_corrects, val_total = 0.0, 0, 0
        
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss_acc += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += inputs.size(0)
                
        val_epoch_loss = val_loss_acc / val_total
        val_epoch_acc = val_corrects / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        scheduler.step(val_epoch_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        
        if early_stopper(model, val_epoch_loss):
            print(f"Early stopping ativado na época {epoch+1}")
            break
            
    return history
