import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import config as conf
from model_architecture2 import ModelBuilder
import traceback  # Add this import to capture detailed error messages

# Ensure configuration values are set
def check_config_values():
    required_attrs = [
        'mode', 'model_save_folder', 'tensorboard_save_folder', 'checkpoint_path',
        'base_folder', 'model_name', 'train_folder', 'valid_folder', 'epochs', 'batch_size',
        'time', 'height', 'width', 'color_channels', 'n_classes'
    ]
    for attr in required_attrs:
        if not hasattr(conf, attr) or getattr(conf, attr) is None:
            raise ValueError(f"Configuration attribute {attr} is missing or None")

check_config_values()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if conf.mode == "train":
    os.makedirs(conf.model_save_folder, exist_ok=True)
    os.makedirs(conf.tensorboard_save_folder, exist_ok=True)

class DataTools(Dataset):
    def __init__(self, data_folder, split_name):
        self.name = split_name
        self.data_folder = data_folder
        self._data = os.listdir(self.data_folder)
        self.it = conf.batch_size if split_name == "train" else 32

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        npz_path = os.path.join(self.data_folder, self._data[idx])
        np_data = np.load(npz_path, "r")
        image_seqs = np_data["name1"] / 255.0  # Normalize images
        labels = np_data["name2"]
        
        # Debug: print the actual shape of image_seqs
        print(f"Expected shape: ({conf.time}, {conf.height}, {conf.width}, {conf.color_channels})")
        print(f"Actual shape: {image_seqs.shape}")
        
        # Ensure the shape is [time, height, width, color_channels]
        assert image_seqs.shape == (conf.time, conf.time, conf.height, conf.width, conf.color_channels), \
            f"Shape mismatch: expected ({conf.time}, {conf.height}, {conf.width}, {conf.color_channels}), got {image_seqs.shape}"
        
        # Reshape to [color_channels, time, height, width]
        # image_seqs = image_seqs.transpose(3, 0, 1, 2)
        # return torch.tensor(image_seqs, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels

def _trainer(network, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(conf.epochs):
        network.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = evaluate_model(network, val_loader)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

        print(f'Epoch [{epoch+1}/{conf.epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(network.state_dict(), conf.checkpoint_path.format(epoch=epoch + 1))

    with open(os.path.join(conf.base_folder, "files", conf.model_name, "training_logs.json"), "w") as w:
        json.dump(history, w)

    return history

def check_data_balance(data_folder, data_type):
    labels = []
    for file in os.listdir(data_folder):
        data = np.load(os.path.join(data_folder, file))
        labels.extend(data['name2'].tolist())  # Convert ndarray to list
        label_counter = Counter(map(tuple, labels))  # Convert each ndarray to tuple
    print(f"Data {data_type}: {label_counter}")

def evaluate_model(network, val_loader):
    network.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    val_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = correct / total

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    return val_loss, val_acc

def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

if __name__ == "__main__":
    try:
        model_tools = ModelBuilder()
        network = model_tools.create_network(conf.model_name).to(device)

        if conf.mode == "train":
            train_dataset = DataTools(conf.train_folder, "train")
            valid_dataset = DataTools(conf.valid_folder, "valid")
            train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, collate_fn=custom_collate_fn)
            val_loader = DataLoader(valid_dataset, batch_size=conf.batch_size, shuffle=False, collate_fn=custom_collate_fn)

            check_data_balance(conf.train_folder, 'train')
            check_data_balance(conf.test_folder, 'test')
            check_data_balance(conf.valid_folder, 'valid')

            history = _trainer(network, train_loader, val_loader)
            plot_history(history)
            evaluate_model(network, val_loader)

    except Exception as e:
        print(f"Main execution failed: {e}")
        traceback.print_exc()  # Print the full traceback to help diagnose the issue
