import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Simple Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,activation_function = nn.ReLU(), dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout_rate = dropout_rate
        if self.dropout_rate: self.dropout = nn.Dropout(p=dropout_rate)  # Add dropout layer
        self.relu = activation_function
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        if self.dropout_rate: out = self.dropout(out)
        out = self.fc2(out)
        return out
    
    
class MoreComplexNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes,activation_function=nn.ReLU()):
        super(MoreComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = activation_function
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = activation_function
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
    
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,device=None):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class CNN1DModel(nn.Module):
    def __init__(self, input_size, num_classes, num_filters=32, kernel_size=3):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        output_size = self._calculate_output_size(input_size, kernel_size)
        self.fc = nn.Linear(output_size, num_classes)

    def _calculate_output_size(self, input_size, kernel_size):
      conv_output_size = input_size - kernel_size + 1
      pool_output_size = conv_output_size // 2
      return pool_output_size * 32

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch_size, input_size, seq_length)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, num_heads=8, hidden_size=64, num_layers=2):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                                        dim_feedforward=hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, num_classes)  # Linear layer after transformer

    def forward(self, x):
        # Input x: (batch_size, sequence_length, input_size)
        # Transformer expects (sequence_length, batch_size, input_size)
        x = x.permute(1, 0, 2)
        out = self.transformer_encoder(x)
        out = out.permute(1, 0, 2)  # Back to (batch_size, seq_length, input_size)
        out = self.fc(out[:, -1, :])  # Use the last time step's output
        return out
    
    
    
# Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, val_loader=None,scheduler=None,device=None):
    model.train()
    train_losses = []  # Store training losses
    val_losses = []    # Store validation losses
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}')

        # Validation loop (if val_loader is provided)
        if val_loader:
            model.eval()  # Set to evaluation mode
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()

            avg_val_loss = val_running_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            # print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}')
            model.train() #switch back to train
        if scheduler:
            scheduler.step()  # Update learning rate

    return train_losses, val_losses


# Evaluation Loop
def evaluate_model(model, test_loader, class_names,model_name='SimpleNN',device=None):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # Store probabilities

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())  # Store predictions, convert to numpy
            all_labels.extend(labels.cpu().numpy())    # Store labels, convert to numpy
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f'./plots/confusion_matrix_{model_name}.png')
    plt.show()

    # ROC AUC and Curves (One-vs-Rest)
    if len(class_names) > 1:
        all_probs = np.array(all_probs)
        # Correctly binarize the labels using NumPy
        all_labels_binarized = np.zeros((len(all_labels), len(class_names)))
        for i in range(len(class_names)):
            all_labels_binarized[:, i] = (np.array(all_labels) == i).astype(int)  # Convert to NumPy array

        roc_auc = roc_auc_score(all_labels_binarized, all_probs, multi_class='ovr')
        print(f"ROC AUC: {roc_auc:.4f}")

        # Plot ROC curves for each class
        plt.figure(figsize=(8, 6))
        fpr = dict()
        tpr = dict()
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(all_labels_binarized[:, i], all_probs[:, i])
            plt.plot(fpr[i], tpr[i], label=f'Class {class_names[i]} vs Rest (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('One-vs-Rest ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(f'./plots/roc_curve_{model_name}.png')
        plt.show()
    else:
        print("ROC AUC and curves are not applicable for single-class problems.")

        
        

def plot_loss_function(training_losses, validation_losses,model_name):
    # --- Plotting Training and Validation Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(f'./plots/loss_function_{model_name}.png')
    plt.show()