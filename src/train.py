import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import numpy as np
from src.config import CONFIG
from src.model import LSTMModel
import pandas as pd
from sklearn.preprocessing import StandardScaler

def select_target(data, target):
    # Select features and target based on target parameter
    if target == 1:
        y_col = "population"
    elif target == 2:
        y_col = "population_2d"
    elif target == 3:
        y_col = "population_3d"
    else:
        raise ValueError("Invalid target value. Choose 1, 2 or 3")

    # Drop unnecessary columns
    X = data.drop(columns=["date", "population", "population_2d", "population_3d"])
    y = data[y_col]

    return X, y


def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X.iloc[i:(i+seq_length)].values)
        ys.append(y.iloc[i+seq_length])
        
    return np.array(Xs), np.array(ys)


def test_train_split(data, target, train_size, batch_size, seq_length):
    X, y = select_target(data, target)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X))
    y = pd.Series(scaler.fit_transform(y.to_frame()).flatten())

    # Create sequences
    X, y = create_sequences(X, y, seq_length)

    # Time-based split
    split_date_train = int(train_size * len(X))

    X_train, X_test = X[:split_date_train], X[split_date_train:]
    y_train, y_test = y[:split_date_train], y[split_date_train:]

    # Further split train set into train and validation sets (using 80-20 split)
    split_date_val = int(0.8 * len(X_train))
    
    X_train, X_val = X_train[:split_date_val], X_train[split_date_val:]
    y_train, y_val = y_train[:split_date_val], y_train[split_date_val:]

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.astype(np.float32))
    X_val_tensor = torch.tensor(X_val.astype(np.float32))
    X_test_tensor = torch.tensor(X_test.astype(np.float32))
    y_train_tensor = torch.tensor(y_train.astype(np.float32)).reshape(-1, 1)
    y_val_tensor = torch.tensor(y_val.astype(np.float32)).reshape(-1, 1)
    y_test_tensor = torch.tensor(y_test.astype(np.float32)).reshape(-1, 1)

    # Create DataLoader for training, validation, and test sets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def train_model(train_loader, val_loader, input_size, criterion):
    model = LSTMModel(input_size, CONFIG["model"]["hidden_size"], CONFIG["model"]["num_layers"], CONFIG["model"]["output_size"], CONFIG["model"]["seq_length"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["model"]["learning_rate"])

    for epoch in range(CONFIG["model"]["num_epochs"]):
        model.train()
        epoch_train_loss = 0.0

        # Training loop
        for i, (inputs, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Calculate average training loss for the epoch
        epoch_train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0.0
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                epoch_val_loss += val_loss.item()

            epoch_val_loss /= len(val_loader)

        # Print progress
        print(f'Epoch [{epoch+1}/{CONFIG["model"]["num_epochs"]}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

    return model


def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_losses = []
    all_y_pred = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            test_loss = criterion(outputs, targets)
            test_losses.append(test_loss.item())
            all_y_pred.append(outputs.numpy())
            all_targets.append(targets.numpy())

    all_y_pred = np.concatenate(all_y_pred)
    all_targets = np.concatenate(all_targets)
    r2 = r2_score(all_targets, all_y_pred)

    print(f"Test Loss: {np.mean(test_losses):.4f}, R^2 Score: {r2:.4f}")

    return all_y_pred
