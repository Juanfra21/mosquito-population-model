import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import numpy as np
from src.config import CONFIG
from src.model import LSTMModel


def test_train_split(data, target, train_size, test_from_start, batch_size):
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

    # Time-based split
    split_date_train = int(train_size * len(data))
    if test_from_start:
        X_train, X_test = X.iloc[: len(data) - split_date_train], X.iloc[len(data) - split_date_train :]
        y_train, y_test = y.iloc[: len(data) - split_date_train], y.iloc[len(data) - split_date_train :]

    else:
        X_train, X_test = X.iloc[:split_date_train], X.iloc[split_date_train:]
        y_train, y_test = y.iloc[:split_date_train], y.iloc[split_date_train:]

    # Further split train set into train and validation sets (using 80-20 split)
    split_date_val = int(0.8 * len(X_train))
    
    if test_from_start:
        X_train, X_val = X.iloc[: len(data) - split_date_val], X.iloc[len(data) - split_date_val :]
        y_train, y_val = y.iloc[: len(data) - split_date_val], y.iloc[len(data) - split_date_val :]

    else:
        X_train, X_val = X_train.iloc[:split_date_val], X_train.iloc[split_date_val:]
        y_train, y_val = y_train.iloc[:split_date_val], y_train.iloc[split_date_val:]

    # Convert to numpy arrays and then to tensors
    X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
    X_val_tensor = torch.tensor(X_val.values.astype(np.float32))

    X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
    y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).reshape(-1, 1)

    y_val_tensor = torch.tensor(y_val.values.astype(np.float32)).reshape(-1, 1)
    y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).reshape(-1, 1)

    # Create DataLoader for training, validation, and test sets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        X_train.values,
        X_val.values,
        X_test.values,
        y_train.values,
        y_val.values,
        y_test.values,
    )


def train_model(train_loader, val_loader, input_size, criterion):
    model = LSTMModel(
        input_size,
        CONFIG["model"]["hidden_size"],
        CONFIG["model"]["num_layers"],
        CONFIG["model"]["output_size"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["model"]["learning_rate"]
    )

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
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch+1}/{CONFIG["model"]["num_epochs"]}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}'
            )

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
