from src.config import CONFIG
from src.data_processing import prepare_data
from src.train import test_train_split, train_model, evaluate_model
from src.utils import plot_results
import torch
import torch.nn as nn
import pandas as pd


def train_best_model(model_key, save = False):
    # Load the configuration of the selected model
    model_config = CONFIG[model_key]

    # Process data
    data = prepare_data(CONFIG["data"]["population_path"], CONFIG["data"]["weather_path"], model_config["population_rolling_window"])
            
    # Prepare data
    train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test, scaler = test_train_split(
        data,
        target=model_config["target"],
        train_size=1, # Use all data to train
        batch_size=model_config["batch_size"],
        seq_length=model_config["seq_length"],
        )
    
    # Loss function
    criterion = nn.MSELoss()

    # Train model
    model = train_model(
        train_loader, 
        val_loader, 
        X_train.shape[2], 
        criterion, 
        model_config["hidden_size"], 
        model_config["num_layers"], 
        model_config["output_size"], 
        model_config["seq_length"], 
        model_config["learning_rate"], 
        model_config["num_epochs"]
        )
    
    # Save the model to a file
    if save:
        torch.save(model.state_dict(), 'models/model.pth')

    return model


def test_on_other_datasets(model, model_key, data_testing_path):
    # Load the configuration of the selected model
    model_config = CONFIG[model_key]

    # Load data
    data_testing = pd.read_csv(data_testing_path)

    # Preapre new data
    train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test, scaler = test_train_split(
                    data_testing,
                    target=model_config["target"],
                    train_size=0,
                    batch_size=model_config["batch_size"],
                    seq_length=model_config["seq_length"],
                    )

    # Loss function
    criterion = nn.MSELoss()

    # Evaluate
    y_test, y_pred, loss, mae, mse, rmse = evaluate_model(model, test_loader, criterion, scaler)

    # Plot results
    plot_results(y_test, y_pred)