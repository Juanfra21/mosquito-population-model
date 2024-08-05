from src.config import CONFIG
from src.data_processing import prepare_data
from src.train import test_train_split, train_model, evaluate_model
from src.utils import plot_results
import torch.nn as nn
import pandas as pd


def main():    
    # Initialize lists to store results
    results = []

    for model_key in CONFIG.keys():
        if model_key.startswith('model'):
            model_config = CONFIG[model_key]

            print("-" * 40)
            print(f"Training and evaluating {model_key}...")
            print("-" * 40)

            # Process data
            data = prepare_data(CONFIG["data"]["population_path"], CONFIG["data"]["weather_path"], model_config["population_rolling_window"])
            
            # Prepare data
            train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test, scaler = test_train_split(
                data,
                target=model_config["target"],
                train_size=model_config["train_size"],
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

            # Evaluate model
            y_test, y_pred, loss, mae, mse, rmse = evaluate_model(model=model, test_loader=test_loader, criterion=criterion, scaler=scaler)

            # Plot results
            plot_results(y_test, y_pred)
            
            # Store metrics
            results.append({
                'model': model_key,
                'mean_loss': loss,
                'mean_mae': mae,
                'mean_mse': mse,
                'mean_rmse': rmse
            })
            
            # Save predictions and real values to CSV
            df_predictions = pd.DataFrame({
                'y_true': y_test,
                'y_pred': y_pred
            })
            df_predictions.to_csv(f'{model_key}_predictions.csv', index=False)
    
    # Convert results to DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('model_comparison.csv', index=False)

    # Print a comparison of all models
    print("-" * 40)
    print("Model Comparison")
    print("-" * 40)
    for _, row in df_results.iterrows():
        print(f"Model: {row['model']}")
        print(f"  Mean Loss: {row['mean_loss']}")
        print(f"  Mean MAE: {row['mean_mae']}")
        print(f"  Mean MSE: {row['mean_mse']}")
        print(f"  Mean RMSE: {row['mean_rmse']}")
        print("-" * 40)


if __name__ == "__main__":
    main()