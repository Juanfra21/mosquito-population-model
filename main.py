from src.config import CONFIG
from src.data_processing import prepare_data
from src.train import test_train_split, train_model, evaluate_model
from src.utils import plot_results
import torch.nn as nn
import numpy as np


def main():
    data = prepare_data()

    test_losses = []
    test_maes = []
    
    for train_size in CONFIG["model"]["train_sizes"]:
        print("-"*20)
        print(f"Cross-Validation. Training using {train_size*100}% of the dataset")
        print("-"*20)
        train_loader, val_loader, test_loader, X_train, X_val, X_test,  y_train,y_val, y_test, scaler = test_train_split(
            data,
            target=CONFIG["model"]["target"],
            train_size=train_size,
            batch_size=CONFIG["model"]["batch_size"],
            seq_length=CONFIG["model"]["seq_length"],
            )

        criterion = nn.MSELoss()
        model = train_model(train_loader=train_loader, val_loader=val_loader, input_size=X_train.shape[2], criterion=criterion)
        y_pred, y_test, loss, mae  = evaluate_model(model=model, test_loader=test_loader, criterion=criterion, scaler=scaler)

        test_losses.append(loss)
        test_maes.append(mae)
        
        # Plot the results
        plot_results(y_test, y_pred)

    print("-"*20)
    print("Cross-Validation results")
    print("-"*20)
    print(f" Mean Loss: {np.mean(test_losses)}")
    print(f" Mean MAE: {np.mean(test_maes)}")


if __name__ == "__main__":
    main()