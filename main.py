from src.config import CONFIG
from src.data_processing import prepare_data
from src.train import test_train_split, train_model, evaluate_model
from src.utils import plot_results
import torch.nn as nn


def main():
    data = prepare_data()
    
    (
        train_loader,
        val_loader,
        test_loader,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = test_train_split(
        data,
        target=CONFIG["model"]["target"],
        train_size=CONFIG["model"]["train_size"],
        test_from_start=False,
        batch_size=CONFIG["model"]["batch_size"],
    )

    criterion = nn.MSELoss()

    model = train_model(train_loader, val_loader, X_train.shape[1], criterion)
    y_pred = evaluate_model(model, test_loader, criterion)

    # Plot the results
    plot_results(y_test, y_pred)


if __name__ == "__main__":
    main()