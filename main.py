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
        seq_length=CONFIG["model"]["seq_length"],
    )

    criterion = nn.MSELoss()
    model = train_model(train_loader=train_loader, val_loader=val_loader, input_size=X_train.shape[2], criterion=criterion)
    y_pred = evaluate_model(model=model, test_loader=test_loader, criterion=criterion)

    # Plot the results
    plot_results(y_test, y_pred)


if __name__ == "__main__":
    main()