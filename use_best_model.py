from src.selected_model import train_best_model, test_on_other_datasets
from src.config import CONFIG

def main():
    model_key = CONFIG["best_model_key"]
    data_testing_path = CONFIG["data_testing_path"]

    model = train_best_model(model_key, save = False)

    test_on_other_datasets(model, model_key, data_testing_path)


if __name__ == "__main__":
    main()