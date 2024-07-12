from src.config import CONFIG
from src.data_processing import prepare_data
from src.utils import plot_relationships, plot_populations

def main():
    data = prepare_data()
    plot_relationships(data,CONFIG["model"]["target"])
    plot_populations(data,CONFIG["model"]["target"])

if __name__ == "__main__":
    main()