from src.config import CONFIG
from src.data_processing import prepare_data
from src.utils import plot_relationships, plot_populations, plot_population_histogram, population_summary

def main():
    data = prepare_data()
    population_summary(data, CONFIG["model"]["target"])
    plot_relationships(data, CONFIG["model"]["target"])
    plot_populations(data, CONFIG["model"]["target"])
    plot_population_histogram(data, CONFIG["model"]["target"])

if __name__ == "__main__":
    main()