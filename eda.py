from src.data_processing import prepare_data
from src.utils import plot_relationships, plot_populations

def main():
    data = prepare_data()
    plot_relationships(data,'population')
    plot_populations(data,'population')

if __name__ == "__main__":
    main()