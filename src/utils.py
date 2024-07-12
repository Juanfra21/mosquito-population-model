import matplotlib.pyplot as plt

def plot_relationships(data, target):
    if target == 1:
        x_column = "population"
    elif target == 2:
        x_column = "population_2d"
    elif target == 3:
        x_column = "population_3d"
    else:
        raise ValueError("Invalid target value. Choose 1, 2 or 3")
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    # Create square heatmaps using hexbin
    variables = ['temp', 'humidity', 'precip', 'lw_avg_temp', 'lw_avg_humidity', 'lw_precip']
    for i, var in enumerate(variables):
        x = data[var]
        y = data[x_column]
        ax = axes[i]
        hb = ax.hexbin(x, y, gridsize=40, cmap='viridis', mincnt=1)
        ax.set_title(f'{var} vs {x_column}')
        fig.colorbar(hb, ax=ax)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_populations(data, target):
    if target == 1:
        x_column = "population"
    elif target == 2:
        x_column = "population_2d"
    elif target == 3:
        x_column = "population_3d"
    else:
        raise ValueError("Invalid target value. Choose 1, 2 or 3")
    
    # Set up the plot
    plt.figure(figsize=(10, 5))
    plt.plot(data['date'], data[x_column])
    plt.xlabel('Date')
    plt.ylabel('Mosquito Population')
    plt.title('Date vs Mosquito Population')
    plt.grid(True)
    plt.show()


def plot_results(y_test, y_pred):
    plt.figure(figsize=(16, 6))
    plt.plot(range(len(y_test)), y_test, label="Actual", marker="o")
    plt.plot(range(len(y_pred)), y_pred, label="Predicted", marker="x")
    plt.title("Predicted vs Actual Blood Feeding Adults")
    plt.xlabel("Index")
    plt.ylabel("Blood Feeding Adults")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
