import matplotlib.pyplot as plt


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
