import matplotlib.pyplot as plt

def plot_graph(data, filename, title, x_label, y_label):

    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()