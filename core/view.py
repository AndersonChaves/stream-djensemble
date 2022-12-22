import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

def print_array_screen(array_list):
    if isinstance(array_list, list):
        n = len(array_list)
        f, axis = plt.subplots(1, n)
        title_list = ["Clustering", "Tiling"]
        for array, i, title in zip(array_list, range(n), title_list):
            axis[i].set_title(title)
            #axis[i].imshow(array, cmap="tab20", vmin=0, vmax=array_list[i].max())
            axis[i].imshow(array, cmap="viridis", vmin=0, vmax=array.max())
    else:
        plt.imshow(array_list, cmap='gray', vmin=0, vmax=array_list[1].max())#, interpolation='nearest')
    plt.show()

def save_figure_from_matrix(matrix : np.array, title: str,
                              parent_directory='', write_values=False):

    matplotlib.use('TkAgg')
    matplotlib.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)

    if write_values:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                c = round(matrix[j, i], 1)
                ax.text(i, j, str(c), va='center', ha='center')
                # print("I, J: ", i, ' ', j)

    fig.tight_layout()
    fig.savefig(parent_directory + title + '.png', dpi=40)
    #plt.pause(1)
    plt.close("all")

def plot_graph_line(error_history, title: str,
                   parent_directory='', write_values=False):
    matplotlib.use('TkAgg')

    fig, ax = plt.subplots()
    print("Error History: ", error_history)
    ax.plot(list(range(len(error_history))), error_history)
    fig.savefig(parent_directory + title + '.png', dpi=40)

    #plt.plot(list(range(len(error_history))), error_history)
    #plt.savefig(parent_directory + title + '.png', dpi=40)

def save_figure_as_scatter_plot(x, y, clusters, title:str, parent_directory=''):
    matplotlib.use('TkAgg')
    figure(figsize=(8, 6), dpi=130)

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=clusters, s=30, cmap='viridis')
    fig.savefig(parent_directory + title + '.png')
    plt.close("all")

if __name__ == '__main__':
    matrix = np.random.random((128, 128))
    #matrix = np.random.random((10, 10))
    save_figure_from_matrix(matrix, "../figures/last-buffer", parent_directory='', write_values=False)
    print("Done")