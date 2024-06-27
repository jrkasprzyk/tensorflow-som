import matplotlib.pyplot as plt
from yaml import safe_load
from numpy import genfromtxt

from src.helper import create_som


def main():

    config = safe_load(open("best_params.yaml"))

    # The CSV has SOWs in columns (observations) and years in rows (features)
    # Note: using raw data because scaling occurs inside SOM training function
    cumulative_supply = genfromtxt('500_sow_cumulative_supply.csv', delimiter=',', skip_header=1)

    # Transform data into SOWs in rows and years in columns
    cumulative_supply = cumulative_supply.transpose()

    data_rows, data_cols = cumulative_supply.shape

    som, umatrix, bmu_loc = create_som(
        input_data=cumulative_supply,
        dims=data_cols,
        num_inputs=data_rows,
        batch_size=128,
        num_rows_neurons=config["y_dim"],
        num_cols_neurons=config["x_dim"],
        weights_init="PCA" # currently not working
    )

    # plot results
    fig = plt.figure()
    plt.imshow(umatrix.reshape((config["y_dim"], config["x_dim"])), origin='lower')
    plt.show(block=True)


if __name__ == "__main__":
    main()
