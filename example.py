import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from src.helper import create_som


def main():

    # make sample data
    num_inputs = 512
    dims = 5
    clusters = 6

    blob_data = make_blobs(
        n_samples=num_inputs,
        n_features=dims,
        centers=clusters)

    # SOM hyperparameters
    m = 20
    n = 20

    som, umatrix, bmu_loc = create_som(
        input_data=blob_data,
        dims=dims,
        num_inputs=num_inputs,
        batch_size=128,
        m=m,
        n=n
    )

    # plot results
    fig = plt.figure()
    plt.imshow(umatrix.reshape((m, n)), origin='lower')
    plt.show(block=True)


if __name__ == "__main__":
    main()
