import pickle
import numpy as np
import matplotlib.pyplot as plt


def main():
    with open("edges.pkl", "rb") as f:
        edges = pickle.load(f)

    # Get coordinates where edge was detected
    binary = edges != 0
    inds = np.transpose(np.indices(binary.shape), (1, 2, 0))
    inds = inds[binary]

    # Normalize
    scale = max(edges.shape)
    x = inds[..., 1] / scale
    y = -inds[..., 0] / scale + 1

    plt.scatter(x, y, marker='.')
    plt.savefig("img.png")
    plt.show()


if __name__ == "__main__":
    main()
