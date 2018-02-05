import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def compute_consumers(x1, x2, V):
    """ Compute the number of unique and shared. """

    U1 = np.where(V[:, x1] == 1)[0]
    U2 = np.where(V[:, x2] == 1)[0]
    S = np.intersect1d(U1, U2)
    U1 = np.setdiff1d(U1, S)
    U2 = np.setdiff1d(U2, S)
    return len(U1), len(U2), len(S)


def captive_consumers(r, fig_name):

    # Parameters
    seed = 123
    np.random.seed(seed)
    n_position = 100

    # Uniform position
    P = np.linspace(0, 1, n_position, endpoint=True)
    P = (np.round(P * (n_position-1))).astype(int)

    # Same constant radius for each consumer
    R = int(np.round(n_position * r)) * np.ones(n_position, dtype=int)

    # Build the local view for each consumer
    V = np.zeros((n_position, n_position))
    for i in range(n_position):
        lower_bound = max(0, P[i]-R[i])
        upper_bound = min(P[i]+R[i], n_position)
        V[i, lower_bound:upper_bound] = 1

    C1 = np.zeros((n_position, n_position))
    C2 = np.zeros((n_position, n_position))
    S = np.zeros((n_position, n_position))
    G = np.zeros((n_position, n_position))

    for x1 in tqdm.trange(n_position):
        for x2 in range(n_position):
            u1, u2, s = compute_consumers(x1, x2, V)
            C1[x1, x2] = u1
            C2[x1, x2] = u2
            S[x1, x2] = s
            G[x1, x2] = n_position - u1 - u2 - s

    # Plot this
    fig = plt.figure(figsize=(7.5, 7))

    ax = plt.subplot()
    ax.set_aspect(1)

    # Relative to colormap
    cmap = plt.get_cmap("viridis")
    bounds = np.arange(0, 51, 2)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Imshow object
    im = ax.imshow(C1, origin='lower', extent=[0, 100, 0, 100], norm=norm)

    # Some tuning on axes
    for tick in ax.get_xticklabels():
        tick.set_fontsize("small")
    for tick in ax.get_yticklabels():
        tick.set_fontsize("small")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.2)

    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0.0", "0.5", "1.0"])
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0.0", "0.5", "1.0"])

    # Add a color bar
    cb = plt.colorbar(im, norm=norm, ticks=(0, 25, 50), cax=cax, label="Number of captive consumers")

    # Add a contour
    n_levels = int(C1.max()*16 / (n_position/2))
    ct = ax.contourf(C1, n_levels, origin='lower', vmax=n_position / 2)

    # Indicate middle by horizontal and vertical line
    ax.axhline(50, color="white", linewidth=0.5, linestyle="--", zorder=10)
    ax.axvline(50, color="white", linewidth=0.5, linestyle="--", zorder=10)

    # Name axes
    ax.set_xlabel("Position $a$", labelpad=10)
    ax.set_ylabel("Position $b$", labelpad=10)

    # Put a title
    ax.set_title("$a$'s captive consumers ($r={:.2f}$)".format(r))

    # Cut margins
    plt.tight_layout()

    # Save fig
    plt.savefig(fig_name)
