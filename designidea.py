import matplotlib.pyplot as plt
import numpy as np

def plot_hand_drawn_style(x, y):
    # Activate XKCD style
    plt.xkcd()

    # Create a figure
    fig, ax = plt.subplots()
    ax.plot(x, y, label="datam", lw=2, color='red')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")

    # Add arrows to the x-axis and y-axis
    ax.annotate('', xy=(max(x), 0), xytext=(min(x), 0),
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5))
    ax.annotate('', xy=(0, max(y)), xytext=(0, min(y)),
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

    # Adjust the plot limits to make space for the arrows
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
    ax.set_ylim(min(y) - 0.5, max(y) + 0.5)

    # Turn off all the spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Turn off all the ticks
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    return fig