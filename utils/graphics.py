from base_packages import *


# Function which creates a figure according to the number of subplots
def create_figure(**kwargs):
    """
    This function is responsible to build figures based on the different arguments the user provides, among which:
        - figsize
        - subplots
        - sharex (Link between the different x axis over the subplots)
        - sharey (Link between the different y axis over the subplots)
    """
    fig = plt.figure(figsize=kwargs.get('figsize', (20, 10)))
    subplots = kwargs.get('subplots', (1, 1))
    sharex = kwargs.get('sharex', False)
    sharey = kwargs.get('sharey', False)
    axes = np.empty(subplots, dtype=object)
    ax_num = 1
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            if ax_num == 1:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num)
            elif sharex and sharey:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num, sharex=axes[0][0], sharey=axes[0][0])
            elif sharex:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num, sharex=axes[0][0])
            elif sharey:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num, sharey=axes[0][0])
            else:
                axes[i][j] = plt.subplot(subplots[0], subplots[1], ax_num)
            axes[i][j].spines['right'].set_visible(False)
            axes[i][j].spines['top'].set_visible(False)
            ax_num += 1

    if kwargs.get('tight_layout', False):
        fig.tight_layout()

    return fig, axes


# Function which completes a figure with the different titles.
# Should be called after the creation of the figure and plotting the data
def complete_figure(fig, axes, **kwargs):
    """
    This function is responsible to complete figures based on the different arguments the user provides, among which:
        - different fontiszes
        - display legends
        - limits of x and y axes
        - x and y ticks
        - save figure (default directory is cts.SNAPSHOTS_DIR)
        These parameters should be provided in a 2D array where the dimensions are: (n_horizontal_subplots, n_vertical_subplots)
    """
    xticks_fontsize = kwargs.get('xticks_fontsize', 28)
    yticks_fontsize = kwargs.get('yticks_fontsize', 28)
    xlabel_fontsize = kwargs.get('xlabel_fontsize', 28)
    ylabel_fontsize = kwargs.get('ylabel_fontsize', 28)
    frameon = kwargs.get('frameon', False)
    x_titles = kwargs.get('x_titles', '' * np.ones(axes.shape, dtype=object))  # No titles
    y_titles = kwargs.get('y_titles', '' * np.ones(axes.shape, dtype=object))  # No titles
    x_lim = kwargs.get('x_lim', 'auto' * np.ones(axes.shape, dtype=object))
    y_lim = kwargs.get('y_lim', 'auto' * np.ones(axes.shape, dtype=object))
    x_ticks = kwargs.get('x_ticks', 'auto' * np.ones(axes.shape, dtype=object))
    y_ticks = kwargs.get('y_ticks', 'auto' * np.ones(axes.shape, dtype=object))
    x_ticks_labels = kwargs.get('x_ticks_labels', 'auto' * np.ones(axes.shape, dtype=object))
    y_ticks_labels = kwargs.get('y_ticks_labels', 'auto' * np.ones(axes.shape, dtype=object))
    put_legend = kwargs.get('put_legend', True * np.ones(axes.shape, dtype=bool))
    loc_legend = kwargs.get('loc_legend', 'best' * np.ones(axes.shape, dtype=object))
    format = kwargs.get('format', 'png')

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i][j].set_xlabel(x_titles[i][j], fontsize=xlabel_fontsize)
            axes[i][j].set_ylabel(y_titles[i][j], fontsize=ylabel_fontsize)
            axes[i][j].tick_params(axis='x', labelsize=xticks_fontsize)
            axes[i][j].tick_params(axis='y', labelsize=yticks_fontsize)
            if put_legend[i][j]:
                axes[i][j].legend(fontsize=kwargs.get('legend_fontsize', 28), loc=loc_legend[i][j], frameon=frameon)
            if x_lim[i][j] != 'auto':
                axes[i][j].set_xlim(x_lim[i][j])
            if y_lim[i][j] != 'auto':
                axes[i][j].set_ylim(y_lim[i][j])
            if x_ticks[i][j] != 'auto':
                axes[i][j].set_xticks(x_ticks[i][j])
            if y_ticks[i][j] != 'auto':
                axes[i][j].set_yticks(y_ticks[i][j])
            if x_ticks_labels[i][j] != 'auto':
                axes[i][j].set_xticklabels(x_ticks_labels[i][j])
            if y_ticks_labels[i][j] != 'auto':
                axes[i][j].set_yticklabels(y_ticks_labels[i][j])
    plt.suptitle(kwargs.get('suptitle', ''), fontsize=kwargs.get('suptitle_fontsize', 28))
    if kwargs.get('savefig', False):
        plt.savefig(cts.SNAPSHOTS_DIR / (kwargs.get('main_title', 'NoName') + '.' + format), format=format, transperant=True, dpi=400)
