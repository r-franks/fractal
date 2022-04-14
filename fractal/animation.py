import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def animate(vals_list, file, fps=60, windowsize=None, figsize=None, keep_axis=True, plot_kwargs={}):    
    """Creates an animation by plotting the
    2d coordinate arrays in vals_list
    sequentially
    
    Parameters
    ----------
    vals_list: list of 2x? numpy arrays or list of
    lists of 2x?
        List of coordinates to be plotted for each
        frame
    file: str
        where to save the animation
    fps: int
        fps of gif
    windowsize: list of 2 lists of length 2
        The range of the plotted
        window's x and y domains:
        [[xmin, xmin], [ymin, ymax]]
    figsize: list of two ints
        size of figure
    keep_axis: bool
        whether to retain axes
    plot_kwargs: dict
        dictionary of plotting specific
        parameters
 
    Returns
    -------
    None (saves an animated gif)
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    all_arrays = np.any([isinstance(vals, np.ndarray)==False for vals in vals_list]) == False
    all_lists = np.any([isinstance(vals, list)==False for vals in vals_list]) == False
    
    if windowsize:
        xmin, xmax = windowsize[0][0], windowsize[0][1]
        ymin, ymax = windowsize[1][0], windowsize[1][1]
    else:
        if all_arrays:
            vals_concat = 1.1*np.concatenate(vals_list)
            xmin, xmax = vals_concat[:,0].min(), vals_concat[:,0].max()
            ymin, ymax = vals_concat[:,1].min(), vals_concat[:,1].max()
        elif all_lists:
            vals_concat = 1.1*np.concatenate([np.concatenate(vals) for vals in vals_list])
            xmin, xmax = vals_concat[:,0].min(), vals_concat[:,0].max()
            ymin, ymax = vals_concat[:,1].min(), vals_concat[:,1].max()
        else:
            raise Exception("vals_list must be a list of 2d arrays or a list of lists of 2d arrays")

    def plot_frame(t):
        ax.clear()
        
        if not keep_axis:
            ax.axis('off')
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        if isinstance(vals_list[t], list):
            for vals in vals_list[t]:
                ax.plot(vals[:,0], vals[:,1], **plot_kwargs)
        else:
            ax.plot(vals_list[t][:,0], vals_list[t][:,1], **plot_kwargs)
        
    anim = animation.FuncAnimation(fig, plot_frame, len(vals_list))
    
    writervideo = animation.PillowWriter(fps=60)
    anim.save(file, writer=writervideo)
    plt.close()