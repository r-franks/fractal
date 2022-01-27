import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def animate(vals_list, file, fps=60, windowsize=None, figsize=None, plot_kwargs={}):    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)    
    if windowsize:
        xmin, xmax = windowsize[0][0], windowsize[0][1]
        ymin, ymax = windowsize[1][0], windowsize[1][1]
    else:
        vals_concat = 1.1*np.concatenate(vals_list)
        xmin, xmax = vals_concat[:,0].min(), vals_concat[:,0].max()
        ymin, ymax = vals_concat[:,1].min(), vals_concat[:,1].max()

    def plot_frame(t):
        ax.clear()
        
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        ax.plot(vals_list[t][:,0], vals_list[t][:,1], **plot_kwargs)
        
    anim = animation.FuncAnimation(fig, plot_frame, len(vals_list))
    
    writervideo = animation.PillowWriter(fps=60)
    anim.save(file, writer=writervideo)
    plt.close()