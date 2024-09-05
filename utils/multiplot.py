import numpy as np
import matplotlib.pyplot as plt

PLOT_DETAIL = 10000 # The maximum number of points to display at once, afterward this amount of points will be uniformly pulled from the set of all points.
MEDIAN_SMOOTHING = 0 # The amount to divide by for median smooth. In this case, 0 = off. 1 should also = off.

class Multiplot():
    # TODO Make the graphs more customizable
    """
    Plots multiple plots in a figure using names.

    Attributes:
        plots (dict[str,list]): Stores y-points for plots by name.
        names (list[str]): The list of line names, including row-breaks/column-breaks with "rb" and "cb".
        fig (matplotlib.figure.Figure): The multiplot figure.
        axs (matplotlib.axes.Axes): The axs of the multiplot.
    """
    def __init__(self, names):
        self.fig, self.axs = plt.subplots(2, 2) # Generate original figure for matplotlib
        """
        The constructor for the Multiplot class.

        Parameters:
            names (list[str]): The list of line names, including row-breaks/column-breaks with "rb" and "cb".
        """
        self.plots = {} 
        self.names = names
        
        # Initialize empty array for each named line, excluding rb and cb flags
        ax_idx = [0, 0]
        curr_ax = self.axs[0][0]
        for n in names:
            if n != "rb" and n!= "cb":
                self.plots[n] = []

            if n == "rb":
                ax_idx[0] += 1

            # Column break
            if n == "cb":
                ax_idx[1] += 1
                ax_idx[0] = 0

            # After changing curr_ax position, clear it.
            if n == "cb" or n == "rb":
                curr_ax = self.axs[ax_idx[0], ax_idx[1]]
                continue

            curr_ax.plot([0, 0], label=n)
            curr_ax.legend()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def add_entry(self, name, entry):
        """
        Appends a y-value to a named line of the Multiplot.

        Parameters:
            name (str): A name from the list this Multiplot was initalized with.
            entry (float | list[float] | ndarray): The y-value of the point (float) or points (1-d array of floats).
        """

        self.plots[name] = np.append(self.plots.get(name), entry)
    
    def plot_all(self, max_x):
        """
        Plots all of the mulitplot lines.

        Parameters:
            max_x (int): The maximum label of the x-axis.
        """
        ax_idx = [0, 0]
        line_idx = 0

        # Clear the top left box, since it will not be preceded by an row break (rb) or column break (cb)
        curr_ax = self.axs[0, 0]
        lines = curr_ax.get_lines()

        # Loop through the names
        for name in self.names:
            # Row break
            if name == "rb":
                ax_idx[0] += 1

            # Column break
            if name == "cb":
                ax_idx[1] += 1
                ax_idx[0] = 0

            # After changing curr_ax position, clear it.
            if name == "cb" or name == "rb":
                curr_ax = self.axs[ax_idx[0], ax_idx[1]]
                lines = curr_ax.get_lines()
                line_idx = 0
                continue
            
            # If the named line has any points, plot them w/ legend
            if len(self.plots[name]) > 0:
                np_plot = np.array(self.plots.get(name))
                
                idxs = np.linspace(0, len(np_plot) - 1, min(len(np_plot), PLOT_DETAIL))
                
                if len(np_plot) > PLOT_DETAIL:
                    np_plot = [np_plot[int(i)] for i in idxs]

                if MEDIAN_SMOOTHING > 0:
                    median = np.median(np_plot)
                    smoothed_plot = median + (np_plot - median) / MEDIAN_SMOOTHING
                else: smoothed_plot = np_plot

                x = np.linspace(0, max_x - 1, min(len(smoothed_plot), PLOT_DETAIL))
                lines[line_idx].set_data(x, smoothed_plot)
            
            curr_ax.relim()
            curr_ax.autoscale()

            line_idx += 1
        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.tight_layout()