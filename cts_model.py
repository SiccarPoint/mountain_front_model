#!/usr/env/python
"""
Hillslope model with block uplift.

TODO NOTES:
- get a realistic set of parms to play with
- clean up the code to make it "externally runnable"
- make a master run script
- start a matrix of runs exploring different u, d, and L
- while that's going, do some profiling to find speed bottlenecks
"""

_DEBUG = False

import time
#import random
from numpy import zeros, bincount, arange, savetxt, sqrt, log10, mean, arctan, pi, random
from pylab import subplots, plot, show, xlabel, ylabel, title, axis, figure, clf, savefig
from landlab import HexModelGrid
from landlab.io.netcdf import write_netcdf
from landlab.ca.celllab_cts import Transition, CAPlotter
from landlab.ca.oriented_hex_cts import OrientedHexCTS
from landlab.ca.boundaries.hex_lattice_tectonicizer import LatticeUplifter
from scipy.optimize import curve_fit
import matplotlib


class CTSModel(object):
    """
    Implement a generic CellLab-CTS model.

    This is the base class from which models should inherit.
    """

    def __init__(self, num_rows=5, num_cols=5, report_interval=1.0e8,
                   grid_orientation='vertical', grid_shape='rect',
                   show_plots=False, **kwds):
        
        self.initialize(num_rows, num_cols, report_interval,
                   grid_orientation, grid_shape, show_plots, **kwds)


    def initialize(self, num_rows=5, num_cols=5, report_interval=1.0e8,
                   grid_orientation='vertical', grid_shape='rect',
                   show_plots=False, **kwds):
        
        # Remember the clock time, and calculate when we next want to report
        # progress.
        self.current_real_time = time.time()
        self.next_report = self.current_real_time + report_interval
    
        # Create a grid
        self.create_grid_and_node_state_field(num_rows, num_cols, 
                                              grid_orientation, grid_shape)

        # Create the node-state dictionary
        ns_dict = self.node_state_dictionary()

        # Initialize values of the node-state grid
        nsg = self.initialize_node_state_grid(ns_dict)
        
        # Create the transition list
        xn_list = self.transition_list()

        # Create the CA object
        self.ca = OrientedHexCTS(self.grid, ns_dict, xn_list, nsg)
        
        # Initialize graphics
        self._show_plots = show_plots
        if show_plots==True:
            self.initialize_plotting(self)


    def create_grid_and_node_state_field(self, num_rows, num_cols, 
                                         grid_orientation, grid_shape):
        """Create the grid and the field containing node states."""
        self.grid = HexModelGrid(num_rows, num_cols, 1.0, 
                                 orientation=grid_orientation, 
                                 shape=grid_shape)
        self.grid.add_zeros('node', 'node_state', dtype=int)


    def node_state_dictionary(self):
        """Create and return a dictionary of all possible node (cell) states.
        
        This method creates a default set of states (just two); it is a
        template meant to be overridden.
        """
        ns_dict = { 0 : 'on', 
                    1 : 'off'}
        return ns_dict


    def transition_list(self):
        """Create and return a list of transition objects.
        
        This method creates a default set of transitions (just two); it is a
        template meant to be overridden.
        """
        xn_list = []
        xn_list.append(Transition((0, 1, 0), (1, 0, 0), 1.0))
        xn_list.append(Transition((1, 0, 0), (0, 1, 0), 1.0))
        return xn_list

        
    def write_output(grid, outfilename, iteration):
        """Write output to file (currently netCDF)."""
        filename = outfilename + str(iteration).zfill(4) + '.nc'
        write_netcdf(filename, grid)

    
    def initialize_node_state_grid(self, ns_dict=None):
        """Initialize values in the node-state grid.
        
        This method should be overridden. The default is random "on" and "off".        
        """
        if ns_dict is None:
            num_states = 2
        else:
            num_states = len(ns_dict)
        for i in range(self.grid.number_of_nodes):
            self.grid.at_node['node_state'][i] = random.randint(num_states)
        return self.grid.at_node['node_state']


    def initialize_plotting(self, **kwds):
        """Create and configure CAPlotter object."""
        self.ca_plotter = CAPlotter(**kwds)
        self.ca_plotter.update_plot()
        axis('off')


    def run_for(self, dt):

        self.ca.run(self.ca.current_time + dt, self.ca.node_state)


    def get_params():
        """Set and return the various model parameters."""
    
        params = {}
        params['num_rows'] = 113
        params['num_cols'] = 127
        params['plot_interval'] = 1.0e9
        params['output_interval'] = 1.0e99
        params['run_duration'] = 1.0e10
        params['report_interval'] = 5.0  # report interval, in real-time seconds
        params['plot_every_transition'] = False
    
        return params





if __name__=='__main__':
    ctsm = CTSModel()
    ctsm.run_for(1.0)
