import numpy as np
import os.path
import pandas as pd
import sys
#find parent directory and import base (travis)
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)
from base.uber_model import UberModel, ModelSharedInputs

#print(sys.path)
#print(os.path)

class MaxsusInputs(ModelSharedInputs):
    """
    Input class for Maxsus.
    """

    def __init__(self):
        """Class representing the inputs for Maxsus"""
        super(MaxsusInputs, self).__init__()
        self.init_pop_size = pd.Series([], dtype="float")
        self.growth_rate_percent = pd.Series([], dtype="float")
        self.time_steps = pd.Series([], dtype="float")
        self.K = pd.Series([], dtype="float")


class MaxsusOutputs(object):
    """
    Output class for Maxsus.
    """

    def __init__(self):
        """Class representing the outputs for Maxsus"""
        super(MaxsusOutputs, self).__init__()
        self.out_pop_time_series = pd.Series(name="out_pop_time_series")
        self.out_max_sus_harvest = pd.Series(name="out_max_sus_harvest")


class Maxsus(UberModel, MaxsusInputs, MaxsusOutputs):
    """
    Maxsus model for population growth.
    """

    def __init__(self, pd_obj, pd_obj_exp):
        """Class representing the Maxsus model and containing all its methods"""
        super(Maxsus, self).__init__()
        self.pd_obj = pd_obj
        self.pd_obj_exp = pd_obj_exp
        self.pd_obj_out = None

    def execute_model(self):
        """
        Callable to execute the running of the model:
            1) Populate input parameters
            2) Create output DataFrame to hold the model outputs
            3) Run the model's methods to generate outputs
            4) Fill the output DataFrame with the generated model outputs
        """
        self.populate_inputs(self.pd_obj, self)
        self.pd_obj_out = self.populate_outputs(self)
        self.run_methods()
        self.fill_output_dataframe(self)

    # Begin model methods
    def run_methods(self):
        """ Execute all algorithm methods for model logic """
        try:
            self.maxsus_grow()
        except Exception as e:
            print(str(e))

    def maxsus_grow(self):
        index_set = range(self.K+1)
        x = np.zeros((len(index_set), 2))
        growth_rate = self.growth_rate_percent / 100
        self.out_max_sus_harvest = growth_rate * self.K/4
        for n in range(1, self.K+1):
            x[n][0] = n
            x[n][1] = growth_rate*n*(1-float(n)/self.K)
        self.out_pop_time_series = x.tolist()

        return self.out_max_sus_harvest, self.out_pop_time_series
