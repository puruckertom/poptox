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

class ExponentialInputs(ModelSharedInputs):
    """
    Input class for Exponential.
    """

    def __init__(self):
        """Class representing the inputs for Exponential"""
        super(ExponentialInputs, self).__init__()
        self.init_pop_size = pd.Series([], dtype="float")
        self.growth_rate = pd.Series([], dtype="float")
        self.time_steps = pd.Series([], dtype="float")


class ExponentialOutputs(object):
    """
    Output class for Exponential.
    """

    def __init__(self):
        """Class representing the outputs for Exponential"""
        super(ExponentialOutputs, self).__init__()
        self.out_pop_time_series = pd.Series(name="out_pop_time_series")


class Exponential(UberModel, ExponentialInputs, ExponentialOutputs):
    """
    Exponential model for population growth.
    """

    def __init__(self, pd_obj, pd_obj_exp):
        """Class representing the Exponential model and containing all its methods"""
        super(Exponential, self).__init__()
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
            self.exponential_growth()
        except Exception as e:
            print(str(e))

    def exponential_growth(self):
        index_set = range(self.time_steps + 1)
        x = np.zeros(len(index_set))
        x[0] = self.init_pop_size
        for n in index_set[1:]:
            x[n] = self.init_pop_size * np.exp(self.growth_rate / 100 * n)
        self.out_pop_time_series = x.tolist()
        return self.out_pop_time_series
