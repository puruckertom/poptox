import numpy as np
import os.path
import pandas as pd
import sys
#find parent directory and import base (travis)
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)
from base.uber_model import UberModel, ModelSharedInputs

# print(sys.path)
# print(os.path)


class LeslieInputs(ModelSharedInputs):
    """
    Input class for Leslie.
    """

    def __init__(self):
        """Class representing the inputs for Leslie"""
        super(LeslieInputs, self).__init__()
        self.init_pop_size = pd.Series([], dtype="float")
        self.stages = pd.Series([], dtype="float")
        self.l_m = pd.Series([], dtype="float")
        self.time_steps = pd.Series([], dtype="float")


class LeslieOutputs(object):
    """
    Output class for Leslie.
    """

    def __init__(self):
        """Class representing the outputs for Leslie"""
        super(LeslieOutputs, self).__init__()
        self.out_pop_matrix = pd.Series(name="out_pop_matrix")


class Leslie(UberModel, LeslieInputs, LeslieOutputs):
    """
    Leslie model for population growth.
    """

    def __init__(self, pd_obj, pd_obj_exp):
        """Class representing the Leslie model and containing all its methods"""
        super(Leslie, self).__init__()
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
            self.leslie_grow()
        except Exception as e:
            print(str(e))

    def leslie_grow(self):
        self.out_pop_matrix = np.zeros(shape=(self.stages, self.time_steps))
        self.out_pop_matrix[:, 0] = self.init_pop_size
        for i in range(1, self.time_steps):
            n = np.dot(self.l_m, self.out_pop_matrix[:, i-1])
            self.out_pop_matrix[:, i] = n.squeeze()
        return self.out_pop_matrix.tolist()
