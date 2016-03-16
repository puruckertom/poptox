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


class LeslieProbitInputs(ModelSharedInputs):
    """
    Input class for LeslieProbit.
    """

    def __init__(self):
        """Class representing the inputs for LeslieProbit"""
        super(LeslieProbitInputs, self).__init__()

        self.animal_name = pd.Series([], dtype="object")
        self.chemical_name = pd.Series([], dtype="object")
        self.app_target = pd.Series([], dtype="object")
        self.ai = pd.Series([], dtype="float")
        self.half_life = pd.Series([], dtype="float")
        self.sol = pd.Series([], dtype="float")
        self.time_steps = pd.Series([], dtype="float")
        self.tot_num_app = pd.Series([], dtype="float")
        self.app_num = pd.Series([], dtype="float")
        self.app_rate = pd.Series([], dtype="float")
        self.app_day = pd.Series([], dtype="float")
        self.b = pd.Series([], dtype="float")
        self.test_species = pd.Series([], dtype="object")
        self.ld50_test = pd.Series([], dtype="float")
        self.bw_test = pd.Series([], dtype="float")
        self.ass_species = pd.Series([], dtype="object")
        self.bw_ass = pd.Series([], dtype="float")
        self.min_scale = pd.Series([], dtype="float")
        self.dd = pd.Series([], dtype="float")
        self.init_pop_size = pd.Series([], dtype="float")
        self.stages = pd.Series([], dtype="float")
        self.l_m = pd.Series([], dtype="float")




class LeslieProbitOutputs(object):
    """
    Output class for LeslieProbit.
    """

    def __init__(self):
        """Class representing the outputs for LeslieProbit"""
        super(LeslieProbitOutputs, self).__init__()
        self.out_pop_matrix = pd.Series(name="out_pop_matrix")


class LeslieProbit(UberModel, LeslieProbitInputs, LeslieProbitOutputs):
    """
    LeslieProbit model for population growth.
    """

    def __init__(self, pd_obj, pd_obj_exp):
        """Class representing the LeslieProbit model and containing all its methods"""
        super(LeslieProbit, self).__init__()
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
            self.leslie_probit_grow()
        except Exception as e:
            print(str(e))

    def leslie_probit_grow(self):
        self.out_pop_matrix = np.zeros(shape=(self.stages, self.time_steps))
        self.out_pop_matrix[:, 0] = self.init_pop_size
        for i in range(1, self.time_steps):
            n = np.dot(self.l_m, self.out_pop_matrix[:, i-1])
            self.out_pop_matrix[:, i] = n.squeeze()
        return self.out_pop_matrix.tolist()
