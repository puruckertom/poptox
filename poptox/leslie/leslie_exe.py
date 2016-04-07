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
        self.out_fecundity = pd.Series(name="out_fecundity")
        self.out_growth = pd.Series(name="out_growth")
        self.out_survival = pd.Series(name="out_survival")
        self.out_eigdom = pd.Series(name="out_eigdom")
        self.out_eigleft = pd.Series(name="out_eigleft")
        self.out_eigright = pd.Series(name="out_eigright")
        self.out_sensitivity = pd.Series(name="out_sensitivity")
        self.out_elasticity = pd.Series(name="out_elasticity")


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
            self.batch_extract_fec()
            self.batch_extract_growth()
            self.batch_extract_survival()
            self.batch_eigen()
            self.batch_sensitivity()
            self.batch_elasticity()
            self.batch_leslie_grow()
        except Exception as e:
            print(str(e))

    def extract_fec(self, idx):
        """ Method to subset fecundity values from first row of Leslie/Lefkovitch matrix """
        index_set = range(self.time_steps[idx] + 1)
        x = np.zeros(shape=(len(index_set)))
        for n in index_set[1:]:
            x[n] = self.l_m[n-1, :]
            self.out_fecundity[:, n] = x.squeeze()
        t = range(0, self.time_steps[idx])
        d = dict(zip(t, x))
        self.out_fecundity[idx].append(d)
        return

    def batch_extract_fec(self):
        for idx in enumerate(self.l_m):
            self.extract_fec(idx)
        return

    def extract_growth(self, idx):
        """ Method to extract growth probability from Leslie/Lefkovitch matrix"""
        index_set = range(self.time_steps[idx] + 1)
        x = np.zeros(shape=len(self.stages)-1)
        for n in index_set[1:]:
            for k in [0, np.ndim(self.stages) - 1]:
                g = self.l_m[k + 1, k]
                x[k] = g
            self.out_growth[n] = x.squeeze()
        t = range(0, self.time_steps[idx])
        d = dict(zip(t, x))
        self.out_growth[idx].append(d)
        return

    def batch_extract_growth(self):
        for idx in enumerate(self.l_m):
            self.extract_growth(idx)
        return

    def extract_survival(self, idx):
        """Method to extract survival probability from Leslie/Lefkovitch model"""
        index_set = range(self.time_steps[idx] + 1)
        x = np.zeros(shape=len(self.stages))
        for n in index_set[1:]:
            for k in [0, self.stages]:
                s = self.l_m[k, k]
                x[k] = s
            self.out_survival[n] = x.squeeze()
        t = range(0, self.time_steps[idx])
        d = dict(zip(t, x))
        self.out_survival[idx].append(d)
        return

    def batch_extract_survival(self):
        for idx in enumerate(self.l_m):
            self.extract_survival(idx)
        return

    def eigen(self, idx):
        """ Calc dominant eigvalue (expected pop growth @ SSD), right eigvec (Stable Stage Distribution), and
         left eigvec (reproductive value)"""
        index_set = range(self.time_steps[idx] + 1)
        eigdom = np.zeros(shape=(self.stages, len(index_set)))
        eigleft = np.zeros(shape=len(self.stages))
        eigright = np.zeros(shape=len(self.stages))
        for n in index_set[1:]:
            eig_val, eig_vec = np.linalg.eig(self.l_m[n])
            eigdom[n] = np.max(abs(eig_val))
            eigleft[n] = eig_vec[0, :]
            eigright[n] = eig_vec[1, :]
        t = range(0, self.time_steps[idx])
        d = dict(zip(t, eigdom))
        e = dict(zip(t, eigleft))
        f = dict(zip(t, eigright))
        self.out_eigdom[idx].append(d)
        self.out_eigleft[idx].append(e)
        self.out_eigright[idx].append(f)
        return

    def batch_eigen(self):
        for idx in enumerate(self.l_m):
            self.eigen(idx)
        return

    def sensitivity(self, idx):
        """ Calculate sensitivity by taking partial derivatives of Leslie matrix"""
        index_set = range(self.time_steps[idx] + 1)
        x = np.zeros(shape=(self.stages, len(index_set)))
        for n in index_set[1:]:
            for k in [0, np.ndim(self.stages)-1]:
                prod = np.zeros(self.stages)
                prod[k-1] = self.out_eigleft[k] * self.out_eigright[k]
                temp = np.sum(prod)
                g = (self.out_eigleft * self.out_eigright) / temp
            self.out_sensitivity[n] = g
        t = range(0, self.time_steps[idx])
        d = dict(zip(t, x))
        self.out_sensitivity[idx].append(d)
        return

    def batch_sensitivity(self):
        for idx in enumerate(self.time_steps):
            self.sensitivity(idx)
        return

    def elasticity(self, idx):
        """ Calculate elasticity of Leslie matrix"""
        index_set = range(self.time_steps[idx] + 1)
        x = np.zeros(shape=(self.stages, len(index_set)))
        for n in index_set[1:]:
            x[n] = self.out_sensitivity[n] * (self.l_m[n]/self.out_eigdom[n])
        t = range(0, self.time_steps[idx])
        d = dict(zip(t, x))
        self.out_elasticity[idx].append(d)
        return

    def batch_elasticity(self):
        for idx in enumerate(self.l_m):
            self.elasticity(idx)
        return

    def leslie_grow(self, idx):
        index_set = range(self.time_steps[idx] + 1)
        x = np.zeros(shape=(self.stages, len(index_set)))
        self.out_pop_matrix[:, 0] = self.init_pop_size[idx]
        for n in index_set[1:]:
            x[n] = np.dot(self.l_m, self.out_pop_matrix[:, n-1])
            self.out_pop_matrix[:, n] = x.squeeze()
        t = range(0, self.time_steps[idx])
        d = dict(zip(t, x))
        self.out_pop_matrix[idx].append(d)
        return

    def batch_leslie_grow(self):
        for idx in enumerate(self.init_pop_size):
            self.leslie_grow(idx)
        return
