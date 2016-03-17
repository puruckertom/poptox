import numpy as np
import os.path
import pandas as pd
import sys
import math 

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

        self.a_n = pd.Series([], dtype="object")
        self.c_n = pd.Series([], dtype="object")
        self.app_target = pd.Series([], dtype="object")
        self.ai = pd.Series([], dtype="float")
        self.hl = pd.Series([], dtype="float")
        self.sol = pd.Series([], dtype="float")
        self.time_steps = pd.Series([], dtype="float")
        self.n_a = pd.Series([], dtype="float")
        self.rate_out = pd.Series([], dtype="float")
        self.day_out = pd.Series([], dtype="float")
        self.b = pd.Series([], dtype="float")
        self.test_species = pd.Series([], dtype="object")
        self.ld50_test = pd.Series([], dtype="float")
        self.bw_test = pd.Series([], dtype="float")
        self.ass_species = pd.Series([], dtype="object")
        self.bw_ass = pd.Series([], dtype="float")
        self.mineau_scaling_factor = pd.Series([], dtype="float")
        self.probit_gamma = pd.Series([], dtype="float")
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
            self.leslie_probit_growth()
        except Exception as e:
            print(str(e))

    def leslie_growth(self):
        self.out_pop_matrix = np.zeros(shape=(self.stages, self.time_steps))
        self.out_pop_matrix[:, 0] = self.init_pop_size
        for i in range(1, self.time_steps):
            n = np.dot(self.l_m, self.out_pop_matrix[:, i - 1])
            self.out_pop_matrix[:, i] = n.squeeze()
        return self.out_pop_matrix.tolist()

    def leslie_probit_growth(self):
        self.conc_out = self.conc()
        self.out = self.dose_bird()
        self.out_no = self.no_dose_bird()
        return
    
    

    def conc(self):
        #Concentration over time
        def C_0(r, a, p):
            return r * a / 100 * p
        
        def C_t(c, h):    
            return c * np.exp(-(np.log(2) / h) * 1)

        if self.n_a == 1:
            C_temp = C_0(self.rate_out[0], self.ai, self.para)
        else:
            C_temp = [] #empty array to hold the concentrations over days       
            n_a_temp = 0  #number of existing applications
            dayt = 0
            day_out_l=len(self.day_out)
            for i in range (0,self.time_steps):
                if i==0:  #first day of application
                    C_temp.append(C_0(self.rate_out[0], self.ai, self.para))
                    n_a_temp = n_a_temp + 1
                    dayt = dayt + 1
                elif dayt<=day_out_l-1 and n_a_temp<=self.n_a: # next application day
                    if i==self.day_out[dayt]:
                        C_temp.append(C_t(C_temp[i-1], self.hl) + C_0(self.rate_out[dayt], self.ai, self.para))
                        n_a_temp = n_a_temp + 1
                        dayt = dayt + 1        
                    else :
                        C_temp.append(C_t(C_temp[i-1], self.hl))
                else:
                    C_temp.append(C_t(C_temp[i-1], self.hl) )
        return C_temp


    def dose_bird(self):
        ####Initial Leslie Matrix and pesticide conc###########
        S = self.l_m.shape[1]
        n_f=np.zeros(shape=(S,self.t))

        l_m_temp=np.zeros(shape=(S,S), dtype=float)
        n_csum=np.sum(self.init_pop_size)
        n_f[:,0]=self.init_pop_size.squeeze()

        fw_bird = (1.180 * (self.aw_bird**0.874))/1000.0
        m=[]
        dose_out = []
        z_out = []

        for i in range(self.t):
            # C_temp = C_temp*np.exp(-(np.log(2)/h_l)*1)
            C_temp = self.conc_all[i]
            if C_temp >= self.sol:
                dose_bird = (fw_bird * C_temp)/(self.aw_bird / 1000)
            else:
                dose_bird = (fw_bird * C_temp[0])/(self.aw_bird / 1000)
            at_bird = (self.ld50_a) * ((self.aw_bird/self.bw_bird)**(self.mineau_scaling_factor-1))
            # print at_bird
            z = self.b*(np.log10(dose_bird)-np.log10(at_bird))
            m_temp = 1-0.5*(1+math.erf(z/1.4142))

            for j in range(0, S):
                l_m_temp[0,j]=self.l_m[0,j]*np.exp(-self.probit_gamma*n_csum)
                if j-1>=0:
                    l_m_temp[j,j-1]=self.l_m[j,j-1]*m_temp
                    l_m_temp[S-1,S-1]=self.l_m[S-1,S-1]*m_temp

            n=np.dot(l_m_temp, init_pop_size)
            n_csum=np.sum(n)
            init_pop_size=n
            n_f[:,i]=n.squeeze()

            m.append(m_temp)
            dose_out.append(dose_bird)
            z_out.append(z)

        return fw_bird, dose_out, at_bird, m, n_f.tolist(), z_out


    def no_dose_bird(self):
        ####Initial Leslie Matrix and pesticide conc###########
        S = self.l_m.shape[1]
        n_f=np.zeros(shape=(S,self.t))
        n_f[:,0]=self.init_pop_size.squeeze()
        for i in range(self.t):
            n=np.dot(self.l_m, init_pop_size)
            init_pop_size=n
            n_f[:,i]=n.squeeze()
        return n_f.tolist()
