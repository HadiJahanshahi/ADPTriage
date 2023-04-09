from logging import exception
import numpy    as np 
from gurobipy import GRB, Model
from torch import normal

class Decision:
    def __init__(self, system_state, v_post_b_n_1, v_post_d_n_1, t, method, idea, path_length = 30, gamma = 0.9,
                 early_asg_cost=True, f_notserv = 50, min_fixing_cost=1, cost_f='linear', type_='integer', normalized=None):
        self.type           = type_.lower()  # type continuous is needed for dual values.
        self.pbt            = {}             # num of bugs with attr b postponed to time t+1
        self.ydbt           = {}             # num of bugs with attr b solved by dev with attr d
        self.hdt            = {}             # The slack of the first constraint
        self.early_asg_cost = early_asg_cost # Whether to impose a cost for early assignments
        self.gamma          = gamma
        self.model          = Model()        # Initializing the IP model
        self.method         = method.lower() # Whether to use Myopic or ADP
        assert self.method in ['adp', 'myopic']
        self.idea           = idea           # Is it idea `a` (sophisticated) or `b` (simplified)?
        self.feasible       = "None"         # T or F: at least a bug and developer exist or not.
        self.cost_f         = cost_f.lower() # How should postponement cost looks
        self.f_notserv      = f_notserv      # coefficient of not serving an order (used for the simplified version)
        self.min_fixing_cost= min_fixing_cost# The initial value of v bar. Instead of initializing by zero, we do it by min fixing cost.
        self.system_state   = system_state
        self.v_post_b_n_1   = v_post_b_n_1   # last v of post decision state of bugs
        self.v_post_d_n_1   = v_post_d_n_1   # last v of post decision state of developers
        self.path_length    = path_length    # The total number of days which encompass one epoch
        self.t              = t              # current timestamp
        self.objective_coef = {}             # Recording the coefficients of the objective function
        if self.t == self.path_length:
            self.last_step = True            # Whether it is the last epoch
        else:
            self.last_step = False
        if normalized == None:
            self.normalized = normalized
        else:
            self.normalized = normalized.lower()
        self.calculate_coefficients() # experience and due-date coefficient calculation
        if self.feasible:
            self.feasible_actions()
            
    def save_objective_coef(self, var_:str, coef_:float):
        """Recording the coefficients of the objective function in case we need to 
        manually compute the objective function value.

        Args:
            var_ (str)   : The variable name
            coef_ (float): The coefficient of that variable
        """
        if var_ not in self.objective_coef:
            self.objective_coef[var_] = coef_
        else:
            assert self.objective_coef[var_] == coef_
    
    def calculate_coefficients(self):
        """This function computes the coefficients of the objective function.

        """
        exp_ls              = [] # list of developers' experiences
        b_types             = [] # list of bugs' types
        b_dues              = [] # list of bugs' due dates
        self.available_bugs = [] # all open bugs
        self.available_devs = [] # all available developers
        for (b_type, b_due) in self.system_state.S_t_b_bug.keys():
            self.available_bugs.append((b_type, b_due))
            if b_type not in b_types:
                # finding all possible bug types at time t
                b_types.append(b_type)
            if b_due not in b_dues:
                # finding all bug due dates (absolute values)
                b_dues.append(abs(b_due))
                
        for key_d, available in self.system_state.S_t_d_dev.items():
            if self.idea == 'a':
                dev_exp = key_d[0]
                dev_sch = key_d[1]
            else:
                dev_exp = key_d
                dev_sch = None # no schedule in the simplistic format.
            if (available) or ((self.idea == 'a') and (dev_sch == 1)):
                if (available and (self.idea == 'a')): assert dev_sch == 0
                self.available_devs.append(key_d)
                # finding all possible dev experience at time t
                exp_ls.extend(list(np.array(dev_exp)[b_types]))
                # experience list is sorted based on bugs' types
                
        if (len(exp_ls)>0) and (len(b_dues)>0):
            self.feasible           = True # if we have at least one bug and one developer, then we can build a model
            exp_max                 = max(exp_ls) # maximum devs' experiences for standardization
            self.f_exp_type         = {} # Normalized Experience Coefficients
            self.f_due_t            = {} # Normalized Due date Coefficients
            self.f_exp_type_NOTnorm = {} # NOT Normalized Experience Coefficients
            self.f_due_t_NOTnorm    = {} # NOT Normalized Due date Coefficients
            for key_d in self.available_devs:
                if self.idea == 'a':
                    dev_exp = key_d[0]
                    dev_sch = key_d[1]
                else:
                    dev_exp = key_d
                    dev_sch = None # no schedule in the simplistic format.
                for (bug_type, bug_due) in self.available_bugs:
                    try:
                        # normalizing experiences and due dates by their max value
                        self.f_exp_type_NOTnorm[(bug_type, bug_due), key_d] = dev_exp[bug_type]
                        self.f_exp_type[(bug_type, bug_due), key_d]         = dev_exp[bug_type] / exp_max
                        max_                                                = max([abs(i) for i in b_dues])
                        self.f_due_t_NOTnorm[(bug_type, bug_due)]           = abs(bug_due)
                        if self.idea == 'a':
                            if   self.cost_f == 'linear':
                                if self.normalized == 'max':
                                    self.f_due_t[(bug_type, bug_due)]       = ((self.path_length - bug_due) / self.path_length) / max_
                                elif self.normalized == 'project_horizon':
                                    self.f_due_t[(bug_type, bug_due)]       = ((self.path_length - bug_due) / self.path_length) / self.path_length
                                elif self.normalized == None:
                                    self.f_due_t[(bug_type, bug_due)]       = ((self.path_length - bug_due) / self.path_length) 
                                else:
                                    raise Exception (f'Normalized {self.normalized} is not covered.')
                            elif self.cost_f == 'exponential':
                                self.f_due_t[(bug_type, bug_due)] = 0.9**bug_due
                            else:
                                raise Exception (f'Cost function {self.cost_f} is not defined.')
                        else: # simplistic case
                            if   bug_due > 0:
                                self.f_due_t[(bug_type, bug_due)] = 0
                            elif bug_due == 0:
                                self.f_due_t[(bug_type, bug_due)] = self.f_notserv
                            else:
                                raise Exception (f'bug due cannot be negative in the simplified idea case.')
                    except ZeroDivisionError:
                        if self.idea == 'a':
                            if   self.cost_f == 'linear':
                                self.f_due_t[(bug_type, bug_due)]  = ((self.path_length - bug_due) / self.path_length) 
                            elif self.cost_f == 'exponential':
                                self.f_due_t[(bug_type, bug_due)]  = 0.9**bug_due
                            else:
                                raise Exception (f'Cost function {self.cost_f} is not defined.')
                        else: # simplistic case
                            if   bug_due > 0:
                                self.f_due_t[(bug_type, bug_due)] = 0
                            elif bug_due == 0:
                                self.f_due_t[(bug_type, bug_due)] = self.f_notserv
                            else:
                                raise Exception (f'bug due cannot be negative in the simplified idea case.')
        else:
            self.feasible   = False
        
    def feasible_actions(self, LogToConsole= False, TimeLimit=60, write=False):
        """IP model to find the optimal actions/assignments
        
        As we need to also get the dual values, we relax the IP by considering 
        real numbers instead of integer values. This is necessary for dual calculation.
        The gap of original and relaxed models are recorded.

        Args:
            LogToConsole (bool, optional): Whether to print the IP output. Defaults to False.
            TimeLimit (int, optional):     Max time to solve the model. Defaults to 60.
            write (bool, optional):        Whether to save the IP model as a file. Defaults to False.
        """
        self.model.params.LogToConsole = LogToConsole
        self.model.params.TimeLimit    = TimeLimit
        
        """ 
        Variables
        """
        for (bug_type, bug_due) in self.available_bugs:
            if self.type == 'integer':
                self.pbt[(bug_type, bug_due)] = (self.model.addVar(vtype= GRB.INTEGER, name=f'p[{(bug_type, bug_due)}]'))
                # if (self.method.lower() == 'adp') and ((bug_type, bug_due-1) not in self.pbt):
                #     self.pbt[(bug_type, bug_due-1)] = (self.model.addVar(vtype= GRB.INTEGER, 
                #                                                          name=f'p[{(bug_type, bug_due-1)}]'))                    
            elif self.type == 'continuous':
                self.pbt[(bug_type, bug_due)] = (self.model.addVar(vtype= GRB.CONTINUOUS, name=f'p[{(bug_type, bug_due)}]'))
                # if (self.method.lower() == 'adp') and ((bug_type, bug_due-1) not in self.pbt):
                #     self.pbt[(bug_type, bug_due-1)] = (self.model.addVar(vtype= GRB.CONTINUOUS, 
                #                                                          name=f'p[{(bug_type, bug_due-1)}]'))                    
            else:
                raise Exception(f'Incorrect Type {self.type}')
            for key_d in self.available_devs:
                if self.type == 'integer':
                    self.ydbt[(bug_type, bug_due), key_d] = (self.model.addVar(vtype= GRB.INTEGER,
                                                                               name=f'y[{(bug_type, bug_due)},{key_d}]'))
                elif self.type == 'continuous':
                    self.ydbt[(bug_type, bug_due), key_d] = (self.model.addVar(vtype= GRB.CONTINUOUS,
                                                                               name=f'y[{(bug_type, bug_due)},{key_d}]'))
                else: 
                    raise Exception(f'Incorrect Type {self.type}')
        for key_d in self.available_devs:
            # slack variable
            if self.type == 'integer':
                self.hdt[key_d] = (self.model.addVar(vtype= GRB.INTEGER, name=f'h[{key_d}]'))
            elif self.type == 'continuous':
                self.hdt[key_d] = (self.model.addVar(vtype= GRB.CONTINUOUS, name=f'h[{key_d}]'))
            else: 
                raise Exception(f'Incorrect Type {self.type}')
            
        """ 
        Objective Function
        """
        obj_func = 0
        for (bug_type, bug_due) in self.available_bugs:
            # if (bug_due < 0) and (self.cost_f == 'linear') and (self.method != 'myopic'): # and (self.last_step)  
            #     # If we exceed the overdue date, we impose a cost for postponement.
            #     # We want to force the model to assign the bug on time, rather than too late.
            #     self.save_objective_coef(f'f_due_t({bug_type}, {bug_due})', self.f_due_t[(bug_type, bug_due)])
            #     obj_func += self.f_due_t[(bug_type, bug_due)] * self.pbt[(bug_type, bug_due)]
            # elif (self.cost_f == 'exponential') and (self.method != 'myopic'):
            #     self.save_objective_coef(f'f_due_t({bug_type}, {bug_due})', self.f_due_t[(bug_type, bug_due)])
            #     obj_func += self.f_due_t[(bug_type, bug_due)] * self.pbt[(bug_type, bug_due)]                
            if (self.method == 'adp'): # and (bug_due >= 0)
                # We impose a cost for postponement based on how far we are from the due date.
                # We want to force the model to assign the bug on time, rather than too late.
                self.save_objective_coef(f'f_due_t({bug_type}, {bug_due})', self.f_due_t[(bug_type, bug_due)])
                obj_func += self.f_due_t[(bug_type, bug_due)] * self.pbt[(bug_type, bug_due)]
                self.save_objective_coef(f'v_post_b_n_1({bug_type}, {bug_due})', 
                                         self.gamma * self.v_post_b_n_1.get((bug_type, bug_due),[self.min_fixing_cost])[-1])
                obj_func += self.gamma * self.v_post_b_n_1.get((bug_type, bug_due),[self.min_fixing_cost])[-1] * self.pbt[(bug_type, bug_due)]
            elif (self.method == 'myopic'):
                # We set a large cost for the postponement to enforce the model to assign the current developers.
                self.save_objective_coef(f'v_post_b_n_1({bug_type}, {bug_due})', 
                                         self.f_notserv * 5)
                obj_func += self.f_notserv * 5 * self.pbt[(bug_type, bug_due)]
            for key_d in self.available_devs:
                self.save_objective_coef(f'fixing(({bug_type}, {bug_due}), {key_d})', 
                                         self.f_exp_type_NOTnorm[(bug_type, bug_due), key_d])                
                if self.normalized == None:
                    # No normalization
                    self.save_objective_coef(f'f_exp_type(({bug_type}, {bug_due}), ({key_d}))', 
                                            self.f_exp_type_NOTnorm[(bug_type, bug_due), key_d])
                    obj_func += self.f_exp_type_NOTnorm[(bug_type, bug_due), key_d] * self.ydbt[(bug_type, bug_due), key_d]
                else:
                    self.save_objective_coef(f'f_exp_type(({bug_type}, {bug_due}), ({key_d}))', 
                                            self.f_exp_type[(bug_type, bug_due), key_d])
                    obj_func += self.f_exp_type[(bug_type, bug_due), key_d] * self.ydbt[(bug_type, bug_due), key_d]                    
                if (bug_due >= 0) and (self.early_asg_cost) and (self.cost_f == 'linear'):
                    # We want to force the model to assign the bug on time, rather than too early.
                    # If the early_assignment_cost is True, we enforce that cost to the obj func.
                    self.save_objective_coef(f'f_due_t({bug_type}, {bug_due})/2', 
                                               self.f_due_t[(bug_type, bug_due)]/2)
                    obj_func +=    (self.f_due_t[(bug_type, bug_due)]/2) * self.ydbt[(bug_type, bug_due), key_d]
                self.model.update()
                self.model.setObjective(obj_func, GRB.MINIMIZE)
        if (self.method == 'adp') and (self.idea == 'a'):
            for key_d in self.available_devs:
                dev_sch = key_d[1]
                Std_Dev_Post = 0
                if dev_sch == 0:   # They are available now
                    Std_Dev_Post += self.hdt[key_d]
                elif dev_sch == 1: # They become available by tomorrow
                    Std_Dev_Post += self.system_state.S_t_d_dev[key_d]
                else:
                    raise Exception(f"Developer Schedule of available ones cannot be equal to {dev_sch}.")
                # .get(key,0) returns the key's value if exists, otherwise, it returns 0.
                # initial value which is 0.
                v_hat_d       = self.v_post_d_n_1.get(key_d, [self.min_fixing_cost])[-1]
                self.save_objective_coef(f'h(({key_d}))', self.gamma * v_hat_d)
                obj_func     += self.gamma * v_hat_d * Std_Dev_Post
        self.model.setObjective(obj_func, GRB.MINIMIZE)
        
        """ 
        Constraint 1
        """
        for key_d  in self.available_devs:
            n_assigned_devs_constraint = 0
            for (bug_type, bug_due) in self.available_bugs:
                n_assigned_devs_constraint += self.ydbt[(bug_type, bug_due), key_d]
            n_assigned_devs_constraint += self.hdt[key_d] # slack variable
            self.model.addConstr(n_assigned_devs_constraint == self.system_state.S_t_d_dev[key_d],
                                 name = "eq:assigned_devs")
        """ 
        Constraint 2
        """
        for (bug_type, bug_due) in self.available_bugs:
            n_assigned_bugs_constraint = 0
            for key_d in self.available_devs:
                n_assigned_bugs_constraint += self.ydbt[(bug_type, bug_due), key_d]
            n_assigned_bugs_constraint += self.pbt[(bug_type, bug_due)]
            self.model.addConstr(n_assigned_bugs_constraint == self.system_state.S_t_b_bug[(bug_type, bug_due)],
                                 name = "eq:assigned_bugs")
        """ 
        Optimizing / Solving / Saving the model
        """            
        self.model.update()
        if write:
            self.model.write('ADP.lp')
        self.model.optimize()
        
    def calculate_pure_obj_func(self):
        """
        After optimizing the model, it can compute the objective function value 
        without considering the estimated v* part of the objective function.
        This is actually the immediate reward only.
        """
        obj_value = 0
        for vy_index, vy_value in self.ydbt.items():
            if round(vy_value.X)>0:
                bug_type_ = vy_index[0][0]
                bug_due_  = vy_index[0][1]
                key_d     = vy_index[1]
                obj_value += self.objective_coef[f'f_exp_type(({bug_type_}, {bug_due_}), ({key_d}))'] * vy_value.X
                if (bug_due_ >= 0) and (self.early_asg_cost) and (self.cost_f == 'linear'):
                    obj_value += self.objective_coef[f'f_due_t({bug_type_}, {bug_due_})/2'] * vy_value.X
        for (bug_type, bug_due), vp_value in self.pbt.items():
            if round(vp_value.X)>0:
                # if (bug_due < 0) and (self.cost_f == 'linear'):
                #     obj_value += self.objective_coef[f'f_due_t({bug_type}, {bug_due})'] * vp_value.X
                # elif (self.cost_f == 'exponential'):
                obj_value += self.objective_coef[f'f_due_t({bug_type}, {bug_due})'] * vp_value.X
        return obj_value