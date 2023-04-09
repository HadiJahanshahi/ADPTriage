from BDG.BDG import BDG
from components.assignee_prob import Dev_prob
from components.assignee_simulated import Developer
from components.bug_prob import Bug_prob
from components.bug_simulated import Bug
from components.decision import Decision
from components.state import State
from utils.functions import Functions
from utils.prerequisites import *

from simulator.first_run import Finding_parameters_training
from simulator.second_run import estimating_state_transitions


class ADP():
    random_state = 0
    def __init__(self, dev_probability:Dev_prob, bug_probability:Bug_prob, project:str, LDA_table, 
                 normalized:str=None, alpha_update:str='constant', alpha:str=0.5, f_notserv:float=None, 
                 path_length:int=30, mode:str='training', v_star_coef_b:list=None, v_star_coef_d:list=None, 
                 gamma:float= 0.9, epsilon:float= 1.01, cost_f:str= 'exponential', method:str='ADP', 
                 toy_example:bool=False, idea:str='a', early_asg_cost=True, verbose = 2):
        """ ADP algorithm

        Args:
            dev_probability (Dev_prob): The probability of having developers available in each epoch
            bug_probability (Bug_prob): The probability of having each bug type in each epoch 
            project (str): The name of the project
            LDA_table (_type_): The LDA table related to the developers' experience
            normalized (str, optional): How to normalize cost values. Defaults to None. Other options:'project_horizon', 'max'
            alpha_update (str, optional): Whether to update alpha using constant, harmonic, and BAKF method. Defaults to 'constant'.
            alpha (str, optional): default value for alpha in `constant` mode of `alpha_update`. Defaults to 0.5.
            f_notserv (float, optional): cost of not fixing a bug on time (Not used now). Defaults to 5.
            path_length (int, optional): The number of inner epochs. Defaults to 30.
            mode (str, optional): Whether it is the training or testing phase. Defaults to 'training'.
            v_star_coef_b (list, optional): Optimal v values for bugs (needed for testing). Defaults to None.
            v_star_coef_d (list, optional): Optimal v values for developers (needed for testing). Defaults to None.
            gamma (float, optional): discounting factor in Bellman equation. Defaults to 0.9.
            epsilon (float, optional): epsilon greedy for taking random action. Defaults to 1, meaning no random action.
            cost_f (str, optional): How the cost function for late assignment should be. Defaults to exponential.
            idea (str, optional): We have ideas a (original) and b (simplified). Defaults to a.
                a. This is more akin to the actual world and more sophisticated. We failed to find a good policy with this.
                b. This is the simplified version of previous one. Developers leave the system and unsolved bugs are removed.
            toy_example (bool, optional): Whether it is a toy example or a real problem. Defaults to False.
            method (str, optional): Which method to use, ADP or Myopic. Defaults to 'ADP'.
            early_asg_cost (bool, optional): Whether to Consider a cost for early assignment. Defaults to True.
            verbose (int, optional): Whether to print output or not (Not used now). Defaults to 0.
        """
        if   (verbose == 'nothing') or (str(verbose) == "0"):
            self.verbose = 0
        elif (verbose == 'some')    or (str(verbose) == "1"):
            self.verbose = 1
        elif (verbose == 'all')     or (str(verbose) == "2"):
            self.verbose = 2
        else:
            raise Exception ('Verbose can be chosen from nothing, some, all, 0, 1, or 2.')
        self.toy_example        = toy_example      # Whether it is a toy example.
        if self.toy_example:
            self.project            = 'ToyExample' # ToyExample
            self.log                = ''
        else:
            self.project            = project      # Project Name            
        self.resolved_bugs      = {}               # Keeping track of resolved bugs
        self.dev_probability    = dev_probability  # coming from phase II
        self.bug_probability    = bug_probability  # coming from phase II
        self.day_counter        = 0
        if   idea.lower() in ['a', 'original', 'actual', 'sophisticated']:
            self.idea           = 'a'         # Developers have schedule and bugs remain the system until they get fixed.
        elif idea.lower() in ['b', 'simplified', 'simple']:
            self.idea           = 'b'         # Developers leave the system and bugs leave the system if remain unfixed.
        else:
            raise Exception (f"undefined idea ({idea}). Choose from ['a', 'original', 'sophisticated', 'b', 'simplified']")
        self.alpha              = alpha
        self.alpha_b            = {}          # tracking alpha values per bug state
        self.alpha_d            = {}          # tracking alpha values per developer state
        self.alpha_update       = alpha_update.lower() # How to update alpha
        self.lambda_            = 25          # Based on Martijn Mes and  Arturo Pérez Rivera's paper for harmonic update
        self.step_b             = {}          # v^0 for bakf algorithm (bugs); initial value =  0.01
        self.step_bar_b         = 0.2         # \bar{v} for bakf algorithm (bugs)
        self.beta_b             = {}          # Initial beta for bakf algorithm (bugs); initial value = 0
        self.delta_b            = {}          # Initial delta for bakf algorithm (bugs); initial value = 0
        self.lamb_b             = {}          # Lambda of BAKF algorithm (bugs)
        self.sigma_sq_b         = {}          # sigma squared of BAKF algorithm (bugs)
        self.n_visits_b         = {}          # How many times we visited a state for a bug.
        self.step_d             = {}          # v^0 for bakf algorithm (developers); initial value =  0.01
        self.step_bar_d         = 0.2         # \bar{v} for bakf algorithm (developers)
        self.beta_d             = {}          # Initial beta for bakf algorithm (developers); initial value = 0
        self.delta_d            = {}          # Initial delta for bakf algorithm (developers); initial value = 0
        self.lamb_d             = {}          # Lambda of BAKF algorithm (developers)
        self.sigma_sq_d         = {}          # sigma squared of BAKF algorithm (developers)
        self.n_visits_d         = {}          # How many times we visited a state for a developer.
        self.gamma              = gamma       # The discounting factor for the Bellman equation.
        self.cost_f             = cost_f      # How the cost function of late assignment should look like.        
        if mode.lower() == 'testing':
            self.epsilon        = 1.01        # Only greedy action, i.e., following the policy.
        else:
            self.epsilon        = epsilon     # Taking e-greedy action during the training phase (the percentage of exploration)
        self.developers_profile = {}          # Dictionary of all developers
        self.fixed_bugs         = {}          # Dictionary of all fixed bugs (NOT USED NOW)
        self.path_length        = path_length # Project Horizon
        self.repetition         = 1_000       # The number of repetition of training (N)
        self.LDA_table          = LDA_table   # Table of Developers' experience on each bug type
        if f_notserv != None:
            self.f_notserv      = f_notserv   # NOT used for now
        else:
            self.f_notserv      = np.median(self.LDA_table.iloc[:,1:].values) # set it to median fixing cost
            print(f"The cost of not serving a service is equal to {self.f_notserv}")
        self.min_fixing_cost    = np.min(self.LDA_table.iloc[:,1:].values) # the min fixing cost is used to initialize v*
        self.dual_gap           = []          # The gap between the Objs of the primal and dual models
        self.mode               = mode.lower()# can be either testing or training
        self.early_asg_cost     = early_asg_cost # Whether to impose a cost for early assignments.
        self.v_post_b_n_1_track = {}          # Tracking the V bar for bugs. For demonstration purpose only.
        self.v_post_d_n_1_track = {}          # Tracking the V bar for devs. For demonstration purpose only.
        if v_star_coef_b != None:
            self.v_post_b_n_1   = v_star_coef_b  # coming from the training phase for bugs.
        else:
            self.v_post_b_n_1   = {}             # the observed marginal value at iteration n-1 for bugs
        if v_star_coef_d != None:
            self.v_post_d_n_1   = v_star_coef_d  # coming from the training phase for developers.
        else:
            self.v_post_d_n_1   = {}             # the observed marginal value at iteration n-1 for developers
        self.v_post_b_n         = {}             # the observed marginal value at iteration n for bugs
        self.v_post_d_n         = {}             # the observed marginal value at iteration n for developers
        self.normalized         = normalized
        self.method             = method.lower() # Which method to be used to assign bugs; ADP or Myopic
        if self.mode == 'testing':
            if v_star_coef_b == None:
                raise Exception ('In the testing mode, we need the optimal v values.')
        self.metrics            = {'n':[], 'NumberOfFixedBugs':[], 'NumberOfOpenBugs':[], 'ObjectiveValue':[],
                                   'AverageFixingTime':[], 'DueDates':[], 'AssignedToWhom':[], 'RemainedOpen':[],
                                   'cost_of_unassigned_bugs':[]}
        self.inner_test_metrics = {'n':[], 'FirstObjectiveValue':[], 'discounted_reward':[], 
                                   'CorrectedFirstObjectiveValue':[], 'corrected_discounted_reward':[],
                                   'cost_of_unassigned_bugs':[], 'corrected_discounted_reward+':[]}
        self.path_developer     = [] # the availability of developers for the next `path_length` epochs
        self.path_bug_number    = [] # The number of incoming bugs of each type for the next `path_length` epochs
        self.path_bug_due       = [] # The due date of incoming bugs of each type for the next `path_length` epochs
        
    def log_output(self, text):
        if (self.toy_example) and (self.verbose == 2):
            print(text)
            self.log += f'{text}\n'
        
    def updating_post_decision_state_value (self):
        """Updating Post Decision State Values based on the dual of the first and second constraint.
            * If there is no previous post decision value, 0 is considered.
            * We may have three options to update alpha:
                - Constant,
                - Harmonic, and
                - BAKF.
             Each of them has its own characteristic. 
            
            * Alpha regulates the trade-off between the previous and the current values. The default is set to 0.5.
        """
        
        """ Determining Alpha based on the updating method. """        
        if   self.alpha_update == 'constant':
            alpha_b = alpha_d = self.alpha
            
        elif self.alpha_update == 'harmonic':
            # $\alpha^n=\max\big{\frac{\lambda}{\lambda+n-1}, \alpha^0 \big}$ with $\lambda$=25 and $\alpha^0=0.05$.
            alpha_b = alpha_d = max(self.lambda_ / (self.lambda_ + self.n_exp - 1), 0.05)
            
        elif self.alpha_update == 'bakf':
            for key_b, v_post_n_key_b in self.v_post_b_n.items():
                key_b_1 = (key_b[0], key_b[1]+1) # In the previous step, the due date was 1 day more.
                if key_b not in self.n_visits_b:
                    self.n_visits_b[key_b] = 1
                else: 
                    self.n_visits_b[key_b] += 1
                # the initial value for step_b is 0.01
                self.step_b[key_b] = self.step_b.get(key_b, 0.01) / (1+self.step_b.get(key_b, 0.01) - self.step_bar_b)
                difference_b       = v_post_n_key_b - self.v_post_b_n_1.get(key_b_1, [self.min_fixing_cost])[-1]
                # the initial value for beta_b and delta_b is 0
                self.beta_b[key_b] = (1-self.step_b[key_b])*self.beta_b.get(key_b,0)  + self.step_b[key_b]*difference_b
                self.delta_b[key_b]= (1-self.step_b[key_b])*self.delta_b.get(key_b,0) + self.step_b[key_b]*difference_b**2
                if self.n_visits_b[key_b] == 1: # first visit of the state (self.n_exp == 0)
                    self.alpha_b[key_b]    = 1
                    self.lamb_b[key_b]     = self.alpha_b[key_b]**2
                else:
                    self.sigma_sq_b[key_b] = (self.delta_b[key_b] - (self.beta_b[key_b])**2) / (1 + self.lamb_b[key_b])
                    if self.delta_b[key_b] != 0:
                        self.alpha_b[key_b]    = 1 - (self.sigma_sq_b[key_b] / self.delta_b[key_b])
                    else:
                        self.alpha_b[key_b]    = 1
                    self.lamb_b[key_b]     = (((1-self.alpha_b[key_b])**2)*self.lamb_b[key_b]) + ((self.alpha_b[key_b])**2)
                if pd.isnull(self.alpha_b[key_b]):
                    raise Exception
            if self.idea == "a":
                # we only consider the dual of the first assignment in the more `complicated` version of implementation.
                for key_d, v_post_n_key_d in self.v_post_d_n.items():
                    if key_d not in self.n_visits_d:
                        self.n_visits_d[key_d] = 1
                    else: 
                        self.n_visits_d[key_d] += 1
                    # the initial value for step_d is 0.01
                    self.step_d[key_d] = self.step_d.get(key_d, 0.01) / (1+self.step_d.get(key_d, 0.01) - self.step_bar_d)
                    difference_d       = v_post_n_key_d - self.v_post_d_n_1.get(key_d, [self.min_fixing_cost])[-1]
                    # the initial value for beta_d and delta_d is 0
                    self.beta_d[key_d] = (1-self.step_d[key_d])*self.beta_d.get(key_d,0) + self.step_d[key_d]*difference_d
                    self.delta_d[key_d]= (1-self.step_d[key_d])*self.delta_d.get(key_d,0) + self.step_d[key_d]*difference_d**2
                    if self.n_visits_d[key_d] == 1: # first visit of the state (self.n_exp == 0)
                        self.alpha_d[key_d]    = 1
                        self.lamb_d[key_d]     = self.alpha_d[key_d]**2
                    else:
                        self.sigma_sq_d[key_d] = (self.delta_d[key_d] - (self.beta_d[key_d])**2) / (1 + self.lamb_d[key_d])
                        try:
                            self.alpha_d[key_d]    = 1 - (self.sigma_sq_d[key_d] / self.delta_d[key_d])
                        except ZeroDivisionError:
                            self.alpha_d[key_d]    = 1
                        self.lamb_d[key_d]     = (((1-self.alpha_d[key_d])**2)*self.lamb_d[key_d]) + ((self.alpha_d[key_d])**2)
        else:
            raise Exception (f'Unknown alpha update method {self.alpha_update}.')
        
        """ Update V """
        for key_b, v_post_n_key_b in self.v_post_b_n.items():
            key_b_1 = (key_b[0], key_b[1]+1) # In the previous step, the due date was 1 day more.
            if self.alpha_update == 'bakf':
                # alpha would be different per state
                alpha_b = self.alpha_b[key_b]
            v_temp_b    = ((1-alpha_b) * self.v_post_b_n_1.get(key_b_1, [self.min_fixing_cost])[-1]) + (alpha_b * v_post_n_key_b)
            if key_b_1 not in self.v_post_b_n_1: # loop for bugs
                self.v_post_b_n_1[key_b_1]                   = deque(maxlen=5)  # deque only keeps last 5 steps of the v bar
                self.v_post_b_n_1_track[key_b_1]             = {}
                self.v_post_b_n_1_track[key_b_1][self.n_exp] = deque(maxlen=10) # deque track only keeps last 10 steps of the v bar
            self.v_post_b_n_1[key_b_1].append(v_temp_b)
            if self.n_exp not in self.v_post_b_n_1_track[key_b_1]:
                self.v_post_b_n_1_track[key_b_1][self.n_exp] = deque(maxlen=10) # deque track only keeps last 10 steps of the v bar
            self.v_post_b_n_1_track[key_b_1][self.n_exp].append(v_temp_b)
            if pd.isnull(self.v_post_b_n_1[key_b_1][-1]):
                raise Exception
        if self.idea == "a":
            for key_d, v_post_n_key_d in self.v_post_d_n.items():
                if self.alpha_update == 'bakf':
                    # alpha would be different per state
                    alpha_d = self.alpha_d[key_d]
                v_temp_d    = ((1-alpha_d) * self.v_post_d_n_1.get(key_d, [self.min_fixing_cost])[-1]) + (alpha_d * v_post_n_key_d)
                if key_d not in self.v_post_d_n_1: # loop for developers
                    self.v_post_d_n_1[key_d]                   = deque(maxlen=5)  # deque only keeps last 5 steps of the v bar
                    self.v_post_d_n_1_track[key_d]             = {}
                    self.v_post_d_n_1_track[key_d][self.n_exp] = deque(maxlen=10) # deque track only keeps last 10 steps of the v bar
                self.v_post_d_n_1[key_d].append(v_temp_d)
                if self.n_exp not in self.v_post_d_n_1_track[key_d]:
                    self.v_post_d_n_1_track[key_d][self.n_exp] = deque(maxlen=10) # deque track only keeps last 10 steps of the v bar
                self.v_post_d_n_1_track[key_d][self.n_exp].append(v_temp_d)
                
    def metrics_update(self, key_, val_, policy_testing = False):
        """Updating the metrics if in the testing mode.

        Args:
            key_ (str): Which metric
            val_ (float/list): What value
            policy_testing (bool, optional): Whether it is for inner test of the policy. Defaults to False.
        """
        if self.mode == 'testing':
            if key_ not in self.metrics:
                self.metrics[key_] = []
            self.metrics[key_].append(val_)
        if policy_testing:
            if key_ not in self.inner_test_metrics:
                self.inner_test_metrics[key_] = []
            self.inner_test_metrics[key_].append(val_)
            
    def create_bugs (self, policy_testing = False):
        """
        Creating bugs according to today's report (random path)
        Args:
            policy_testing (bool, optional): Whether it is for inner test of the policy. Defaults to False.
        """
        if policy_testing:
            # o is for inner loop and t is for outer loop.
            current_time = self.o 
        else:
            current_time = self.t
        for bug_type, bug_n in enumerate(self.path_bug_number[current_time]):
            """ 
            bug_type indicates which bug type should be generated
            bug_n says how many of bug_type is needed.
            """
            for __ in range(bug_n):
                b_id = Bug.bug_id 
                self.current_bugs[b_id] = Bug(bug_type,
                                              self.path_bug_due[current_time][bug_type],
                                              self.LDA_table, n_epoch= self.path_length, t=current_time)
    
    def update_devs_availability(self, policy_testing = False):
        """ Updating the developers' availability according to their random path

        Args:
            policy_testing (bool, optional): Whether it is for inner test of the policy. Defaults to False.
        """
        if policy_testing:
            # o is for inner loop and t is for outer loop.
            current_time = self.o 
        else:
            current_time = self.t
        for dev_id, dev_availability in enumerate(self.path_developer[current_time]):
            if policy_testing:
                """ 
                We check to see whether based on the today's random path, the dev should be available. 
                """
                if (self.developers_profile[dev_id].busy_test == 0) and (dev_availability == 0):
                    self.developers_profile[dev_id].busy_test = 1                        
            else:
                if (self.developers_profile[dev_id].busy      == 0) and (dev_availability == 0):
                    self.developers_profile[dev_id].busy      = 1
                
    def waybackmachine(self, repetition = None):
        """
        The core process:
            We repeat the experiment for multiple times to either learn a policy or
            Apply the obtained policy during the testing phase.
            If the number of repetition is not determined, the default value will be used.
        """
        if repetition != None:
            self.repetition = repetition
        if (self.toy_example) and (self.verbose==2):
            self.repetition = 30 # overwrite the number of repetition
        # repeating the experiment multiple times.
        dev_emails = [dev.email for dev in self.developers_profile.values()]
        if not self.toy_example:
            for dev_id, dev_info in self.dev_probability.developers_info.items():
                if dev_info.email not in dev_emails:
                    id_                          = Developer.dev_id
                    self.developers_profile[id_] = Developer(dev_info.email, dev_info.LDA_experience)
        else: # If it is a toy example, we create only two developers
            # for dev_email, dev_experience in {'a@a.a': [2,1,4,5], 'b@b.b':[5,4,1,2]}.items():
            # for dev_email, dev_experience in {'a@a.a': [2,4,9], 'b@b.b':[9,6,5]}.items():
            #     id_                          = Developer.dev_id
            #     self.developers_profile[id_] = Developer(dev_email, dev_experience) 
            for idx in self.LDA_table.index:
                id_                          = Developer.dev_id
                self.developers_profile[id_] = Developer(self.LDA_table.loc[idx, "Unnamed: 0"], # email
                                                         list(self.LDA_table.loc[idx])[1:])     # experience
        for self.n_exp in tqdm(range(self.repetition), position=0, leave=True):
            self.log_output(f'#####################\n#####################\nEPOCH: {self.n_exp}')
            # create a path of random bugs
            self.path_bug_number, self.path_bug_due = self.bug_probability.random_path_n_gap(self.path_length,
                                                                                             toy_example=self.toy_example)
            # create random availability/path of the developers
            self.path_developer                     = self.dev_probability.random_path(self.path_length,
                                                                                       toy_example=self.toy_example)
            self.log_output(f""" 
path_bug_number:
        * {self.path_bug_number}
path_bug_due:
        * {self.path_bug_due}
path_developer:
        * {self.path_developer}                            
                            """)         
            # adding the schedule of each developer for the next `self.path_length` days to their profile. 
            for idx_, dev_info in enumerate(self.developers_profile.values()):
                dev_info.schedule_seq = np.array([self.path_developer[i][idx_] for i in range(len(self.path_developer))])
                dev_info.schedule     = np.argmax(dev_info.schedule_seq)
                dev_info.busy         = 1-self.path_developer[0][idx_]

            self.current_bugs             = {}
            # exploring the random path
            for self.t in range(self.path_length):
                # creating bugs based on today's random path
                self.create_bugs() 
                # updating devs' availabilities based on today's random path
                self.update_devs_availability()
                self.log_output(f""" %%% \nt = {self.t}
developers_profile:
        * {self.developers_profile}
current_bugs:
        * {self.current_bugs}    """)
                # Now checking the system state at time/day t
                state_t    = State(self.developers_profile, self.current_bugs, self.t, idea=self.idea)
                self.log_output(f"""States:
        State of the bugs = {state_t.S_t_b_bug}
        State of the developers = {state_t.S_t_d_dev}
                                """)
                action_t   = Decision(state_t, self.v_post_b_n_1, self.v_post_d_n_1, t = self.t, method = self.method,
                                      path_length = self.path_length, early_asg_cost = self.early_asg_cost,
                                      gamma = self.gamma, cost_f=self.cost_f, idea = self.idea, min_fixing_cost=self.min_fixing_cost,
                                      normalized = self.normalized, f_notserv = self.f_notserv, type_='integer')
                if (self.toy_example) and (self.verbose == 2):
                    action_t.model.write('test.lp')
                    with open('test.lp') as f:
                        lines = f.read()
                    self.log_output(f"""IP Model:\n{lines}""")                                                        
                if action_t.feasible:
                    obj_val          = action_t.model.ObjVal
                    if self.method == 'adp':
                        action_t_dual    = Decision(state_t, self.v_post_b_n_1, self.v_post_d_n_1, t = self.t, 
                                                    method = self.method, path_length = self.path_length,
                                                    gamma = self.gamma, cost_f=self.cost_f, idea = self.idea, 
                                                    min_fixing_cost=self.min_fixing_cost,
                                                    early_asg_cost = self.early_asg_cost, normalized = self.normalized,
                                                    f_notserv = self.f_notserv, type_='continuous')
                        obj_val_dual     = action_t_dual.model.ObjVal
                        self.dual_gap.append(obj_val-obj_val_dual)
                    """
                    Computing some performance metrics
                    """
                    total_bugs       = 0
                    bugs_to_be_fixed = {}
                    fixing_time      = {}
                    remained_open    = {}
                    due_dates        = []
                    assigned_to_whom = [] # Check which bug is assigned to which developer
                    n_of_assigned    = 0 # Number of assigned / Fixed bugs
                    for vy_index, vy_value in action_t.ydbt.items():
                        value_x = round(vy_value.X)
                        assert value_x >= 0
                        if value_x>0:
                            random.seed(ADP.random_state)
                            ADP.random_state += 1
                            if (random.random() < self.epsilon) or (self.method != 'adp') or (self.mode == 'testing'):
                                # with the likelihood of self.epsilon, postpone the bug instead of assigning it.
                                bugs_to_be_fixed[vy_index] = int(vy_value.X)
                            total_bugs                += vy_value.X
                            n_of_assigned             += vy_value.X
                            for num__ in range(int(vy_value.X)):
                                bug_type_ = vy_index[0][0]
                                bug_due_  = vy_index[0][1]
                                key_d     = vy_index[1]
                                index_fixing = f'fixing(({bug_type_}, {bug_due_}), {key_d})'
                                assert index_fixing in action_t.objective_coef
                                found__ = 0
                                for bug in state_t.bugs_dict.values():
                                    if (bug.bug_type == bug_type_) and ((bug.due_date-bug.days_remained_open) == bug_due_):
                                        # If the bug type and due date matches, then `bug` is the one to be assigned.
                                        found__ += 1
                                        if (found__-1) == num__:
                                            break
                                if found__ == 0: # not found
                                    raise Exception (f'the bug is not found!!')                                
                                if bug.ID not in fixing_time:
                                    fixing_time[bug.ID]   = []
                                if bug.ID not in remained_open:
                                    remained_open[bug.ID] = []                                    
                                fixing_time[bug.ID].append(action_t.objective_coef[index_fixing])
                                remained_open[bug.ID].append(bug.days_remained_open)
                                assigned_to_whom.append(vy_index)
                                due_dates.append(bug_due_)
                        elif (vy_value.X)<0:
                            raise Exception ('Negative values in the variables of the IP model.')
                    n_of_postponed   = 0 # Number of Postponed bugs
                    for vp_value in action_t.pbt.values():
                        if round(vp_value.X)>0:
                            total_bugs                += vp_value.X
                            n_of_postponed            += vp_value.X
                        elif (vp_value.X)<0:
                            raise Exception ('Negative values in the variables of the IP model.')

                    for vh_value in action_t.hdt.values():
                        if round(vh_value.X)<0:
                            raise Exception ('Negative values in the slacks of the IP model.')

                    self.metrics_update('ObjectiveValue', obj_val)
                    self.metrics_update('AssignedToWhom', assigned_to_whom)
                    self.metrics_update('AverageFixingTime', fixing_time)
                    self.metrics_update('DueDates', due_dates)
                    self.metrics_update('NumberOfFixedBugs', n_of_assigned)
                    self.metrics_update('NumberOfOpenBugs', n_of_postponed)
                    self.metrics_update('RemainedOpen', remained_open)
                    self.metrics_update('n', self.n_exp) # current epoch

                    if total_bugs != len(self.current_bugs):
                        print(total_bugs, self.current_bugs)
                        print(action_t.ydbt)
                        print(action_t.pbt)
                        raise Exception
                    """
                    Assigning the bugs
                    """
                    for ((bug_type, due), key_d), num in bugs_to_be_fixed.items():
                        if self.idea == 'a':
                            dev_exp = key_d[0]
                            dev_sch = key_d[1]
                        else:
                            dev_exp = key_d
                            dev_sch = None # no schedule in the simplistic format.
                        # finding the assigned developer
                        found__ = False
                        for dev in state_t.devs_dict.values():
                            if (tuple(dev.experience) == dev_exp) and ((dev.schedule == dev_sch) or (self.idea == 'b')):
                                # If the experience matches, then `dev` is the one to be assigned.
                                # If it is idea "a", the schedule should also matches.
                                found__ = True
                                break
                        if not found__:
                            raise Exception (f'developer is not found!!')
                        for __ in range(num):
                            # For the total number of bugs to be assigned to the `dev`
                            current_bugs_copy = self.current_bugs.copy()
                            for bug_id, bug_value in current_bugs_copy.items():
                                if ((bug_value.bug_type == bug_type) and 
                                    ((bug_value.due_date - bug_value.days_remained_open) == due)):
                                    dev.assign_a_bug(self.current_bugs[bug_id], t = self.t, idea = self.idea)
                                    self.log_output(f""" Assign the bug:
bug state = {bug_value.__repr__()}
developer email, experience:
        * {dev.email} 
        * {dev.experience} """)
                                    self.resolved_bugs[bug_id] = self.current_bugs[bug_id]
                                    del self.current_bugs[bug_id]
                                    break
                    """
                    Updating v* for ADP algorithm
                    """
                    if self.method == 'adp':
                        # Finding dual values
                        dual_values_b          = [] # for bugs (2nd constraint)
                        self.v_post_b_n        = {} # bugs
                        dual_values_d          = [] # for developers (1st constraint)
                        self.v_post_d_n        = {} # developers
                        for c in action_t_dual.model.getConstrs():
                            if c.ConstrName == 'eq:assigned_bugs':
                                dual_values_b.append(c.Pi)
                            elif c.ConstrName == 'eq:assigned_devs':
                                dual_values_d.append(c.Pi)
                        for id_, val_ in enumerate(action_t_dual.available_bugs): # completing bugs
                            self.v_post_b_n[val_] = dual_values_b[id_]
                        for id_, val_ in enumerate(action_t_dual.available_devs): # completing developers
                            self.v_post_d_n[val_] = dual_values_d[id_]
                        self.log_output(f""" Dual values:
    Dual values of bugs       = {dual_values_b}
    Dual values of developers = {dual_values_d}""")

                        if (self.mode == 'training'):
                            # We update the post decision state based on the alpha value and duals.
                            self.updating_post_decision_state_value()
                        self.log_output(f"""
Post Decision for bugs:
        * {self.v_post_b_n_1}
Post Decision for devs:
        * {self.v_post_d_n_1}            """)
                """
                Postponing the open bugs
                """
                current_bugs_copy = self.current_bugs.copy()
                if len(self.current_bugs) > 0:
                    self.log_output(f"Postpone the bug:")
                for bug_id, bug_value in current_bugs_copy.items():
                    # postpone the bugs that still remain open.
                    if self.idea == 'b':
                        # In the simplified version,
                        if bug_value.due_date-bug_value.days_remained_open == 0:
                            # If a bug is not assigned in its due date, it will linger in the system forever.
                            # i.e., it never gets assigned.
                            self.log_output(f""" Remove the unsolved bugs after due date:
bug id = {bug_id}
bug info:
        * {self.current_bugs[bug_id].__repr__()} """)
                            self.resolved_bugs[bug_id] = self.current_bugs[bug_id]                           
                            del self.current_bugs[bug_id]
                    if bug_id in self.current_bugs:
                        self.log_output(f"""
bug id = {bug_id}
bug info before postponement:
        * {self.current_bugs[bug_id].__repr__()} """)
                    bug_value.postponed()
                    if bug_id in self.current_bugs:
                        self.log_output(f""" 
bug info after postponement:
        * {self.current_bugs[bug_id].__repr__()} """)
                """
                Updating developers' schedules
                """
                for dev in self.developers_profile.values():
                    # Reducing their busyness by 1 at the end of the day since 1 day has passed.
                    dev.update_busyness(t=self.t, idea=self.idea)

            cost_of_unassigned_bugs = 0
            for bug_id, bug_value in self.current_bugs.items():
                # postpone the bugs that still remain open.
                cost_of_unassigned_bugs += bug_value.worst_possible_fixing
            self.metrics_update('cost_of_unassigned_bugs', cost_of_unassigned_bugs)
            
            """
            Testing the policy every 100 epochs
            """            
            # Shall we test the performance of our proposed model at this timestamp?
            if (self.n_exp % 100 == 0) and (self.method == 'adp') and (self.mode == 'training'):
                """
                Every 100 iterations, we check to see whether the policy has improved.
                We choose always the same developers' and bugs' path to ensure comparability.
                We do it only in the training phase where we find the optimal policy.
                """
                # create a path of random bugs with specific seed (always the same path)
                self.path_bug_number, self.path_bug_due = self.bug_probability.random_path_n_gap(n    = self.path_length,
                                                                                                 toy_example=self.toy_example,
                                                                                                 seed_ = self.repetition*self.path_length+10)
                # create random availability/path of the developers with specific seed (always the same path)
                self.path_developer                     = self.dev_probability.random_path(n    = self.path_length,
                                                                                           toy_example=self.toy_example,
                                                                                           seed_ = self.repetition*self.path_length+10)
                # adding the schedule of each developer for the next `self.path_length` days to their profile. 
                for idx_, dev_info in enumerate(self.developers_profile.values()):
                    dev_info.reset_for_testing()
                    dev_info.schedule_seq_test = np.array([self.path_developer[i][idx_] for i in range(len(self.path_developer))])
                    dev_info.schedule_test     = np.argmax(dev_info.schedule_seq_test)
                    dev_info.busy_test         = 1-self.path_developer[0][idx_]
                self.current_bugs           = {}
                " exploring the random path "
                discounted_reward           = 0 # discounted reward considering v*
                corrected_discounted_reward = 0 # discounted immediate reward only
                for self.o in range(self.path_length):
                    # creating bugs based on today's random path
                    self.create_bugs(policy_testing=True) 
                    # updating devs' availabilities based on today's random path
                    self.update_devs_availability(policy_testing=True) 
                    # Now checking the system state at time/day t
                    state_t    = State(self.developers_profile, self.current_bugs, self.o, self.idea, policy_testing=True)
                    if self.n_exp == 0:
                        v_post_b_n_1 = {}
                        v_post_d_n_1 = {}
                    else:
                        v_post_b_n_1 = self.v_post_b_n_1
                        v_post_d_n_1 = self.v_post_d_n_1
                    action_t   = Decision(state_t, v_post_b_n_1, v_post_d_n_1, t = self.o, method = self.method,
                                          path_length = self.path_length, early_asg_cost = self.early_asg_cost,
                                          gamma = self.gamma, cost_f=self.cost_f, idea = self.idea, min_fixing_cost=self.min_fixing_cost,
                                          normalized = self.normalized, f_notserv = self.f_notserv, type_='integer')
                    if action_t.feasible:
                        obj_val            = action_t.model.ObjVal
                        corrected_obj_val  = action_t.calculate_pure_obj_func()
                        if self.o == 0:
                            self.metrics_update('FirstObjectiveValue', obj_val, policy_testing=True)
                            self.metrics_update('CorrectedFirstObjectiveValue', corrected_obj_val, policy_testing=True)
                        # discounted reward
                        discounted_reward           +=      obj_val      * (0.9**self.o)
                        corrected_discounted_reward += corrected_obj_val * (0.9**self.o)
                        bugs_to_be_fixed             = {}
                        for vy_index, vy_value in action_t.ydbt.items():
                            if round(vy_value.X)>0:
                                bugs_to_be_fixed[vy_index] = int(vy_value.X)
                        for ((bug_type, due), key_d), num in bugs_to_be_fixed.items():
                            if self.idea == 'a':
                                dev_exp = key_d[0]
                                dev_sch = key_d[1]
                            else:
                                dev_exp = key_d
                                dev_sch = None # no schedule in the simplistic format.
                            # finding the assigned developer
                            found__ = False
                            for dev in state_t.devs_dict.values():
                                if (tuple(dev.experience) == dev_exp) and ((dev.schedule_test == dev_sch) or (self.idea == 'b')):
                                    # If the experience matches, then `dev` is the one to be assigned.
                                    # If it is idea "a", the schedule should also matches.
                                    found__ = True
                                    break
                            if not found__:
                                raise Exception (f'developer is not found!!')
                            for __ in range(num):
                                # For the total number of bugs to be assigned to the `dev`
                                current_bugs_copy = self.current_bugs.copy()
                                for bug_id, bug_value in current_bugs_copy.items():
                                    if ((bug_value.bug_type == bug_type) and 
                                        ((bug_value.due_date - bug_value.days_remained_open) == due)):
                                        dev.assign_a_bug(self.current_bugs[bug_id], t = self.o, 
                                                         idea = self.idea, policy_testing = True)
                                        del self.current_bugs[bug_id]
                                        break
                                    
                    current_bugs_copy = self.current_bugs.copy()
                    for bug_id, bug_value in current_bugs_copy.items():
                        # postpone the bugs that still remain open.
                        if self.idea == 'b':
                            # In the simplified version,
                            if bug_value.due_date-bug_value.days_remained_open == 0:
                                # If a bug is not assigned in its due date, it will linger in the system forever.
                                # i.e., it never gets assigned.
                                del self.current_bugs[bug_id]
                        bug_value.postponed()

                    for dev in self.developers_profile.values():
                        # Reducing their busyness by 1 at the end of the day since 1 day has passed.
                        dev.update_busyness(t=self.o, idea=self.idea, policy_testing = True)

                self.metrics_update('discounted_reward', discounted_reward, policy_testing=True)
                self.metrics_update('corrected_discounted_reward', corrected_discounted_reward, policy_testing=True)
                self.metrics_update('n', self.n_exp, policy_testing=True)
                self.metrics_update('n_unassigned_bugs', len(self.current_bugs), policy_testing=True)
                cost_of_unassigned_bugs = 0
                for bug_id, bug_value in self.current_bugs.items():
                    # postpone the bugs that still remain open.
                    cost_of_unassigned_bugs += bug_value.worst_possible_fixing
                self.metrics_update('cost_of_unassigned_bugs', cost_of_unassigned_bugs, policy_testing=True)
                # including the cost of unassigned bugs
                corrected_discounted_reward_plus = corrected_discounted_reward + cost_of_unassigned_bugs * (0.9**self.o) 
                self.metrics_update('corrected_discounted_reward+', corrected_discounted_reward_plus, policy_testing=True)
                if self.verbose == 2:
                    action_t.model.write('test.lp')
                if self.idea == 'a':
                    # we have the dual of the first constraint only in version `a` of the implementation.
                    for dev_key in range(3):
                        which_dev = list(self.v_post_d_n_1.keys())[dev_key]
                        if self.n_exp == 0:
                            self.metrics_update(f'{which_dev}', self.v_post_d_n_1[which_dev][0], policy_testing=True)
                        else:
                            self.metrics_update(f'{which_dev}', self.v_post_d_n_1[which_dev][-1], policy_testing=True)
                for bug_key in [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (1,2), (1,-1), (1,-2), (1,-3)]:
                    self.metrics_update(f'{bug_key}', self.v_post_b_n_1.get(bug_key, [self.min_fixing_cost])[-1], policy_testing=True)
        if (self.toy_example) and (__name__ == "__main__"):
            if verbose == 2:
                try:
                    os.mkdir(f"dat/{self.project}")
                except: pass            
                f_name = def_file_name(self.alpha_update, self.early_asg_cost, self.method, self.gamma, self.epsilon, self.idea)
                with open(f"dat/{self.project}/ToyExample_{f_name}.txt", 'w') as f:
                    f.write(self.log)

"""
Time to Run the third phase
"""
#######################################
##                                   ##
##      author: Hadi Jahanshahi      ##
##     hadi.jahanshahi@ryerson.ca    ##
##          Data Science Lab         ##
##                                   ##
#######################################

if __name__ == "__main__":    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        elif v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser = argparse.ArgumentParser(description='The parser will handle hyperparameters of the last Run')

    parser.add_argument(
        '--project',
        default = 'GCC',
        type    = str,
        help    = 'it can be selected from this list: [LibreOffice, Mozilla, EclipseJDT, GCC]'
    )

    parser.add_argument(
        '--verbose',
        default = 0,
        type    = int,
        help    = 'it can be either: [0, 1, 2, nothing, some, all]'
    )
    
    parser.add_argument(
        '--epochs_training',
        default = 20_000,
        type    = int,
        help    = 'The number of epochs for the training phase'
    )    

    parser.add_argument(
        '--epochs_testing',
        default = 100,
        type    = int,
        help    = 'The number of epochs for the testing phase'
    )
    
    parser.add_argument(
        '--project_horizon',
        default = 30,
        type    = int,
        help    = 'The number of days for each epoch.'
    )    
    
    parser.add_argument(
        '--method',
        default = 'ADP',
        type    = str,
        help    = 'Indicating which approach we want to use.'
    )    

    parser.add_argument(
        '--alpha_update',
        default = 'constant',
        type    = str,
        help    = 'Indicating how to update alpha in adp algorithm for this set [constant, harmonic, bakf].'
    ) 
        
    parser.add_argument(
        '--alpha',
        default = 0.5,
        type    = float,
        help    = 'alpha value for constant case.'
    )

    parser.add_argument(
        '--early_asg_cost',
        default = False,
        type    = str2bool,
        help    = 'Whether to impose a cost for early assignments.'
    ) 

    parser.add_argument(
        '--gamma',
        default = 0.9,
        type    = float,
        help    = 'the discounting coefficient of the Bellman equation.'
    ) 

    parser.add_argument(
        '--epsilon',
        default = 0.25,
        type    = float,
        help    = 'the ratio of exploration (e-greedy approach). 1 means no exploration.'
    ) 
    
    parser.add_argument(
        '--cost_f',
        default = 'Linear',
        type    = str,
        help    = 'How the cost function of the late assignment should be.'
    )     
    
    parser.add_argument(
        '--idea',
        default = 'a',
        type    = str,
        help    = 'Which idea to use? (a) Sophisticated (b) Simplified.'
    )
    
    parser.add_argument(
        '--toy_example',
        default = True,
        type    = str2bool,
        help    = 'Whether it is a toy example.'
    )    
    
    wayback_param        = parser.parse_args()
    project              = wayback_param.project
    verbose              = wayback_param.verbose
    epochs_training      = wayback_param.epochs_training
    epochs_testing       = wayback_param.epochs_testing
    project_horizon      = wayback_param.project_horizon
    method               = wayback_param.method.lower()
    alpha                = wayback_param.alpha
    alpha_update         = wayback_param.alpha_update.lower()
    early_asg_cost       = wayback_param.early_asg_cost
    gamma                = wayback_param.gamma
    epsilon              = wayback_param.epsilon
    cost_f               = wayback_param.cost_f.lower()
    idea                 = wayback_param.idea.lower()
    toy_example          = wayback_param.toy_example
    
    def def_file_name(alpha_update, early_asg_cost, method, gamma, epsilon, idea):
        if alpha_update == 'constant':
            alpha_name = ''
        elif alpha_update == 'harmonic':
            alpha_name = '_Harmonic'
        elif alpha_update == 'bakf':
            alpha_name = '_BAKF'
        else:
            raise Exception (f'Wrong alpha_update {alpha_update}')
        if early_asg_cost:
            early_asg_cost_name = ''
        else:
            early_asg_cost_name = '_no_early_asg_cost'
        if method == 'adp':
            gam_eps_name = f'_gamma{gamma}_epsilon{epsilon}'
        else:
            gam_eps_name = ''
        if   idea.lower() in ['a', 'original', 'sophisticated']:
            idea_name     = ''          # Developers have schedule and bugs remain the system until they get fixed.
        elif idea.lower() in ['b', 'simplified', 'simple']:
            idea_name    = '_simplified' # Developers leave the system and bugs leave the system if remain unfixed.
        else:
            raise Exception (f"undefined idea ({idea}). Choose from ['a', 'original', 'sophisticated', 'b', 'simplified']")
        file_name = f'{method}{alpha_name}_{cost_f}{early_asg_cost_name}{gam_eps_name}{idea_name}'
        return file_name
    
    f_name = def_file_name(alpha_update, early_asg_cost, method, gamma, epsilon, idea)
    
    if not toy_example:
        LDA_table            = pd.read_csv(f"dat/{project}/time_to_fix_LDA.csv")
    else: #toy_example
        LDA_table            = pd.read_csv(f"dat/ToyExample/time_to_fix_LDA.csv")
    
    # training
    phase_ = 'training'
    with open (f"dat/{project}/phaseII_{phase_}.pickle", "rb") as file:
        second_phase = pickle.load(file)
    dev_probability      = Dev_prob(second_phase.developers_info, second_phase.status     )
    bug_probability      = Bug_prob(second_phase.status,          second_phase.bug_info_db)    
    adp                  = ADP(dev_probability, bug_probability, project, LDA_table, path_length = project_horizon,
                               early_asg_cost = early_asg_cost, mode = phase_, alpha_update=alpha_update, idea = idea,
                               toy_example = toy_example,
                               gamma = gamma, epsilon = epsilon, alpha=alpha, method = method, verbose = verbose)
    adp.waybackmachine(epochs_training)
    """ Keep a backup of the model at the end of the training """
    with open (f'dat/{adp.project}/phaseIII_{phase_}_{f_name}.pickle', 'wb') as file:
        pickle.dump(adp, file) # use `pickle.load` to do the reverse

    if not adp.toy_example: # If it is not a toy example
        # testing
        Developer.dev_id  = 0 # resetting the developers' ID for the testing phase.
        Bug.bug_id        = 0 # resetting the bugs' ID for the testing phase.
        phase_ = 'testing'
        with open (f"dat/{adp.project}/phaseII_{phase_}.pickle", "rb") as file:
            second_phase_testing = pickle.load(file)
        dev_probability_test = Dev_prob(second_phase_testing.developers_info, second_phase_testing.status, when=phase_)
        bug_probability_test = Bug_prob(second_phase_testing.status         , second_phase_testing.bug_info_db, when=phase_)
        adp_test             = ADP(dev_probability_test, bug_probability_test, project, LDA_table, path_length = project_horizon,
                                early_asg_cost = early_asg_cost, mode=phase_, v_star_coef_b = adp.v_post_b_n_1, 
                                v_star_coef_d = adp.v_post_d_n_1, method = method, alpha_update=alpha_update, alpha=alpha,
                                toy_example = toy_example, idea = idea, gamma = gamma, cost_f = cost_f, verbose = verbose)
        adp_test.waybackmachine(epochs_testing)
        """ Keep a backup of the model at the end of the testing """
        with open(f'dat/{adp_test.project}/phaseIII_{phase_}_{f_name}.pickle', 'wb') as file:
            pickle.dump(adp_test, file) # use `pickle.load` to do the reverse


"""Commands Used for Running"""
# conda activate python3.9
# ADPConstantEarlyAssignNoGamma0.9eps0.25Exponential
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Constant --early_asg_cost=False --gamma=0.9 --epsilon=0.25 --cost_f=exponential
# ADPConstantEarlyAssignNoGamma0.99eps0.25Exponential
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Constant --early_asg_cost=False --gamma=0.99 --epsilon=0.25 --cost_f=exponential
# ADPConstantEarlyAssignNoGamma0.9eps1Exponential
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Constant --early_asg_cost=False --gamma=0.9 --epsilon=1 --cost_f=exponential
# ADPConstantEarlyAssignNoGamma0.99eps1Exponential
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Constant --early_asg_cost=False --gamma=0.99 --epsilon=1 --cost_f=exponential
# ADPConstantEarlyAssignNoGamma0.9eps0.25Linear
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Constant --early_asg_cost=False --gamma=0.9 --epsilon=0.25 --cost_f=Linear
# ADPConstantEarlyAssignNoGamma0.99eps0.25Linear
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Constant --early_asg_cost=False --gamma=0.99 --epsilon=0.25 --cost_f=Linear
# ADPConstantEarlyAssignNoGamma0.9eps1Linear
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Constant --early_asg_cost=False --gamma=0.9 --epsilon=1 --cost_f=Linear
# ADPConstantEarlyAssignNoGamma0.99eps1Linear
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Constant --early_asg_cost=False --gamma=0.99 --epsilon=1 --cost_f=Linear

# conda activate python3.9
# ADPHarmonicEarlyAssignNoGamma0.9eps0.25Exponential
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Harmonic --early_asg_cost=False --gamma=0.9 --epsilon=0.25 --cost_f=exponential
# ADPHarmonicEarlyAssignNoGamma0.99eps0.25Exponential
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Harmonic --early_asg_cost=False --gamma=0.99 --epsilon=0.25 --cost_f=exponential
# ADPHarmonicEarlyAssignNoGamma0.9eps1Exponential
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Harmonic --early_asg_cost=False --gamma=0.9 --epsilon=1 --cost_f=exponential
# ADPHarmonicEarlyAssignNoGamma0.99eps1Exponential
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Harmonic --early_asg_cost=False --gamma=0.99 --epsilon=1 --cost_f=exponential
# ADPHarmonicEarlyAssignNoGamma0.9eps0.25Linear
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Harmonic --early_asg_cost=False --gamma=0.9 --epsilon=0.25 --cost_f=Linear
# ADPHarmonicEarlyAssignNoGamma0.99eps0.25Linear
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Harmonic --early_asg_cost=False --gamma=0.99 --epsilon=0.25 --cost_f=Linear
# ADPHarmonicEarlyAssignNoGamma0.9eps1Linear
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Harmonic --early_asg_cost=False --gamma=0.9 --epsilon=1 --cost_f=Linear
# ADPHarmonicEarlyAssignNoGamma0.99eps1Linear
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Harmonic --early_asg_cost=False --gamma=0.99 --epsilon=1 --cost_f=Linear

# conda activate python3.9
# ADPBAKFEarlyAssignNoGamma0.9eps0.25Exponential
# python3.9 simulator/third_run.py --project=LibreOffice --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=BAKF --early_asg_cost=False --gamma=0.9 --epsilon=0.25 --cost_f=exponential --toy_example=True
# ADPBAKFEarlyAssignNoGamma0.99eps0.25Exponential
# python3.9 simulator/third_run.py --project=LibreOffice --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=BAKF --early_asg_cost=False --gamma=0.99 --epsilon=0.25 --cost_f=exponential --toy_example=True
# ADPBAKFEarlyAssignNoGamma0.9eps1Exponential
# python3.9 simulator/third_run.py --project=LibreOffice --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=BAKF --early_asg_cost=False --gamma=0.9 --epsilon=1 --cost_f=exponential --toy_example=True
# ADPBAKFEarlyAssignNoGamma0.99eps1Exponential
# python3.9 simulator/third_run.py --project=LibreOffice --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=BAKF --early_asg_cost=False --gamma=0.99 --epsilon=1 --cost_f=exponential --toy_example=True
# ADPBAKFEarlyAssignNoGamma0.9eps0.25Linear
# python3.9 simulator/third_run.py --project=LibreOffice --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=BAKF --early_asg_cost=False --gamma=0.9 --epsilon=0.25 --cost_f=Linear --toy_example=True
# ADPBAKFEarlyAssignNoGamma0.99eps0.25Linear
# python3.9 simulator/third_run.py --project=LibreOffice --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=BAKF --early_asg_cost=False --gamma=0.99 --epsilon=0.25 --cost_f=Linear --toy_example=True
# ADPBAKFEarlyAssignNoGamma0.9eps1Linear
# python3.9 simulator/third_run.py --project=LibreOffice --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=BAKF --early_asg_cost=False --gamma=0.9 --epsilon=1 --cost_f=Linear --toy_example=True
# ADPBAKFEarlyAssignNoGamma0.99eps1Linear
# python3.9 simulator/third_run.py --project=LibreOffice --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=BAKF --early_asg_cost=False --gamma=0.99 --epsilon=1 --cost_f=Linear --toy_example=True

# conda activate python3.9
# MyopicEarlyAssignNoLinear
# python3.9 simulator/third_run.py --project=LibreOffice --epochs_training=20000 --epochs_testing=100 --method=myopic --early_asg_cost=False --cost_f=Linear --alpha_update=constant --toy_example=True
# MyopicEarlyAssignNoExponential
# python3.9 simulator/third_run.py --project=LibreOffice --epochs_training=20000 --epochs_testing=100 --method=myopic --early_asg_cost=False --cost_f=exponential --alpha_update=constant  --toy_example=True


# conda activate python3.9
# ADPBAKFEarlyAssignNoGamma0.99eps1LinearSimplified
# python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=10000 --epochs_testing=100 --method=ADP --alpha_update=BAKF --early_asg_cost=False --gamma=0.99 --epsilon=1 --cost_f=Linear --idea=b