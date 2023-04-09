import numpy    as np 
from   gurobipy import GRB, Model

class Bug:
    bug_id = 0
    def __init__(self, bug_type:int, due_date:int):
        self.ID                 = Bug.bug_id
        Bug.bug_id             += 1
        self.bug_type           = bug_type # LDA Category
        self.due_date           = due_date
        self.days_remained_open = 0
        
    def postponed(self):
        self.days_remained_open += 1
        
    def __repr__(self):
        return str(self.ID)
    
class Developer:
    dev_id = 0
    def __init__(self, simultaneous_job:int, filled_capacity:list, experience:list, Total_limit:int):
        self.ID                  = Developer.dev_id
        Developer.dev_id        += 1
        self.simultaneous_job    = simultaneous_job
        self.filled_capacity     = filled_capacity
        self.Total_limit         = Total_limit # Time horizon
        self.assigned_bugs       = []
        self.time_limit          = [Total_limit-used for used in filled_capacity]
        self.experience          = experience # LDA_experience
        self.schedule            = self.update_schedule()    #T_{jt}^d
        
    def assign_a_bug(self, bug: Bug, which_slot: int):
        self.filled_capacity[which_slot] += Bug.fixing_time
        self.time_limit[which_slot]      -= Bug.fixing_time
        assert (self.filled_capacity[which_slot] <= Total_limit)
        self.assigned_bugs.append(bug.ID)
        
    def __repr__(self):
        return str(self.ID)
    
    def update_schedule(self):
        schedule = []
        for filled_capacity in (self.filled_capacity):
            schedule.append([0 if i<=filled_capacity else 1 for i in range(self.Total_limit)])
        return schedule
    
    
class State:
    def __init__(self, dev_dict:dict, bugs_dict:dict, time):
        self.devs_dict        = dev_dict
        self.bugs_dict        = bugs_dict
        self.time             = time
        self.S_t_d_dev        = {}
        self.S_t_b_bug        = {}
        self.state_update()
        
    def state_update(self):
        self.S_t_d_dev    = {}
        for dev in self.devs_dict.values():
            if tuple(dev.experience) not in self.S_t_d_dev:
                self.S_t_d_dev[tuple(dev.experience)] = 0
            self.S_t_d_dev[tuple(dev.experience)] += 1
        self.S_t_b_bug    = {}
        for bug in self.bugs_dict.values():
            if (bug.bug_type) not in self.S_t_b_bug:
                self.S_t_b_bug[(bug.bug_type)] = 0
            self.S_t_b_bug[(bug.bug_type)] += 1

class Decision:
    def __init__(self, system_state):
        self.pbt            = {} # num of bugs with attr b postponed to time t+1
        self.ydbt           = {} # num of bugs with attr b solved by dev with attr d
        self.model          = Model()
        self.feasible_actions(system_state)
        
    def feasible_actions(self, system_state, LogToConsole= False, TimeLimit=60, write=False):
        n_bugs                         = len(system_state.S_t_b_bug)
        n_devs                         = len(system_state.S_t_d_dev)        
        self.model.params.LogToConsole = LogToConsole
        self.model.params.TimeLimit    = TimeLimit
        for bug_type in range(n_bugs):
            self.pbt[bug_type] = (self.model.addVar(vtype= GRB.INTEGER, name=f'p[{bug_type}]'))
        for bug_type in range(n_bugs):
            for dev_type in range(n_devs):
                self.ydbt[bug_type,dev_type] = (self.model.addVar(vtype= GRB.INTEGER, name=f'y[{bug_type},{dev_type}]'))
                        
        obj_func = 0
        not_served = 0
        for bug_type in range(n_bugs):
            if system_state.bugs_dict:
                not_served += 1
        for bug_id in range(n_bugs):
            for dev_id in range(n_devs):
                n_slots = dev_dict[dev_id].simultaneous_job
                for slot_n in range(n_slots):
                    for time in range(Total_limit):
                        obj_func += P[bug_id,dev_id]* x[bug_id,dev_id,slot_n,time]
                        
        self.model.setObjective(0, GRB.MINIMIZE)
        
        for dev_type, dev_d in enumerate(system_state.S_t_d_dev):
            n_assigned_devs_constraint = 0
            for bug_type in range(n_bugs):
                n_assigned_devs_constraint += self.ydbt[bug_type,dev_type]
            self.model.addConstr(n_assigned_devs_constraint <= system_state.S_t_d_dev[dev_d],
                                 name = "eq:assigned_devs")

        for bug_type, bug_b in enumerate(system_state.S_t_b_bug):
            n_assigned_bugs_constraint = 0
            for dev_type in range(n_devs):
                n_assigned_bugs_constraint += self.ydbt[bug_type,dev_type]
            n_assigned_bugs_constraint += self.pbt[bug_type]
            self.model.addConstr(n_assigned_bugs_constraint == system_state.S_t_b_bug[bug_b],
                                 name = "eq:assigned_bugs")
        if write:
            self.model.write('ADP.lp')
        self.model.optimize()