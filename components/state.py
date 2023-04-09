class State:
    def __init__(self, dev_dict:dict, bugs_dict:dict, time:int, idea:str, policy_testing = False):
        self.devs_dict        = dev_dict
        self.bugs_dict        = bugs_dict
        self.time             = time
        self.S_t_d_dev        = {}
        self.S_t_b_bug        = {}
        self.state_update(idea, policy_testing)

    def state_update(self, idea, policy_testing = False):
        """Updating the state of the system for bugs and developers

        Args:
            idea (str): Is it idea `a` (sophisticated) or `b` (simplistic)?
            policy_testing (bool, optional): Whether it is for inner test of the policy. Defaults to False.
        """
        self.S_t_d_dev    = {}
        for dev in self.devs_dict.values():
            if policy_testing:
                schedule_tmp = dev.schedule_test
            else:
                schedule_tmp = dev.schedule
            if idea == 'a':
                # schedules matter
                key_ = (tuple(dev.experience), schedule_tmp)
            else:
                # simplistic version.
                key_ = tuple(dev.experience)
            if key_ not in self.S_t_d_dev:
                self.S_t_d_dev[key_] = 0
            if (dev.busy == 0) and (not policy_testing):
                self.S_t_d_dev[key_] += 1
            elif (dev.busy_test == 0) and (policy_testing):
                self.S_t_d_dev[key_] += 1
            else:
                assert ((dev.busy > 0) or (dev.busy_test > 0)) 
        self.S_t_b_bug    = {}
        for bug in self.bugs_dict.values():
            bug_tuple = (bug.bug_type, bug.due_date-bug.days_remained_open)
            if bug_tuple not in self.S_t_b_bug:
                self.S_t_b_bug[bug_tuple] = 0
            self.S_t_b_bug[bug_tuple] += 1

    def __repr__(self):
        n_bugs = len(self.S_t_b_bug)
        n_dev  = len(self.S_t_d_dev)
        return f'We have {n_bugs} bugs and {n_dev} developers at time {self.time}.'