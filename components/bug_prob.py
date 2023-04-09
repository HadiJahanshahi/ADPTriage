from utils.prerequisites import *

class Bug_prob:
    random_state = 0    
    def __init__(self, sys_status, bug_info_db, when='training'):
        self.sys_status               = sys_status
        # considering only feasible dates
        if when == 'training':
            filter_date_index             = ((np.array(self.sys_status["date"]) >= "2010-01-01") & 
                                             (np.array(self.sys_status["date"]) < "2018-01-01"))
        elif when == 'testing':
            filter_date_index             = ((np.array(self.sys_status["date"]) >= "2018-01-01") & 
                                             (np.array(self.sys_status["date"]) < "2020-01-01"))
        else:
            raise Exception(f'unknown when attribute {when}.')
        self.sys_status["date"      ] = list(np.array(self.sys_status["date"      ])[filter_date_index])
        self.sys_status["developers"] = list(np.array(self.sys_status["developers"])[filter_date_index])
        self.sys_status["bugs"      ] = list(np.array(self.sys_status["bugs"      ])[filter_date_index])
        
        self.bug_types_status         = {i:np.array(self.sys_status["bugs"])[:,i] for i in 
                                         range(len(self.sys_status["bugs"][0]))}
        # finding probabilities and applying add-one smoothing
        self.prob                     = {i:((pd.value_counts(val)+1)/(sum(pd.value_counts(val))+len(self.bug_types_status)))
                                         for i,val in self.bug_types_status.items()}
        Q1, Q3                        = bug_info_db.assignment_gap[bug_info_db.assignment_gap!=np.inf
                                                                  ].quantile([0.25,0.75])
        IQR                           = Q3 - Q1
        Max_gap                       = Q3 + (1.5*IQR)
        bug_info_db.LDA               = bug_info_db.LDA.map(int)
        self.new_bug_info_db          = bug_info_db[bug_info_db.assignment_gap < Max_gap][
            ['LDA', 'assignment_gap']].copy()
        
    def random_path_n_gap(self, n, toy_example = False, seed_ = None):
        """
        n is the length of path
        toy_example tells us whether it is a real problem or not.
        seed is set for reproducibility.
        It returns a path for bug and their possible gaps (deadlines).
        """
        path   = []
        path_g = []
        for n_ in range(n):
            tmp   = []
            tmp_g = []
            if not toy_example:
                for i, prob_dist in self.prob.items():
                    if seed_ == None:
                        random.seed(Bug_prob.random_state)
                        Bug_prob.random_state += 1
                    else:
                        random.seed(seed_)
                        seed_ += 1
                    tmp.append(random.choices(prob_dist.index, weights = prob_dist.values)[0])
                    
                    db_i                   = self.new_bug_info_db[self.new_bug_info_db.LDA == i].copy()
                    if seed_ == None:
                        random.seed(Bug_prob.random_state)
                        Bug_prob.random_state += 1
                    else:
                        random.seed(seed_)
                        seed_ += 1
                    tmp_g.append(max(random.choices(list(db_i['assignment_gap']))[0], 1))
            else: # if it is a toy example
                # for i, prob_dist in {0:{0:.6,1:.3,2:.1}, 1:{0:.9,1:.09,2:.01}, 2:{0:.3,1:.7}, 4:{0:.8,1:.1,2:.1}}.items():
                # for i, prob_dist in {0:{0:.6,1:.3,2:.1}, 1:{0:.9,1:.09,2:.01}, 2:{0:.4,1:.6}}.items():
                #     if seed_ == None:
                #         random.seed(Bug_prob.random_state)
                #         Bug_prob.random_state += 1
                #     else:
                #         random.seed(seed_)
                #         seed_ += 1
                #     tmp.append(random.choices(list(prob_dist.keys()), weights = list(prob_dist.values()))[0])
                for i in range(5):
                    if i == 0: # 5 times a week
                        if (n_%7 != 0) and (n_%6 != 0):
                            tmp.append(1)
                    if i == 1:  # every other day
                        if n_%2 != 0:
                            tmp.append(1)
                    if i == 2: # every 3 days
                        if n_%3 == 0:
                            tmp.append(1)
                    if i == 3: # every 5 days
                        if n_%5 == 0:
                            tmp.append(1)
                    if i == 4: # every 10 days
                        if n_%10 == 0:
                            tmp.append(1)                        
                    # assignment_gap_dict = {0:[4,4,5,5], 1:[3,3,7,8,11], 2:[3,8,9,9,10,18], 4:[1,1,2,3,4]}
                    assignment_gap_dict = {0:[2], 1:[3], 2:[4], 3:[5], 4:[10]}
                    if seed_ == None:
                        random.seed(Bug_prob.random_state)
                        Bug_prob.random_state += 1
                    else:
                        random.seed(seed_)
                        seed_ += 1
                    tmp_g.append(max(random.choices(assignment_gap_dict[i])[0], 1))
            path.append  (tmp  )
            path_g.append(tmp_g)
        return path, path_g