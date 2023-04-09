from utils.prerequisites import *

class Dev_prob:
    random_state = 0
    def __init__(self, developers_info, sys_status, when='training'):
        self.developers_info          = developers_info
        for dev_value in self.developers_info.values():
            dev_value.LDA_experience = [int(i) for i in dev_value.LDA_experience]        
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
        self.prob                     = []        
        for dev in self.developers_info.values():
            dev.bugs                = []
            dev.components          = []
            dev.components_tracking = []
            dev.n_assigned_bugs     = 0
            dev.status              = []   # new, the actual availability of the devs
            dev.prob                = None # new, the prob of dev being available
        self.reading_dev_availabilities()
            
    def reading_dev_availabilities(self):
        for each_day in self.sys_status["developers"]:
            for i, (id_, dev_info) in enumerate(self.developers_info.items()):
                dev_info.status.append(each_day[i])
        for dev_info in self.developers_info.values():
            dev_info.prob = sum(dev_info.status) / len(dev_info.status)
            self.prob.append(dev_info.prob)
        # avoiding zero values for probabilities using Laplace smoothing with alpha = 0.01
        for idx, prob in enumerate(self.prob):
            self.prob[idx] = (prob+0.01) / (sum(np.array(self.prob))+(0.01*len(self.prob)))
            
    def random_path(self, n, toy_example=False, seed_=None):
        """
        n is the length of path
        toy_example tells us whether it is a real problem or not.
        seed is set for reproducibility.
        It returns a path for developers' availabilities.
        """
        path = []
        for i in range(n):
            tmp = []
            if toy_example:
                self.prob = [.2, .25, .3, .35, .45, .5, .55, .6, .65, .7] # For toy example, we have 10 developers.
            for prob in self.prob:
                if seed_ == None:
                    random.seed(Dev_prob.random_state)
                    Dev_prob.random_state += 1
                else:
                    random.seed(seed_)
                    seed_ += 1
                # increasing the chance of being available and adding some uncertainties
                if not toy_example:
                    random_increase = random.uniform(-0.05,0.45)
                else:
                    random_increase = 0
                if random.random() < (prob + random_increase): 
                    tmp.append(1)
                else:
                    tmp.append(0)
            path.append(tmp)
        return path