from utils.prerequisites import *

class Bug:
    bug_id = 0
    def __init__(self, bug_type:int, due_date:int, LDA_tab:pd.core.frame.DataFrame, n_epoch:int, t:int):
        self.ID                    = Bug.bug_id
        Bug.bug_id                += 1
        self.bug_type              = bug_type # LDA Category
        # The due date for assignment, cannot exceed the project horizon.
        self.due_date              = min(due_date, n_epoch-t+1) # \bug_\Due = \min ({\nepochs-\epoch-1, \nduedate)
        # The fastest time to fix a bug if the developer is available.
        self.best_possible_fixing  = int(min(LDA_tab[str(bug_type)])) 
        self.worst_possible_fixing = int(max(LDA_tab[str(bug_type)]))
        self.top1                  = list(LDA_tab.nsmallest(1, str(bug_type))[str(bug_type)]) # top-3 shortest fixing costs 
        self.top3                  = list(LDA_tab.nsmallest(3, str(bug_type))[str(bug_type)]) # top-3 shortest fixing costs 
        self.top5                  = list(LDA_tab.nsmallest(5, str(bug_type))[str(bug_type)]) # top-3 shortest fixing costs 
        self.days_remained_open    = 0
        self.assignee              = None
        self.fixing_time           = None # It will be registered after being assigned.
        self.whether_top1          = None # After assignment, we check to see if the dev is the best one.
        self.whether_top3          = None # After assignment, we check to see if the dev is among top 3 devs.
        self.whether_top5          = None # After assignment, we check to see if the dev is among top 5 devs.
        
    def postponed(self):
        self.days_remained_open   += 1
        
    def __repr__(self):
        return f'Bug {self.ID} of Type {self.bug_type} with {self.due_date-self.days_remained_open} more days to be solved.'