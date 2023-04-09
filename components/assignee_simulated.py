from utils.prerequisites import *
from components.bug_simulated import Bug
class Developer:
    dev_id = 0
    def __init__(self, email:str, experience:list):
        self.ID                  = Developer.dev_id
        Developer.dev_id        += 1
        self.email               = email
        self.assigned_bugs       = []
        self.experience          = [int(i) for i in experience] # LDA_experience
        """
            When we assign a bug, we increase the `busy` value by the number of fixing days
        to make the person unavailable for few days.
        0 means the developer is `available` currently.
        """
        self.busy                = 0
        self.busy_test           = 0 # Same but for inner policy testing
        """
        A sequence of the schedule of the developers. 
        It shows when the developer is available for the whole project horizon. 
        """
        self.schedule_seq        = np.array([]) # [0,0,1,1,1,1,0,0,1] # length = 30, 30 days
        self.schedule_seq_test   = np.array([]) # Same but for inner policy testing
        """ 
        When the developer would be available. 0 means the developer is available now,
        whereas value i means, they will be available i epoch(s) later.
        """
        self.schedule               = 0 #np.argmax(self.schedule_seq)
        self.schedule_test          = 0 # Same but for inner policy testing
        self.previous_schedule      = 0 #np.argmax(self.schedule_seq)
        self.previous_schedule_test = 0 # Same but for inner policy testing
        
    def assign_a_bug(self, bug: Bug, t:int, idea:str, policy_testing = False):
        """Assigning a bug to the developer and keeping track of the assigned bugs.
        We also update the ``busyness`` of the developer based on the fixing time of the bug.

        Args:
            bug (Bug): The bug to be assigned.
            t (int)  : The current timestamp
            idea (str): Is it idea `a` (sophisticated) or `b` (simplistic)?
            policy_testing (bool, optional): Whether it is for inner test of the policy. Defaults to False.
        """
        if not policy_testing:
            self.assigned_bugs.append(bug.ID)
            if idea == 'a':
                # If it is the sophisticated, realistic idea, we update the schedule.
                self.busy                       += self.experience[bug.bug_type]
                self.schedule_seq[t:t+self.busy] = 0 # Make upcoming `self.busy` epoch(s) of their schedule busy.
            bug.assignee       = self.email
            bug.fixing_time    = self.experience[bug.bug_type]
            bug.whether_top1   = bug.fixing_time in bug.top1
            bug.whether_top3   = bug.fixing_time in bug.top3
            bug.whether_top5   = bug.fixing_time in bug.top5
        else: #inner test
            # if we are testing the policy:
            if idea == 'a':
                # If it is the sophisticated, realistic idea, we update the schedule.
                self.busy_test                             += self.experience[bug.bug_type]
                self.schedule_seq_test [t:t+self.busy_test] = 0 # Make upcoming `self.busy_test` epoch(s) of their schedule busy.
            
        
    def update_busyness(self, t:int, idea:str, policy_testing = False):
        """
        Everyday, we decrease the busyness of a developer as one day of fixing has passed.       
        This value cannot be negative.
        
        Args:
            t (int): The current time stamp
            idea (str): Is it idea `a` (sophisticated) or `b` (simplistic)?
            policy_testing (bool, optional): Whether it is for inner test of the policy. Defaults to False.
        """
        if not policy_testing:
            if idea == 'a':
                self.busy = max(0, self.busy-1)
            else:
                self.busy = 0
            if (t+1) != len(self.schedule_seq):
                # if we are not at the last episode
                # and if it is the sophisticated, realistic idea, we update the schedule.
                if sum(self.schedule_seq[t+1:])>0: # At least one of the upcoming day is available
                    # Determine the first available day (when its schedule_seq is 1)
                    self.previous_schedule = self.schedule
                    self.schedule          = np.argmax(self.schedule_seq[t+1:])
        else: #inner test
            # if we are testing the policy:
            if idea == 'a':
                self.busy_test = max(0, self.busy_test-1)
            else:
                self.busy_test = 0
            if (t+1) != len(self.schedule_seq_test):
                # if we are not at the last episode
                if sum(self.schedule_seq_test[t+1:])>0: # At least one of the upcoming day is available
                    # Determine the first available day (when its schedule_test is 1)
                    self.previous_schedule_test = self.schedule_test
                    self.schedule_test          = np.argmax(self.schedule_seq_test[t+1:])
                    if self.schedule_test < self.busy_test:
                        print(self.schedule_test)
                        print(self.schedule_seq_test)
                        print(t)
                        print(self.busy_test)
                        
    def reset_for_testing(self):
        """ Resetting developers' information for testing the current policy
        """
        self.busy_test           = 0 # Same but for inner policy testing
        self.schedule_seq_test   = np.array([]) # Same but for inner policy testing
        self.schedule_test       = 0 # Same but for inner policy testing
        
    def __repr__(self):
        return f'Dev # {self.ID} with email {self.email}, busyness of {self.busy}'