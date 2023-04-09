from utils.prerequisites import *
from utils.functions import Functions
class Bug:
    n_bugs = 0
    n_arcs = 0
    def __init__(self, idx, creation_time, severity, priority, last_status, 
                 description, summary, network, component, assigned_to_id, assigned_to,
                 time_to_solve, LDA_category, dev_list, acceptable_solving_time, bug_db,
                 project_horizon=None, valid4assignment= None):
        self.idx                 = int(idx)
        self.severity            = severity
        self.priority            = priority
        self.component           = component
        if LDA_category != None:
            self.LDA_category    = int(LDA_category)
        else:
            self.LDA_category    = None
        self.network             = network
        """
        The network is used to monitor a bug in the bug dependency graph. If it is solved, it will be moved to 
        solved bug network.
        """        
        self.open                = [creation_time]
        self.close               = []
        self.blocks              = []
        self.depends_on          = []
        self.status              = "NEW"
        self.last_status         = last_status
        self.real_fixing_time    = 0 # how many days after assignment the bug remained opened until being fixed
        # The below is calculated based on the bug evolutionary database without considering reopening
        self.time_to_solve       = time_to_solve
        self.description         = Functions.convert_nan_to_space(description)
        self.summary             = Functions.convert_nan_to_space(summary)
        self.sum_desc_lemmatized = self.summary_description_lemmatized(ls = True)
        self.sum_desc_lem_not_ls = self.summary_description_lemmatized(ls = False)
        self.assigned_to         = assigned_to.lower()
        self.assigned_to_id      = assigned_to_id
        self.reason_of_invalidity= '-'
        if valid4assignment == None:
            self.valid4assignment    = self.valid_bug_for_assignment(dev_list, acceptable_solving_time, bug_db)
        else:
            self.valid4assignment    = valid4assignment
        if self.valid4assignment:
            self.solving_time_after_simulation             = None
            self.solving_time_after_simulation_accumulated = None
        self.assigned_to_rec     = None
        self.assigned_time       = []
        self.assignment_accuracy = None
        self.assignment_accuracyT= None
        if self.last_status != 'DUPLICATE':
            network.add_node(self)
        self.time_to_be_assigned = 0 # the total time it took for a bug to be assigned since its report
        self.project_horizon     = project_horizon
        self.bug_attribute       = [self.LDA_category, self.project_horizon]
        Bug.add_bug()
            
    def summary_description_lemmatized(self, ls = True):
        return Functions.lemmatizing(self.summary+" "+self.description, ls = ls)
    
    def is_assigned_info_available(self, db):
        """Checks whether the assignee information is available 

        Args:
            db ([DataFrame]): [All bug's information]

        Returns:
            [boolean]: [Whether we found the info or not]
        """
        filtered_db        = db[db.bug == self.idx].copy()
        double_filtered_db = filtered_db[filtered_db.status == 'assigned_to'].copy()
        if len(double_filtered_db) == 0:
            return False
        else:
            return self.assigned_to in np.unique(double_filtered_db.detail)    
    
    def valid_bug_for_assignment(self, dev_list, max_acceptable_solving_time, db): 
        """ Checks whether a bug is valid for assignment

        if assignee info is not available, time to solve is not defined, the developer is not a real person, or it is a META bug, 
        then it returns False. 

        It returns True only if fixing time is less than maximum fixing time, developer is feasible and summary is available.
        
        Args:
            dev_list ([list]): [List of all feasible developers]
            max_acceptable_solving_time ([float]): [Total acceptable solving time]
            db ([DataFrame]): [All bug's information]

        Returns:
            [boolean]: [Whether a bug is valid for assignment or not]
        """
        infeasible_dev_emails = ['inbox@', 'triaged@', 'nobody@', '@js.bugs', '@seamonkey.bugs', 'core.bugs',
                                 'gfx.bugs', 'nss.bugs', 'disabled.tld', 'firefox.bugs', 'mozilla-org.bugs',
                                 'mpopp@mozilla.com', 'mozilla.bugs', '@evangelism',
                                 'libreoffice-bugs@lists.freedesktop.org', '-maint@', '-bugs@',
                                 '-list@redhat.com', '-qa@redhat.com', '-team@', '-mgr@redhat.com',
                                 '-rhel@redhat.com', 'glibc-bugzilla@redhat.com', '-docs@redhat.com',
                                 '-orphan@fedoraproject.org', '-devel@redhat.com', 'rh-bugzilla@ensc.de',
                                 '-bugzilla@linuxnetz.de', '@kernel-bugs.osdl.org', '@vger.kernel.org',
                                 'linuxwifi@intel.com', 'bugme-admin@osdl.org', '@lists.sourceforge.net',
                                 '@kernel-bugs.kernel.org', '@lists.infradead.org', '@lists.arm.linux.org.uk',
                                 'xfs-masters@oss.sgi.com', '-bugs@kde.org', '-design@kde.org', '-devel@kde.org',
                                 '-null@']
        if '[meta]' in self.summary.lower():
            self.reason_of_invalidity = 'META bug'
            return False
        elif not self.is_assigned_info_available(db):
            self.reason_of_invalidity = 'No assigned INFO'
            return False
        elif self.time_to_solve in [0, np.inf]:
            self.reason_of_invalidity = 'No fixing time'
            return False
        elif self.time_to_solve > 356:
            self.reason_of_invalidity = 'Fixing time more than a year'
            return False        
        elif (max_acceptable_solving_time != None):
            if ((self.assigned_to in dev_list) and 
                (self.time_to_solve <= max_acceptable_solving_time) and
                (len(self.sum_desc_lemmatized)>0)):
                return True
            else:
                if (self.assigned_to not in dev_list):
                    self.reason_of_invalidity = 'Infeasible Developer'
                elif (self.time_to_solve > max_acceptable_solving_time):
                    self.reason_of_invalidity = 'Lengthy solving time'
                elif len(self.sum_desc_lemmatized)<=0:
                    self.reason_of_invalidity = 'No summary Description'
                return False
        elif Functions.is_in(self.assigned_to.lower(), infeasible_dev_emails):
            self.reason_of_invalidity = 'Assigned to infeasible person'
            return False
        elif (self.assigned_to in dev_list):
            return True
        else:
            self.reason_of_invalidity = 'Unknown reason!'
            return False

    def update_opening_time (self, reopen_time):
        if reopen_time not in self.open:
            self.open.append(reopen_time)

    def update_closing_time (self, closing_time):
        if (closing_time not in self.close) or (len(self.open) - len(self.close) == 1):
            self.close.append(closing_time)
        assert len(self.close) == len(self.open)
        if (len(self.assigned_time) > 0) and (self.real_fixing_time == 0):
            assert (self.close[-1] - self.assigned_time[-1] + 1) > 0
            self.real_fixing_time = (self.close[-1] - self.assigned_time[-1] + 1) # update fixing duration

    """
    Dependency update
    """
    def blocks_bug (self, blocked_bug, how = 'mutual', add_arc=True):
        """If the current bug blocks a bug, we need to add an arc from the blocking bug to the blocked bug
        
        We ignore META bugs in our BDG as they are not real blocking/blocked bugs.
        Args:
            blocked_bug (Bug): [A bug of class Bug that is blocked by the current bug]
            how (str, optional): [updating both (mutual) or only update one (one-sided)]. Defaults to 'mutual'.
            add_arc (bool, optional): [Should we add arc between two points?]. Defaults to True.
        """
        if ('[meta]' not in self.summary.lower()) and ('[meta]' not in blocked_bug.summary.lower()):
            """If it is not a META bug"""
            self.blocks.append(blocked_bug)
            if how == 'try_mutual':
                try:
                    blocked_bug.depends_on.append(self)
                    if add_arc:
                        self.network.add_arc(self, blocked_bug)
                except:
                    pass
            elif how == 'mutual':
                assert self not in blocked_bug.depends_on
                blocked_bug.depends_on.append(self)
                if add_arc:
                    self.network.add_arc(self, blocked_bug)
            self.degree = len(self.blocks)  # updating the degree of the bugs
            Bug.n_arcs += 1
    def depends_on_bug (self, blocking_bug, how = 'mutual', add_arc=True):
        """ If the current bug is blocked by a bug, we need to add an arc from the blocking bug to the blocked bug
        
        We ignore META bugs in our BDG as they are not real blocking/blocked bugs.
        Args:
            blocking_bug (Bug): [A bug of class Bug that blocks the current bug]
            how (str, optional): [updating both (mutual) or only update one (one-sided)]. Defaults to 'mutual'.
            add_arc (bool, optional): [Should we add arc between two points?]. Defaults to True.
        """
        if ('[meta]' not in self.summary.lower()) and ('[meta]' not in blocking_bug.summary.lower()):
            """If it is not a META bug"""
            #how? updating both (mutual) or only update one (one-sided)
            self.depends_on.append(blocking_bug)
            if how == 'try_mutual':
                try:
                    assert self not in blocking_bug.blocks
                    blocking_bug.blocks.append(self)
                    if add_arc:
                        self.network.add_arc(blocking_bug, self)
                except:
                    pass
            elif how == 'mutual':
                assert self not in blocking_bug.blocks
                blocking_bug.blocks.append(self)
                if add_arc:
                    self.network.add_arc(blocking_bug, self)
            blocking_bug.degree = len(blocking_bug.blocks)
            Bug.n_arcs += 1
    def delete_blocks_bug (self, blocked_bug):
        self.blocks.remove(blocked_bug)
        assert self in blocked_bug.depends_on
        blocked_bug.depends_on.remove(self)
        self.network.remove_arc(self, blocked_bug)
        self.degree = len(self.blocks)
        Bug.n_arcs -= 1
    def delete_depends_on_bug (self, blocking_bug):
        self.depends_on.remove(blocking_bug)
        assert self in blocking_bug.blocks
        blocking_bug.blocks.remove(self)
        self.network.remove_arc(blocking_bug, self)
        Bug.n_arcs -= 1
    def update_dependencies_after_resolution (self):
        """
        We keep its dependencies in case it is reopened, and 
        we only remove dependencies of its neighbors which are still in the network.
        """
        self_blocks_copy = self.blocks.copy()
        self_depends_on_copy = self.depends_on.copy()
        if len(self_blocks_copy) > 0:
            for blocked_bug in self_blocks_copy:
                assert blocked_bug in self.network.network
                blocked_bug.depends_on.remove(self)
        if len(self_depends_on_copy) > 0:
            for blocking_bug in self_depends_on_copy:
                assert blocking_bug in self.network.network
                blocking_bug.blocks.remove(self)
                blocking_bug.degree = len(blocking_bug.blocks)
        self.degree = len(self.blocks)
    """
    Resolution
    """
    def resolve_bug(self, time):
        """ Fix a bug by updating its dependencies, network, fixing time, etc.

        Sometimes, a bug is resolved more than once! which means it is solved first with one status and then
        its status is updated. For instance, RESOLVED as INCOMPLETE then changed to UNCONFIRMED and finally changed to
        RESOLVED as INVALID. We take care of it using an if check. 

        Args:
            time ([int]): [The fixing date]
        """
        if self.network != 'RESOLVED':
            self.update_closing_time(time)
            self.update_dependencies_after_resolution()
            self.network.remove_node(self)
            self.status = 'RESOLVED'
            self.network = 'RESOLVED'
    
    """
    Reopening
    """
    def reopen (self, network, time):
        """ Reopen a bug after it is fixed

        While reopening a bug, its possible dependencies should become active again.

        Args:
            network (BDG): [The Bug Dependency Graph that the opened bug is going to be added to]
            time ([int]):  [The date that this reopening happens]
        """        
        if self.status == 'REOPENED':
            raise Exception(f'Bug {self.idx} is double-reopened!')
        assert self.network == 'RESOLVED'
        self.network = network
        self.network.add_node(self)
        if len(self.blocks) > 0:
            self_blocks_copy = self.blocks.copy()
            for blocked_bug in self_blocks_copy:
                if blocked_bug.network == self.network: # If they are in the same network.
                    assert self not in blocked_bug.depends_on
                    blocked_bug.depends_on.append(self)
                    self.network.add_arc(self, blocked_bug)
                else:
                    assert blocked_bug.network == 'RESOLVED'
                    if (self not in blocked_bug.depends_on):
                        blocked_bug.depends_on.append(self)
                    self.blocks.remove(blocked_bug)
        if len(self.depends_on) > 0:
            self_depends_on_copy = self.depends_on.copy()
            for blocking_bug in self_depends_on_copy:
                if blocking_bug.network == self.network: # If they are in the same network.
                    assert self not in blocking_bug.blocks
                    blocking_bug.blocks.append(self)
                    self.network.add_arc(blocking_bug, self)
                else:
                    assert blocking_bug.network == 'RESOLVED'
                    if self not in blocking_bug.blocks:
                        blocking_bug.blocks.append(self)
                    self.depends_on.remove(blocking_bug)
        self.status = "REOPENED"
        self.open.append(time)
        self.bug_attribute[1] = self.project_horizon #restarting bug attribute
        
    """
    How the object of the class to be printed (presented)
    """
    def __repr__(self):
        return str(self.idx)
    def __str__(self):
        return str(self.__dict__)
    """
    Others
    """    
    def copy(self):
        return copy.deepcopy(self)
    @classmethod
    def add_bug(cls): #class method not for an object
        cls.n_bugs += 1    