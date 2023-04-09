from BDG.BDG import BDG
from components.assignee import Assignee
from components.bug import Bug
from utils.functions import Functions
from utils.prerequisites import *
from simulator.first_run import Finding_parameters_training

class estimating_state_transitions:
    """
    Running the model for the training phase once and 
    extracting transition between different states.
    """
    random_state = 0
    date_number  = 0
    def __init__(self, bug_evolutionary_db, bug_info_db, project, infeasible_dev_emails,
                 project_horizon, max_acceptable_solving_time, available_developers_email,
                 developers_info, resolution = 'actual', verbose = 0):
        if   (verbose == 'nothing') or (str(verbose) == "0"):
            self.verbose = 0
        elif (verbose == 'some')    or (str(verbose) == "1"):
            self.verbose = 1
        elif (verbose == 'all')     or (str(verbose) == "2"):
            self.verbose = 2
        else:
            raise Exception ('Verbose can be chosen from nothing, some, all, 0, 1, or 2.')
        self.project                             = project
        self.resolution                          = resolution
        self.project_horizon                     = project_horizon # comes from the previous phase
        self.resolved_bugs                       = {}    # Keeping track of resolved bugs
        """ sorted evolutionary DB """
        self.bug_evolutionary_db                 = bug_evolutionary_db
        """ Bug information DB """
        self.bug_info_db                         = bug_info_db
        """ Bug Dependency Graph """
        self.BDG                                 = BDG() # initializing bug dependency graph
        """ real start of the project """
        self.assigned_bug_tracker                = pd.DataFrame(columns=self.bug_evolutionary_db.columns)
        self.death                               = string_to_time("2020-01-01 00:00:00", '%Y-%m-%d %H:%M:%S') # end of phase I
        self.testing_time                        = string_to_time("2018-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
        self.last_decade                         = pd.date_range(start=self.bug_evolutionary_db.time.min().date(),
                                                                 end='01/01/2020')
        self.day_counter                         = 0
        self.date                                = self.last_decade[self.day_counter]
        self.infeasible_dev_emails               = infeasible_dev_emails       # coming from phase I
        self.max_acceptable_solving_time         = max_acceptable_solving_time # coming from phase I
        self.available_developers_email          = available_developers_email  # coming from phase I
        self.developers_info                     = developers_info             # coming from phase I
        for dev in self.developers_info.values():
            dev.bugs                = []
            dev.components          = []
            dev.components_tracking = []
            dev.n_assigned_bugs     = 0
        self.today_active_developers             = []
        self.status                              = {"date":[],
                                                    "developers":[],
                                                    "bugs":[]}

    def verboseprint(self, code, **kwargs):
        """
        Args:
            code ([str]): [It tells where the print should happen and what should be printed]
        """
        if self.verbose == 0:
            """It prints nothing!"""
            pass
        if self.verbose in [1,2]:
            """It prints important Warnings."""
            if code == 'testing_phase_started':
                print(f'Starting the TESTING phase using the method {self.resolution}.\n'
                      f'Today is {self.date}.')
            elif code == 'bug_reintroduction':
                print(f'Bug ID {kwargs["bug_id"]} is already in BDG but is now introduced!')
            elif code == 'bug_removal_not_possible':
                print(f'Bug ID {kwargs["bug_id"]} can not be removed!!')
            elif code == 'bug_not_found':
                print(f'where is Bug#{kwargs["bug_id"]}.')
            elif code == "training_time":
                print (f"It took almost {round(self.running_time['training_time']/60)} mins to finish training.")
            elif code == "preparation_time":
                print (f"It took almost {round(self.running_time['model_preparation_for_testing']/60)} "
                        "mins to finish model preparation.")
            elif code == "testing_time":
                print (f"It took almost {round(self.running_time['testing_time']/60)} mins to finish testing.")
            elif code == "SVM_is_fitted":
                print ("SVM model is fitted.")
        if self.verbose == 2:
            """It prints all Warnings and daily progress."""
            if code == 'print_date':
                print(f'Bug ID {kwargs["date"]} can not be removed!!')

    def update_date(self):
        self.day_counter += 1
        self.date         = self.last_decade[self.day_counter]
        if self.project_horizon != None:
            for dev in self.developers_info:
                if self.date >= self.testing_time:
                    self.developers_info[dev].increase_time_limit(self.resolution)
                else:
                    self.developers_info[dev].increase_time_limit()
        estimating_state_transitions.date_number += 1
        for bug_info in self.BDG.bugs_dictionary.values():
            bug_info.bug_attribute[1] -= 1
        if self.date == self.death:
            self.verboseprint(code="testing_time")
            """
            Updating bug validity at the end of the testing phase.
            In this case, we will have an updated bug_info_db with 'valid' column.
            """
            for idx, bug_info in tqdm(self.BDG.bugs_dictionary.items(), desc="Updating DB before finishing 1", position=0, leave=True):
                bug_info.valid4assignment = bug_info.valid_bug_for_assignment(self.available_developers_email,
                                                                              self.max_acceptable_solving_time,
                                                                              self.bug_evolutionary_db)
                self.bug_info_db.loc[idx, "valid"]                = bug_info.valid4assignment
                self.bug_info_db.loc[idx, "reason_of_invalidity"] = bug_info.reason_of_invalidity
            for idx, bug_info in tqdm(self.resolved_bugs.items(), desc="Updating DB before finishing 2", position=0, leave=True):
                bug_info.valid4assignment = bug_info.valid_bug_for_assignment(self.available_developers_email,
                                                                              self.max_acceptable_solving_time,
                                                                              self.bug_evolutionary_db)
                self.bug_info_db.loc[idx, "valid"]                = bug_info.valid4assignment
                self.bug_info_db.loc[idx, "reason_of_invalidity"] = bug_info.reason_of_invalidity
     
    def filter_on_the_date (self, which_data):
        mask = (which_data['time'] > Functions.start_of_the_day(self.date)) & (
                which_data['time'] <= Functions.end_of_the_day(self.date))
        return which_data.loc[mask].copy()
    
    def waybackmachine(self):
        date_formatted               = f'{self.date.year}-{self.date.month}-{self.date.day}'
        self.today_active_developers = [] # resetting the list of active developers daily
        """ Updating based on bug evolutionary info """
        self.filtered_date           = self.filter_on_the_date(self.bug_evolutionary_db)
        bugs_todays_status           = [0 for i in range(len(self.bug_info_db.LDA.unique()))]
        for i in range(len(self.filtered_date)):
            bug_id           = int(self.filtered_date.iloc[i].bug)
            self.current_bug = bug_id
            status_i         = self.filtered_date.iloc[i].status
            self.time        = self.filtered_date.iloc[i].time
            detail           = self.filtered_date.iloc[i].detail 
            if   status_i.lower() == 'introduced' :
                if bug_id not in self.BDG.bugs_dictionary:
                    Bug(idx                     = bug_id,
                        creation_time           = self.day_counter, 
                        severity                = self.bug_info_db.loc[bug_id, 'severity_num'         ],
                        priority                = self.bug_info_db.loc[bug_id, 'priority_num'         ],
                        last_status             = self.bug_info_db.loc[bug_id, 'status'               ],
                        description             = self.bug_info_db.loc[bug_id, 'description'          ],
                        summary                 = self.bug_info_db.loc[bug_id, 'summary'              ],
                        network                 = self.BDG,
                        component               = self.bug_info_db.loc[bug_id, 'component'            ],
                        assigned_to_id          = self.bug_info_db.loc[bug_id, 'assigned_to_detail.id'],
                        assigned_to             = self.bug_info_db.loc[bug_id, 'assigned_to'          ],
                        time_to_solve           = self.bug_info_db.loc[bug_id, 'solving_time'         ],
                        LDA_category            = self.bug_info_db.loc[bug_id, 'LDA'                  ],
                        dev_list                = self.available_developers_email,
                        acceptable_solving_time = self.max_acceptable_solving_time,
                        bug_db                  = self.bug_evolutionary_db,
                        project_horizon         = self.project_horizon,
                        valid4assignment        = self.bug_info_db.loc[bug_id, 'valid'                ],
                       )
                    bugs_todays_status[self.BDG.bugs_dictionary[bug_id].LDA_category] += 1
                else:
                    self.verboseprint(code = 'bug_reintroduction', bug_id = bug_id)
                    
            elif status_i.lower() == 'assigned_to':
                dev_email = detail.lower()
                if bug_id in self.resolved_bugs:
                    """ If the bug is solved before being assigned! """
                    pass
                elif ((self.date < self.testing_time) or
                     (not self.BDG.bugs_dictionary[bug_id].valid4assignment)):
                    if dev_email in self.available_developers_email: # it is a feasible developer
                        for dev_info in self.developers_info.values():
                            if dev_info.search_by_email(dev_email):
                                found = True
                                if (bug_id in self.BDG.bugs_dictionary):
                                    if ((self.date < self.testing_time) or 
                                        (not self.BDG.bugs_dictionary[bug_id].valid4assignment)):
                                        mode_='not_tracking'
                                        if self.BDG.bugs_dictionary[bug_id] not in dev_info.bugs:
                                            """ If the bug is not reassigned """
                                            dev_info.assign_and_solve_bug(bug         = self.BDG.bugs_dictionary[bug_id],
                                                                          time_       = self.day_counter,
                                                                          mode_       = mode_,
                                                                          resolution_ = self.resolution)
                                            self.today_active_developers.append(dev_email)
                                            self.assigned_bug_tracker = pd.concat([self.assigned_bug_tracker,
                                                                                   pd.DataFrame([[bug_id,'ASSIGNED_TO', dev_email ,"-"]],
                                                                                                columns=self.bug_evolutionary_db.columns)], 
                                                                                  ignore_index=True)
                                    else:
                                        raise Exception("problem with if statement.")
                                break
                        if not found:
                            raise Exception(f"Developer not found for bug id {bug_id}")

            elif (status_i.lower() == 'resolved') or (status_i.lower() == 'closed'):
                if bug_id in self.BDG.bugs_dictionary:
                    if ((self.date < self.testing_time) or 
                        (not self.BDG.bugs_dictionary[bug_id].valid4assignment)): #or
                        # (self.BDG.bugs_dictionary[bug_id].open[0] < self.testing_time_counter)):
                        """
                        If we are in training phase or 
                        the bug is not assigned to an active developer
                        """
                        if bug_id not in self.resolved_bugs:
                            self.resolved_bugs[bug_id] = self.BDG.bugs_dictionary[bug_id]
                            self.BDG.bugs_dictionary[bug_id].resolve_bug(self.day_counter)
                        """
                        Removing the fixed bug from the assigned bug tracker.                        
                        """
                        self.assigned_bug_tracker = self.assigned_bug_tracker[self.assigned_bug_tracker.bug != bug_id].copy()
                        try:
                            self.today_active_developers.append(self.BDG.bugs_dictionary[bug_id].assigned_to_rec)
                        except KeyError:
                            self.today_active_developers.append(self.resolved_bugs[bug_id].assigned_to_rec)
                        
            if (self.resolution != 'actual'):
                raise Exception(f'Wrong resolution = {self.resolution}')

        self.verboseprint(code = 'print_date', date = date_formatted)

        """ Updating system status """    
        self.status["date"].append(date_formatted    )
        self.status["bugs"].append(bugs_todays_status)
        dev_todays_availability   = []
        for dev_email in self.available_developers_email:
            if (dev_email in self.assigned_bug_tracker.detail) or (dev_email in self.today_active_developers):
                dev_todays_availability.append(1)
            else:
                dev_todays_availability.append(0)
        self.status["developers"].append(dev_todays_availability)
        self.update_date()

    """"How to print it"""
    def __repr__(self):
        return f'A Wayback Machine for {str(self.project)}; Second run to extract Transitions'
    def __str__(self):
        return str(self.__dict__.keys())
    
"""
Time to Run the second phase
"""
#######################################
##                                   ##
##      author: Hadi Jahanshahi      ##
##     hadi.jahanshahi@ryerson.ca    ##
##          Data Science Lab         ##
##                                   ##
#######################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The parser will handle hyperparameters of the second Run')

    parser.add_argument(
        '--project', 
        default = 'EclipseJDT',
        type    = str,
        help    = 'it can be selected from this list: [LibreOffice, Mozilla, EclipseJDT, GCC]'
    )

    parser.add_argument(
        '--verbose',
        default = 0,
        type    = int,
        help    = 'it can be either: [0, 1, 2, nothing, some, all]'
    )

    wayback_param        = parser.parse_args()
    project              = wayback_param.project
    verbose              = wayback_param.verbose

    with open (f"dat/{project}/phaseI.pickle", "rb") as file:
        phaseI = pickle.load(file)
    
    bug_evolutionary_db         = phaseI.bug_evolutionary_db
    bug_info_db                 = phaseI.bug_info_db
    infeasible_dev_emails       = phaseI.infeasible_dev_emails
    available_developers_email  = phaseI.available_developers_email
    project_horizon             = phaseI.project_horizon
    max_acceptable_solving_time = phaseI.max_acceptable_solving_time
    developers_info             = phaseI.developers_info

    simulation           = estimating_state_transitions(bug_evolutionary_db, bug_info_db, project, infeasible_dev_emails,
                                                        project_horizon, max_acceptable_solving_time, available_developers_email,
                                                        developers_info, resolution = 'actual', verbose = 0)
    stop_date            = len(pd.date_range(start=simulation.bug_evolutionary_db.time.min().date(), end='31/12/2019'))
    stop_training        = len(pd.date_range(start=simulation.bug_evolutionary_db.time.min().date(), end='31/12/2017'))#training
    stop_testing         = len(pd.date_range(start='01/01/2018', end='31/12/2019')) # whole testing period

    for i in tqdm(range(stop_training), desc="simulating days", position=0, leave=True):
        simulation.waybackmachine()

    """ Keep a backup of the model at the end of the training """
    with open(f'dat/{project}/phaseII_training.pickle', 'wb') as file:
        pickle.dump(simulation, file) # use `pickle.load` to do the reverse
        
    for i in tqdm(range(stop_testing), desc="simulating days (testing)", position=0, leave=True):
        simulation.waybackmachine()

    """ Keep a backup of the model at the end of the testing """
    with open(f'dat/{project}/phaseII_testing.pickle', 'wb') as file:
        pickle.dump(simulation, file) # use `pickle.load` to do the reverse
        
# python3.9 simulator/second_run.py --project=LibreOffice