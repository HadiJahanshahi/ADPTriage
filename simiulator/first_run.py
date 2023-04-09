from unicodedata import name
from BDG.BDG import BDG
from components.assignee import Assignee
from components.bug import Bug
from utils.functions import Functions
from utils.prerequisites import *


class Finding_parameters_training:
    """
    Running the model for the training + testing phase once and 
    extracting LDA categories, assignment time, and SVM model.
    """
    random_state = 0
    date_number  = 0
    def __init__(self, bug_evolutionary_db, bug_info_db, project, infeasible_dev_emails, 
                 resolution = 'actual', verbose = 0):
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
        self.project_horizon                     = None # self.time_limit_calculation()
        self.summary_plus_desc                   = [] # list
        self.summary_plus_desc_not_ls            = [] # not list
        self.SVM_Y_label                         = []
        self.LDA_time_to_solve                   = []
        self.priorityY                           = []
        self.all_opened_bugs                     = []
        self.resolved_bugs                       = {}    # Keeping track of resolved bugs
        """ sorted evolutionary DB """
        self.bug_evolutionary_db                 = bug_evolutionary_db
        """ Bug information DB """
        self.bug_info_db                         = bug_info_db
        self.bug_info_db['solving_time'        ] = None
        self.bug_info_db['LDA_category'        ] = None
        self.bug_info_db['assignment_gap'      ] = None
        self.bug_info_db['pred_assignment_gap' ] = None
        self.bug_info_db['reason_of_invalidity'] = None
        self.bug_info_db['valid'               ] = None
        """ Bug Dependency Graph """
        self.BDG                                 = BDG() # initializing bug dependency graph
        """ real start of the project """
        self.death                               = string_to_time("2020-01-01 00:00:00", '%Y-%m-%d %H:%M:%S') # end of phase I
        self.testing_time                        = string_to_time("2018-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
        self.last_decade                         = pd.date_range(start=self.bug_evolutionary_db.time.min().date(),
                                                                 end='01/01/2020')
        self.day_counter                         = 0
        self.date                                = self.last_decade[self.day_counter]
        for each_bug in tqdm(self.bug_info_db.index, desc="Fixing time calculation", position=0, leave=True):
            self.bug_info_db.loc[each_bug,'solving_time'  ] = self.fixing_time_calculation(each_bug)        
        for each_bug in tqdm(self.bug_info_db.index, desc="Assignment gap calculation", position=0, leave=True):
            self.bug_info_db.loc[each_bug,'assignment_gap'] = self.assignment_gap_calculation(each_bug)
        self.infeasible_dev_emails               = infeasible_dev_emails
        self.max_acceptable_solving_time         = self.acceptable_solving_time()
        self.available_developers_email          = self.possible_developers()
        self.developers_info                     = self.developers_info_extraction()
        self.running_time                        = {"training_time": 0, "model_preparation_for_testing":0,
                                                    "testing_time":0} # execution time per phase in seconds
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

    def developers_info_extraction(self):
        "dev info db"
        developers_db = {}
        for dev in self.available_developers_email:
            dev_info = self.bug_info_db[self.bug_info_db['assigned_to_detail.email'].map(lambda x: x.lower()
                                                                                        ) == dev].iloc[0]
            idd      = dev_info['assigned_to_detail.id']
            email_   = dev_info['assigned_to_detail.email'].lower()
            developers_db[idd] = Assignee(email           = email_,
                                          id_             = idd,
                                          name            = dev_info['assigned_to_detail.real_name'],
                                          LDA_experience  = None,
                                          time_limit_orig = self.project_horizon)
        return developers_db

    def assignment_gap_calculation(self, bug_id):
        """Using bug info and bug evolutionary database to calculate 
        the time it takes for a bug to be assigned 
        
        Assigned time = Assigned date - Reporting date + 1

        Args:
            bug_id ([int]): [Bug ID]

        Returns:
            [int]: [Assignment Gap]
        """
        who_assigned  = self.bug_info_db.loc    [bug_id, 'assigned_to']
        when_reported = self.bug_info_db.loc    [bug_id, 'creation_time'].date()
        filtered      = self.bug_evolutionary_db[self.bug_evolutionary_db.bug == bug_id].copy()
        try:
            if 'assigned_to' in list(filtered.status):
                assigned_filt = filtered[(filtered.status == 'assigned_to') & (filtered.detail == who_assigned)].copy()
                if len(assigned_filt) != 0:
                    assigned_time = list(assigned_filt.time)[0].date()
                else:
                    assigned_filt = filtered[(filtered.status == 'ASSIGNED')]
                    assigned_time = list(assigned_filt.time)[0].date() 
            else:
                assigned_filt = filtered[(filtered.status == 'ASSIGNED')]
                assigned_time = list(assigned_filt.time)[0].date()            
            assignment_gap = (assigned_time - when_reported).days + 1
        except IndexError:
            assignment_gap = np.inf
        except ValueError:
            print(bug_id)
            print(who_assigned)
            print(filtered)
            raise Exception('Check the prints')
        return assignment_gap
            
    def fixing_time_calculation(self, bug_id):
        """Using bug info and bug evolutionary database to calculate fixing time 
        
        Fixing time = Fixing date - Assigned date + 1

        Args:
            bug_id ([int]): [Bug ID]

        Returns:
            [int]: [Fixing Duration]
        """
        who_assigned  = self.bug_info_db.loc    [bug_id, 'assigned_to']
        filtered      = self.bug_evolutionary_db[self.bug_evolutionary_db.bug == bug_id].copy()
        try:
            if 'assigned_to' in list(filtered.status):
                assigned_filt = filtered[(filtered.status == 'assigned_to') & (filtered.detail == who_assigned)].copy()
                if len(assigned_filt) != 0:
                    assigned_time = list(assigned_filt.time)[0].date()
                else:
                    assigned_filt = filtered[(filtered.status == 'ASSIGNED')]
                    assigned_time = list(assigned_filt.time)[0].date() 
            else:
                assigned_filt = filtered[(filtered.status == 'ASSIGNED')]
                assigned_time = list(assigned_filt.time)[0].date()            
            assigned_idx  = assigned_filt.index[0]
            filtered_res  = filtered[filtered.index >= assigned_idx].copy() # filter on dates after assigning
            resolved_filt = filtered_res[(filtered_res.status == 'RESOLVED') | (filtered_res.status == 'CLOSED')]
            resolved_time = list(resolved_filt.time)[0].date()
            solving_time  = (resolved_time - assigned_time).days + 1
        except IndexError:
            solving_time = np.inf
        except ValueError:
            print(bug_id)
            print(who_assigned)
            print(filtered)
            raise Exception('Check the prints')
        return solving_time

    def update_developers(self):
        new_db = {}
        for dev in self.developers_info:
            if (self.developers_info[dev].email.lower()) in self.available_developers_email:
                new_db[dev]                 = self.developers_info[dev]
                new_db[dev].time_limit      = [self.project_horizon]
                new_db[dev].time_limit_orig = self.project_horizon
            # else:
            #     raise Exception (f"Developer {self.developers_info[dev].email.lower()} is not in available_developers_email.")
            #     """ WE MAY NEED ANOTHER LOOP FOR self.available_developers_email in that case."""
        for dev in self.available_developers_email:
            dev_info = self.bug_info_db[self.bug_info_db['assigned_to_detail.email'].map(lambda x: x.lower()
                                                                                        ) == dev].iloc[0]
            idd      = dev_info['assigned_to_detail.id']
            if idd not in new_db:
                email_   = dev_info['assigned_to_detail.email'].lower()
                new_db[idd] = Assignee(email           = email_,
                                       id_             = idd,
                                       name            = dev_info['assigned_to_detail.real_name'],
                                       LDA_experience  = None,
                                       time_limit_orig = self.project_horizon)
        return new_db.copy()

    def acceptable_solving_time(self):
        """We find IQR of the bugs opened during the training period."""
        training_bugs = self.bug_info_db[self.bug_info_db.creation_time < self.testing_time].copy()
        solving_times = training_bugs.solving_time
        all_numbers   = solving_times[solving_times != np.inf]
        all_numbers   = all_numbers  [all_numbers   < 356*2  ] # taking less than two years
        all_numbers   = all_numbers  [all_numbers   !=      0]
        Q1, Q3        = all_numbers.quantile([0.25,0.75])
        IQR           = Q3 - Q1
        Max_acceptable_solving_time = Q3 + (1.5*IQR)
        print('Max acceptable solving time', Max_acceptable_solving_time, f'(Q1:{Q1}-Q3:{Q3}-IQR:{IQR})')
        return Max_acceptable_solving_time

    def possible_developers(self):
        if self.max_acceptable_solving_time != None:
            dbcopy   = self.bug_info_db[self.bug_info_db.solving_time <= self.max_acceptable_solving_time].copy()
        else:
            dbcopy   = self.bug_info_db.copy()
        testing               = dbcopy[dbcopy['creation_time'] >= self.testing_time] # filtering on test data
        dbcopy                = dbcopy[dbcopy['creation_time'] <  self.testing_time] # filtering on training data
        dbcopy_feasible_bugs  = set(dbcopy.index) # indexes of all the bugs that have acceptable solving times
        dev_freq              = {}
        evol_db               = self.bug_evolutionary_db[((self.bug_evolutionary_db.time < self.testing_time) &
                                                          (self.bug_evolutionary_db.status == 'assigned_to'))]
        for i in range(len(evol_db)):
            dev = evol_db.iloc[i].detail
            bug = evol_db.iloc[i].bug
            if Functions.is_in(dev.lower(), self.infeasible_dev_emails):
                """Bug is in inbox or not assigned yet. Infeasible assignee emails"""
                pass
            elif (dev.lower() not in dev_freq):
                dev_freq[dev.lower()]  = []
                if (bug not in dev_freq[dev.lower()]) and (bug in dbcopy_feasible_bugs):
                    dev_freq[dev.lower()].append(bug)
            else:
                if (bug not in dev_freq[dev.lower()]) and (bug in dbcopy_feasible_bugs):
                    dev_freq[dev.lower()].append(bug)
        dev_freq = {key:len(val) for key, val in dev_freq.items()}
        """
        determining how many feasible bugs each developer has fixed during the training phase
        """
        q75, q25 = np.percentile(list(dev_freq.values()), [75 ,25])
        IQR      = q75 - q25
        """ filter out low seasonal developers (Less than IQR)"""
        key_val  = dev_freq.copy().items()
        for dev_, freq in key_val:
            if freq < IQR:
                del dev_freq[dev_]
            if Functions.is_in(dev_.lower(), self.infeasible_dev_emails):
                raise Exception ('why are bugs that are in inbox and not triaged still here?')
        dev_in_testing = testing.assigned_to.unique()
        keys_ = dev_freq.copy().keys()
        for dev_ in keys_:
            if dev_ not in dev_in_testing:
                """
                Removing developers that are not active in the testing phase
                """
                del dev_freq[dev_]            
        print('The # developers', len(dev_freq))
        return list(dev_freq.keys())
    
    def is_assigned_info_available(self, bug_id):
        """Checks whether the assignee information is available 

        Args:
            bug_id ([int]): [ID of the bug we are going to check]

        Returns:
            [boolean]: [Whether we found the info or not]
        """
        filtered_db        = self.bug_evolutionary_db[self.bug_evolutionary_db.bug == bug_id].copy()
        double_filtered_db = filtered_db[filtered_db.status == 'assigned_to'].copy()
        if len(double_filtered_db) == 0:
            return False
        elif self.assigned_to in np.unique(double_filtered_db.detail):
            """
            Is the bug assigned to the exact developer?
            """
            return True
        elif 'ASSIGNED' in filtered_db.status:
            """Even if the exact developer is not determined, if _ASSIGNED_ status is available,
            We still consider it to be assigned to the same developer.
            """
            return True
        else:
            return False
    
    def valid_bug_for_assignment(self, bug_id): 
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
        if '[meta]' in self.summary.lower():
            self.reason_of_invalidity = 'META bug'
            return False
        elif not self.is_assigned_info_available(bug_id):
            self.reason_of_invalidity = 'No assigned INFO'
            return False
        elif self.time_to_solve in [0, np.inf]:
            self.reason_of_invalidity = 'No fixing time'
            return False
        elif self.time_to_solve > 356:
            self.reason_of_invalidity = 'Fixing time more than a year'
            return False        
        elif (self.max_acceptable_solving_time != None):
            if ((self.assigned_to in self.available_developers_email) and 
                (self.time_to_solve <= self.max_acceptable_solving_time) and
                (len(self.sum_desc_lemmatized)>0)):
                return True
            else:
                if (self.assigned_to not in self.available_developers_email):
                    self.reason_of_invalidity = 'Infeasible Developer'
                elif (self.time_to_solve > self.max_acceptable_solving_time):
                    self.reason_of_invalidity = 'Lengthy solving time'
                elif len(self.sum_desc_lemmatized)<=0:
                    self.reason_of_invalidity = 'No summary Description'
                return False
        elif Functions.is_in(self.assigned_to.lower(), self.infeasible_dev_emails):
            self.reason_of_invalidity = 'Assigned to infeasible person'
            return False
        elif (self.assigned_to in self.available_developers_email):
            return True
        else:
            self.reason_of_invalidity = 'Unknown reason!'
            return False

    def LDA_Topic_Modeling (self, min_values=5):
        """ Finding the optimal number of topics using arun_metric """
        self.arun_metric()
        """applying LDA"""
        self.LDA_model       = models.LdaMulticore(self.corpus_tfidf, num_topics=self.optimal_n_topics,
                                                   id2word=self.dictionary, passes=20, workers=4)
        LDA_category         = [Functions.max_tuple_index(self.LDA_model[i]) for i in self.bow_corpus]
        categories            = np.arange(self.optimal_n_topics)
        """Creating Dev/LDA/time-to-fix table"""
        self.time_to_fix_LDA = pd.DataFrame(None, index = self.available_developers_email,
                                            columns = categories)
        for dev in self.available_developers_email:
            index                      = Functions.return_index(dev, self.SVM_Y_label)
            filtered_LDA_cat           = np.array(LDA_category)[index]
            filtered_LDA_time_to_solve = np.array(self.LDA_time_to_solve)[index]
            for cat in categories:
                self.time_to_fix_LDA.loc[dev, cat] = self.average_days(filtered_LDA_time_to_solve,
                                                                       Functions.return_index(cat, filtered_LDA_cat))
        """updating None Values of table"""
        self.dev_profile_collaborative(min_values)
        for dev_id, dev_info in self.developers_info.items():
            dev_info.LDA_experience = list(self.time_to_fix_LDA.loc[dev_info.email])
    
    @staticmethod
    def average_days (list_, idx):
        if len(list_[idx]) > 0:
            return np.ceil(list_[idx].mean())
        else: 
            return None
                
    @staticmethod
    def resort_dict (dictionary, new_keys):
        """ needed to sort developer list according to the SVM model """
        assert len(dictionary) == len(new_keys)
        new_dic = {}
        for dev_email in new_keys:
            for dev_id, dev_info in dictionary.items():
                if dev_info.search_by_email(dev_email.lower()):
                    new_dic[dev_id] = dev_info
                    break
        return new_dic        
        
    @staticmethod
    def symmetric_kl_divergence(p, q):
        """ Calculates symmetric Kullback-Leibler divergence.
        """
        
        return np.sum([stats.entropy(p, q), stats.entropy(q, p)])
    
    def dev_profile_collaborative(self, min_value=5):
        """
        min_value=1: the updated date cannot be less than a day.
        """
        dev_profile     = self.time_to_fix_LDA.copy()
        dev_profile_upd = self.time_to_fix_LDA.copy()
        theta           = dev_profile_upd.shape[1]/2
        for dl in dev_profile.index:
            Pu   = dev_profile[dev_profile.index==dl]
            F_Pu = Pu.max(axis=1)[0]
            for bug in dev_profile.columns:
                if (dev_profile.at[dl,bug]==None):
                    Nu = []
                    for de in dev_profile.index:
                        if ((de!=dl) and (dev_profile.at[de,bug]!=None)):
                            Nu.append(de)
                    Nu_upd = []
                    for Pv_dev in Nu:
                        Num = 0
                        den = 0
                        Pv_dev_prof = dev_profile[dev_profile.index==Pv_dev]
                        Pv_sub      = []
                        Pu_sub      = []
                        for i in dev_profile.columns:
                            if (((Pu[i][0])!=None) & ((Pv_dev_prof[i][0])!=None)):
                                Pu_sub.append(Pu[i][0])
                                Pv_sub.append(Pv_dev_prof[i][0])
                        if len(Pv_sub) > 0:
                            weight     = min((len(Pv_sub)/theta),1)
                            cosine_sim = (np.dot(np.array(Pu_sub),np.array(Pv_sub)) /
                                          (np.linalg.norm(Pu_sub)*np.linalg.norm(Pv_sub)))
                            sim_pu_pv  = (cosine_sim)*weight
                            Nu_upd.append((sim_pu_pv,Pv_dev))
                    Nu_new = sorted(Nu_upd,reverse=True)
                    for i, val in enumerate(Nu_new[:10]):
                        F_Pv = dev_profile[dev_profile.index==val[1]].max(axis=1)[0]
                        Num += ((val[0])*(dev_profile.at[val[1],bug]/F_Pv))
                        den += val[0]
                    try:
                        dev_profile_upd.at[dl,bug] = np.ceil(max(round((F_Pu*(Num/den)),2), min_value))
                    except ZeroDivisionError:
                        """
                           It may happen that none of the similar dev has experience with the category 
                        that is None for the current developer. Then "den" will become zero.
                        """
                        dev_profile_upd.at[dl,bug] = np.ceil(min_value)
        self.time_to_fix_LDA = dev_profile_upd

    def predict_LDA (self, bug_n):
        try:
            doc2bow = self.dictionary.doc2bow(self.BDG.bugs_dictionary[bug_n].sum_desc_lemmatized)
            self.BDG.bugs_dictionary[bug_n].LDA_category = Functions.max_tuple_index(self.LDA_model[doc2bow])
        except:
            doc2bow = self.dictionary.doc2bow(self.resolved_bugs[bug_n].sum_desc_lemmatized)
            self.resolved_bugs[bug_n].LDA_category = Functions.max_tuple_index(self.LDA_model[doc2bow])

    def arun_metric(self, min_topics=5, max_topics=25, iteration=5):
        """ Calculates Arun et al metric.."""
        corpus_length_vector = np.array([sum(frequency for _, frequency in document) for document in self.corpus_tfidf])   
        Kl_matrix  = []
        for j in tqdm(range(iteration), desc="LDA_best_n_topics", position = 0):
            result = []
            topic  = []
            for i in range(min_topics, max_topics):
                # initiates LDA.
                lda = models.ldamodel.LdaModel(corpus      = self.corpus_tfidf,
                                               id2word     = self.dictionary,
                                               num_topics  = i,
                                               iterations  = 80,
                                               random_state= Finding_parameters_training.random_state+j+i*10)
                Finding_parameters_training.random_state += 1
                # Calculates raw LDA matrix.
                matrix                     = lda.expElogbeta
                # Calculates SVD for LDA matrix.
                U, document_word_vector, V = np.linalg.svd(matrix)
                # Gets LDA topics.
                lda_topics                 = lda[self.corpus_tfidf]
                # Calculates document-topic matrix.
                term_document_matrix       = matutils.corpus2dense(lda_topics, lda.num_topics).transpose()
                document_topic_matrix      = corpus_length_vector.dot(term_document_matrix)
                document_topic_vector      = document_topic_matrix + 0.0001
                document_topic_norm        = np.linalg.norm(corpus_length_vector)
                document_topic_vector      = document_topic_vector / document_topic_norm
                result.append(self.symmetric_kl_divergence(document_word_vector,document_topic_vector))
                topic.append(i)
            Kl_matrix.append(result)
        output                = np.array(Kl_matrix).mean(axis=0)
        self.optimal_n_topics = topic[output.argmin()]
        
    def corpus_update(self):
        """Creating a vocabulary corpus consisting of words in summary and description of the bugs. 
        """
        all_words_freq = {}
        for idx, bug in self.resolved_bugs.items():
            temp_sum = bug.sum_desc_lemmatized
            for word in temp_sum:
                if word not in all_words_freq:
                    all_words_freq[word] = 0
                all_words_freq[word] += 1
        for idx, bug in self.BDG.bugs_dictionary.items():
            temp_sum = bug.sum_desc_lemmatized
            for word in temp_sum:
                if word not in all_words_freq:
                    all_words_freq[word] = 0
                all_words_freq[word] += 1
        self.corpus_freq = []
        for word, freq in all_words_freq.items():
            """Removing infrequent or too frequent words."""
            if (freq > 15) and (freq < (len(self.bug_info_db)/2)):
                self.corpus_freq.append(word)
                
    def filter_low_freq (self, text, StopWords = STOPWORDS):
        """Filtering stop words and peculiar vocabularies

        Args:
            text ([str]): [A sentence to be cleaned]
            StopWords ([list], optional): [List of stopwords]. Defaults to STOPWORDS.

        Returns:
            [list]: [cleaned list of meaningful vocabularies in the given texts]
        """
        text = [word for word in text if ((word in self.corpus_freq) and (word not in StopWords)
                                          and (len(word) < 20)  and (len(word) > 1))]
        return text
    
    @staticmethod
    def no_transform(text):
        return text
    
    def create_db_for_SVM_LDA (self):
        training_bugs = list(self.bug_info_db[self.bug_info_db.creation_time <self.testing_time].index)
        time_to_fix = []
        """resolved bugs"""
        for bug in self.resolved_bugs.values():
            if (bug.valid4assignment) and (bug.time_to_solve in [0, np.inf]):
                bug.valid4assignment     = False # ignoring the bugs with fixing time equal to zero (Same day fixing)
                bug.reason_of_invalidity = 'No fixing time'
#            elif (bug.valid4assignment) and (bug.time_to_solve > 0):
            elif (bug.idx in training_bugs) and (bug.time_to_solve not in [0, np.inf]) and (bug.time_to_solve < 356):
                time_to_fix.append(bug.time_to_solve)
        """reopened bugs"""
        for bug in self.BDG.bugs_dictionary.values():
            # if (bug.valid4assignment) and (bug.time_to_solve > 0):
            if (bug.idx in training_bugs) and (bug.time_to_solve not in [0, np.inf]) and (bug.time_to_solve < 356):# and (bug.valid4assignment):
                time_to_fix.append(bug.time_to_solve)
        """finding the max_acceptable_solving_time"""
        Q1, Q3                           = pd.Series(time_to_fix).quantile([0.25, 0.75])
        self.project_horizon             = int(Q3) # self.time_limit_calculation()
        if self.project_horizon > 30:
            """ We limit project_horizon to a month max"""
            self.project_horizon = 30
        print(f'Project Horizon is equal to {self.project_horizon}.')
        self.mean_solving_time           = np.mean(time_to_fix)
        IQR                              = Q3 - Q1
        self.max_acceptable_solving_time = Q3 + (1.5*IQR)
        print('NEW Max acceptable solving time', self.max_acceptable_solving_time, f'(Q1:{Q1}-Q3:{Q3}-IQR:{IQR})')        
        """resolved bugs"""
        for bug in self.resolved_bugs.values():
            bug.valid4assignment = bug.valid_bug_for_assignment(self.available_developers_email,
                                                                self.max_acceptable_solving_time,
                                                                self.bug_evolutionary_db)
            if   (bug.valid4assignment) and (bug.time_to_solve > self.max_acceptable_solving_time):
                bug.valid4assignment     = False # ignoring the bugs with outlier fixing time
                bug.reason_of_invalidity = 'Lengthy solving time'
            elif ((bug.valid4assignment) and 
                  (bug.time_to_solve not in [0, np.inf]) and 
                  (bug.time_to_solve <= self.max_acceptable_solving_time)):
                self.summary_plus_desc.append(bug.sum_desc_lemmatized)
                self.summary_plus_desc_not_ls.append(bug.sum_desc_lem_not_ls)
                self.SVM_Y_label.append(bug.assigned_to)
                self.LDA_time_to_solve.append(bug.time_to_solve)
                self.priorityY.append(bug.priority)
        """reopened bugs"""
        for bug in self.BDG.bugs_dictionary.values():
            bug.valid4assignment = bug.valid_bug_for_assignment(self.available_developers_email,
                                                                self.max_acceptable_solving_time,
                                                                self.bug_evolutionary_db)
            if   (bug.valid4assignment) and (bug.time_to_solve > self.max_acceptable_solving_time):
                bug.valid4assignment     = False # ignoring the bugs with outlier fixing time
                bug.reason_of_invalidity = 'Lengthy solving time'
            elif ((bug.valid4assignment) and 
                  (bug.time_to_solve not in [0, np.inf]) and 
                  (bug.time_to_solve <= self.max_acceptable_solving_time)):
                self.summary_plus_desc.append(bug.sum_desc_lemmatized)
                self.summary_plus_desc_not_ls.append(bug.sum_desc_lem_not_ls)
                self.SVM_Y_label.append(bug.assigned_to)
                self.LDA_time_to_solve.append(bug.time_to_solve)
                self.priorityY.append(bug.priority)
        self.dictionary   = Dictionary(self.summary_plus_desc)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
        self.bow_corpus   = [self.dictionary.doc2bow(doc) for doc in self.summary_plus_desc]
        self.tfidf        = models.TfidfModel(self.bow_corpus)
        self.corpus_tfidf = self.tfidf[self.bow_corpus]
        
    def SVM_model(self):
        self.corpus_update() # updating the corpus needed for filter_low_freq
        self.tfidf_svm                  = TfidfVectorizer(tokenizer=self.no_transform, 
                                                          preprocessor=self.filter_low_freq)
        X_tfidf                         = self.tfidf_svm.fit_transform(self.summary_plus_desc)
        self.svm_model                  = svm.SVC(C=1000.0, kernel='linear', degree=5, 
                                                  gamma=0.001, probability=True)
        self.svm_model.fit(X_tfidf, self.SVM_Y_label)
        self.svm_model_priority         = svm.SVR(C=1000.0, kernel='linear', degree=5, gamma=0.001)
        self.svm_model_priority.fit(X_tfidf, self.priorityY)
        self.available_developers_email = list(self.svm_model.classes_)
        self.developers_info            = self.update_developers() # updating available developers info

    def update_date(self):
        self.day_counter += 1
        self.date         = self.last_decade[self.day_counter]
        if self.project_horizon != None:
            for dev in self.developers_info:
                if self.date >= self.testing_time:
                    self.developers_info[dev].increase_time_limit(self.resolution)
                else:
                    self.developers_info[dev].increase_time_limit()
        Finding_parameters_training.date_number += 1
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
        if Finding_parameters_training.date_number == 0:                              ####################
            """                                                                       ####################
            It is the start of the training phase.                                    ####################
            """                                                                       ####################
            self.start_training = time.time()                                         #     PHASE(I)     #
        date_formatted = f'{self.date.year}-{self.date.month}-{self.date.day}'        #      STARTS      #        
        """ Updating based on bug evolutionary info """                               #       HERE       #
        self.filtered_date = self.filter_on_the_date(self.bug_evolutionary_db)        ####################
        for i in range(len(self.filtered_date)):                                      ####################
            bug_id           = int(self.filtered_date.iloc[i].bug)                    ####################
            self.current_bug = bug_id                                                 ####################
            status_i         = self.filtered_date.iloc[i].status
            self.time        = self.filtered_date.iloc[i].time
            detail           = self.filtered_date.iloc[i].detail
            if bug_id not in self.all_opened_bugs:
                """Add it to the list of opened bugs"""
                self.all_opened_bugs.append(bug_id)
            if   status_i.lower() == 'introduced' :
                # creating a bug.
                if bug_id not in self.BDG.bugs_dictionary:
                    """ Ensuring the bug is not added before """ 
                    if (self.date < self.testing_time):
                        ac_sol_t = 100# self.max_acceptable_solving_time 
                        # 100 # We do not accept bugs with fixing time more than 100 days;
                    else:
                        ac_sol_t = self.max_acceptable_solving_time
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
                        LDA_category            = None,
                        dev_list                = self.available_developers_email,
                        acceptable_solving_time = ac_sol_t,
                        bug_db                  = self.bug_evolutionary_db
                       )
                    self.bug_info_db.loc[self.bug_info_db.index==bug_id, 'valid'] = self.BDG.bugs_dictionary[bug_id].valid4assignment
                    if (self.date >= self.testing_time):
                        self.predict_LDA(bug_id)
                else:
                    self.verboseprint(code = 'bug_reintroduction', bug_id = bug_id)

            elif status_i.lower() == 'assigned_to':
                """
                If we are in Waybackmachine mode or still in training phase or 
                the bug is not assigned in real world.
                """
                dev_email = detail.lower()
                if bug_id in self.resolved_bugs:
                    """ If the bug is solved before being assigned! """
                    pass
                elif ((self.resolution == 'actual')   or
                      (self.date < self.testing_time) or
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
                                            if self.BDG.bugs_dictionary[bug_id].valid4assignment:
                                                solving_date  = self.date + np.timedelta64(
                                                    math.ceil(self.BDG.bugs_dictionary[bug_id
                                                    ].solving_time_after_simulation_accumulated), 'D')
                                    elif ((self.date >= self.testing_time) and 
                                          (self.resolution == 'actual') and
                                          (self.BDG.bugs_dictionary[bug_id].valid4assignment)):
                                        mode_='not_tracking'
                                        if self.BDG.bugs_dictionary[bug_id] not in dev_info.bugs:
                                            """ If the bug is not reassigned """
                                            if self.BDG.bugs_dictionary[bug_id].LDA_category == None:
                                                self.predict_LDA(bug_id)
                                            dev_info.assign_and_solve_bug(bug         = self.BDG.bugs_dictionary[bug_id],
                                                                          time_       = self.day_counter,
                                                                          mode_       = mode_,
                                                                          resolution_ = self.resolution)
                                    else:
                                        raise Exception("problem with if statement.")
                                break
                        if not found:
                            raise Exception(f"Developer not found for bug id {bug_id}")

            elif status_i.lower() == 'reopened'   :
                """ 
                Maybe the bug is assigned but not solved yet. So we cannot re-open it.
                """
                if (bug_id in self.resolved_bugs) and (self.date < self.testing_time):
                    self.resolved_bugs[bug_id].reopen(self.BDG, self.day_counter)
                    del self.resolved_bugs[bug_id]
                    
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
            
            if (self.resolution != 'actual'):
                raise Exception(f'Wrong resolution = {self.resolution}')

        self.verboseprint(code = 'print_date', date = date_formatted)
        self.update_date()

        if self.date == self.testing_time:
            self.running_time["training_time"]   = round(time.time() - self.start_training)
            self.verboseprint(code = 'training_time')
            self.start_model_updates_for_testing = time.time()
            self.verboseprint(code = 'testing_phase_started')
            """Prepare data for SVM and LDA""" 
            self.create_db_for_SVM_LDA()
            self.SVM_model()
            self.verboseprint(code = 'SVM_is_fitted')
            self.LDA_Topic_Modeling(self.mean_solving_time) 
            # sorting dev info according to SVM labels.
            # This is crucial to have a consistent result 
            self.developers_info  = self.resort_dict(self.developers_info, self.svm_model.classes_)
            for bug_info in self.BDG.bugs_dictionary.values():
                """Updating the validity of all bugs after having the models ready"""
                bug_info.valid4assignment = bug_info.valid_bug_for_assignment(self.available_developers_email,
                                                                              self.max_acceptable_solving_time,
                                                                              self.bug_evolutionary_db)
                self.bug_info_db.loc[self.bug_info_db.index==bug_info.idx, 'valid'] = bug_info.valid4assignment
                self.predict_LDA(bug_info.idx)
                self.bug_info_db.loc[self.bug_info_db.index==bug_info.idx, 'LDA']   = bug_info.LDA_category
            for bug_info in self.resolved_bugs.values(): 
                """Updating the validity of all RESOLVED bugs after having the models ready"""
                bug_info.valid4assignment = bug_info.valid_bug_for_assignment(self.available_developers_email,
                                                                              self.max_acceptable_solving_time,
                                                                              self.bug_evolutionary_db)                
                self.bug_info_db.loc[self.bug_info_db.index==bug_info.idx, 'valid'] = bug_info.valid4assignment
                self.predict_LDA(bug_info.idx)
                self.bug_info_db.loc[self.bug_info_db.index==bug_info.idx, 'LDA']   = bug_info.LDA_category
            self.running_time["model_preparation_for_testing"] = round(time.time() - self.start_model_updates_for_testing)
            self.verboseprint(code = 'preparation_time')
            self.start_testing_time                            = time.time()                                                            
        elif self.date == self.death:
            for bug_info in self.BDG.bugs_dictionary.values():
                """Updating the validity of all bugs after having the models ready"""
                bug_info.valid4assignment = bug_info.valid_bug_for_assignment(self.available_developers_email,
                                                                              self.max_acceptable_solving_time,
                                                                              self.bug_evolutionary_db)
                self.bug_info_db.loc[self.bug_info_db.index==bug_info.idx, 'valid'] = bug_info.valid4assignment
                self.predict_LDA(bug_info.idx)
                self.bug_info_db.loc[self.bug_info_db.index==bug_info.idx, 'LDA']   = bug_info.LDA_category
            for bug_info in self.resolved_bugs.values(): 
                """Updating the validity of all RESOLVED bugs after having the models ready"""
                bug_info.valid4assignment = bug_info.valid_bug_for_assignment(self.available_developers_email,
                                                                              self.max_acceptable_solving_time,
                                                                              self.bug_evolutionary_db)                
                self.bug_info_db.loc[self.bug_info_db.index==bug_info.idx, 'valid'] = bug_info.valid4assignment
                self.predict_LDA(bug_info.idx)
                self.bug_info_db.loc[self.bug_info_db.index==bug_info.idx, 'LDA']   = bug_info.LDA_category            
            with open(f"dat/{self.project}/list_of_developers.txt", "wb") as fp:   #Pickling
                pickle.dump(list(self.time_to_fix_LDA.index), fp)
            self.time_to_fix_LDA.to_csv(f"dat/{self.project}/time_to_fix_LDA.csv")
            pickle.dump(self.svm_model,          open(f"dat/{self.project}/SVM.sav", 'wb'))
            pickle.dump(self.svm_model_priority, open(f"dat/{self.project}/SVM_priority.sav", 'wb'))
            pickle.dump(self.tfidf_svm,          open(f"dat/{self.project}/Tfidf_vect.pickle", "wb"))
            self.bug_info_db.to_csv(f"dat/{self.project}/bug_info_db_phaseI.csv")

    """"How to print it"""
    def __repr__(self):
        return f'A Wayback Machine for {str(self.project)}; First run to extract LDA and Assignment Time'
    def __str__(self):
        return str(self.__dict__.keys())


"""
Time to Run the first phase
"""
#######################################
##                                   ##
##      author: Hadi Jahanshahi      ##
##     hadi.jahanshahi@ryerson.ca    ##
##          Data Science Lab         ##
##                                   ##
#######################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The parser will handle hyperparameters of the initial Run')

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
    Tfidf_vect           = None

    [bug_evolutionary_db, bug_info_db, list_of_developers, time_to_fix_LDA, 
    SVM_model, feasible_bugs_actual, embeddings, infeasible_dev_emails] = Functions.read_files(project)

    simulation           = Finding_parameters_training(bug_evolutionary_db, bug_info_db, project, 
                                                    infeasible_dev_emails, resolution='actual', verbose=verbose)
    stop_date            = len(pd.date_range(start=simulation.bug_evolutionary_db.time.min().date(), end='31/12/2019'))
    stop_training        = len(pd.date_range(start=simulation.bug_evolutionary_db.time.min().date(), end='31/12/2017'))#training
    stop_testing         = len(pd.date_range(start='01/01/2018', end='31/12/2019')) # whole testing period

    for i in tqdm(range(stop_date), desc="simulating days", position=0, leave=True):
        simulation.waybackmachine()

    """ Keep a backup of the model at the end of the training """
    with open(f'dat/{project}/phaseI.pickle', 'wb') as file:
        pickle.dump(simulation, file) # use `pickle.load` to do the reverse
        
# python3.9 simulator/first_run.py --project=LibreOffice