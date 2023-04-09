# ADPTriage
ADPTriage: Approximate Dynamic Programming for Bug Triage


This is a Python implementation of the paper named ADPTriage: Approximate Dynamic Programming for Bug Triage.

## Folder Structure
- **BDG**: This folder contains files related to the bug dependency graph.
- **components**: This folder includes bug assignee class, bug class, decision class, and state class. They are used for the ADP.
- **simulator**: This folder includes three files:
    * `first_run.py`: This file includes the Finding_parameters_training class, runs the model for the training + testing phase once and extracts LDA categories, assignment time, and SVM model.
    * `second_run.py`: This file includes the estimating_state_transitions class, runs the model for the training phase once, and extracts transition between different states.
    * `third_run.py`: This file includes the ADP class for the ADP algorithm with many inputs.
