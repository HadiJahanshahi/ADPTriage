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
- **dat**: This folder contains the data related to three software projects - `EclipseJDT`, `GCC`, and `Mozilla`. These projects have been used to test the model in the package. Each project has its own subfolder within the `dat` folder. In addition to these project folders, there is also a `ToyExample` folder, which contains a small example dataset that can be used to test the model.
- **utils**: This includes three files:
   * `release_dates.py`: This file contains a list of release dates for the EclipseJDT, GCC, and Mozilla projects. These release dates are used to simulate a real-world scenario where we do not have access to future release dates and evaluate the performance of our model based on historical data.
   * `prerequisites.py`: This file imports all the necessary packages required to run the program. It is used to ensure that all the required packages are installed and available in the user's environment before running the program.

   * `functions.py`: This file contains some manual written packages that are used in the program. As an example, one of these functions is undirected_network, which is used to convert a directed graph to an undirected graph. 


## Prerequisite

Here are the prerequisites needed to run this package using Python 3.7+:

* numpy (version 1.19.5 or later)
* pandas (version 1.2.4 or later)
* matplotlib (version 3.4.2 or later)
* networkx (version 2.5.1 or later)
* plotly (version 5.1.0 or later)
* gensim (version 4.0.1 or later)
* gurobipy (version 9.1.2 or later)
* nltk (version 3.6.2 or later)
* scipy (version 1.6.3 or later)
* sklearn (version 0.24.2 or later)
* tensorflow (version 2.5.0 or later)
* tqdm (version 4.60.0 or later)


## Running the project

To run the project, there are three phases, each requiring a different python file to be executed. The first phase is handled by `first_run.py`, the second by `second_run.py`, and the third by `third_run.py`. Additionally, the project has three different datasets available: `EclipseJDT`, `GCC`, and Mozilla`.

Before running the code, the datasets should be unzipped. 

`first_run.py` performs the training and testing of the model and extracts LDA categories, assignment time, and SVM models. It accepts the following arguments:

* `--project`: Specifies the name of the project to run. It accepts a string value from the list: `[LibreOffice, Mozilla, EclipseJDT, GCC]`. Default value is `EclipseJDT`.
* `--verbose`: Specifies the verbosity of the output. It accepts integer values from 0 to 2 or string values of `[nothing, some, all]`. The default value is 0.

An example of running `first_run.py` for EclipseJDT project is as follows:

```terminal
python3.9 simulator/first_run.py --project=EclipseJDT
```

The second phase can be run using `second_run.py`. This phase estimates the state transitions by running the model for the training phase once. The arguments for this phase are similar to the first phase, as follows:

* `--project`: It specifies the dataset to use, and it can be selected from the list of available datasets: `[LibreOffice, Mozilla, EclipseJDT, GCC]`.
* `--verbose`: It determines the level of verbosity of the output. It can be either: `[0, 1, 2, nothing, some, all]`.

Here's an example of how to run the second phase for the EclipseJDT dataset:
```terminal
python3.9 simulator/second_run.py --project=EclipseJDT
```

The third phase can be run using `third_run.py`, which implements the Approximate Dynamic Programming (ADP) algorithm. This phase has many arguments that you can set to customize the algorithm's behavior. Here are the arguments for this phase:

* `--project`: It specifies the dataset to use, and it can be selected from the list of available datasets: `[LibreOffice, Mozilla, EclipseJDT, GCC]`.
* `--verbose`: It determines the level of verbosity of the output. It can be either: `[0, 1, 2, nothing, some, all]`.
* `--epochs_training`: It specifies the number of epochs for the training phase.
* `--epochs_testing`: It specifies the number of epochs for the testing phase.
* `--project_horizon`: It specifies the number of days for each epoch.
* `--method`: It indicates which approach we want to use.
* `--alpha_update`: It indicates how to update alpha in the ADP algorithm for this set `[constant, harmonic, bakf]`.
* `--alpha`: It is the alpha value for the constant case.
* `--early_asg_cost`: It specifies whether to impose a cost for early assignments.
* `--gamma`: It is the discounting coefficient of the Bellman equation.
* `--epsilon`: It is the ratio of exploration (e-greedy approach). 1 means no exploration.
* `--cost_f`: It specifies how the cost function of the late assignment should be.
* `--idea`: It specifies which idea to use: (a) Sophisticated, (b) Simplified.
* `--toy_example`: It specifies whether it is a toy example.

Here's an example of how to run the third phase for the EclipseJDT dataset:

```terminal
python3.9 simulator/third_run.py --project=EclipseJDT --epochs_training=20000 --epochs_testing=100 --method=ADP --alpha_update=Constant --early_asg_cost=False --gamma=0.9 --epsilon=0.25 --cost_f=exponential
```

In this example, the ADP algorithm is used with constant alpha, no cost for early assignments, a discounting coefficient of 0.9, an exploration ratio of 0.25, and an exponential cost function for late assignments. The training phase will run for 20000 epochs, and the testing phase will run for 100 epochs. The dataset used is EclipseJDT.


