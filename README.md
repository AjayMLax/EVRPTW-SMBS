# EVRPTW-SMBS
Electric Vehicle Routing Problem with Time Windows : Pyomo-Gurobi optimization and heuristic solutions
## Table Of Contents
- [Description](#description)
- [Data](#data)
  - [Dataset](#dataset)
  - [Parsing and storing the data](#parsing-and-storing-the-data)
- [Methodology](#methodology)
  - [Greedy Algorithm](#greedy-algorithm)
  - [Optimization Model](#optimization-model)
- [Results](#results)
- [Conclusion](#conclusion)
  - [Potential Improvements](#potential-improvements)

---

## Description

The data for this project is taken from [Goeke & Schneider](https://data.mendeley.com/datasets/h3mrm5dhxw/1), which contains 148 instance-solution pairs for varying customer sizes (5 to 100). For this project, 6 representative instance-solution pairs were chosen. You can find them in the [Dataset folder](Dataset) along with the provided [data description](Dataset/DataDescription.txt). The full dataset is publicly available via the link above. 

The goal of the project is to build an optimization model in Python using the Pyomo library to solve the Electric Vehicle Routing Problem with Time Windows and Station-based Mobile Battery Swapping (EVRPTW-SMBS). The obtained solutions are then compared with the provided benchmark solutions.

## Data
### Dataset

The original instance data can be viewed [here](Dataset/Instances). Each instance contains a table of values with the following fields:

  - Node ID: 0 corresponds to the depot; all others are customers.
  - Coordinates (x, y): spatial positions of the depot and customers.
  - Demand: amount of load required by each customer.
  - Time window: earliest and latest service start time allowed.
  - Service time: duration required to serve the customer.

In addition, each instance file specifies:

  - Fuel tank capacity
  - Vehicle load capacity
  - Consumption rate (set to 1 in all instances)
  - Velocity (set to 1 in all instances)
  - Swap service time

The benchmark solution provided for the chosen 6 instances can be viewed [here](Solutions/BenchmarkSolutions). It includes:

  - Optimal cost
  - Total distance traveled
  - Total distance travelled by ECVs and BSVs
  - Total number of vehicles used
  - Total number of ECVs and BSVs
  - Total number of swaps
  - Optimal routes for ECVs and BSVs

All of this information is stored in text file format.

### Parsing and storing the data

In order to make the dataset usable within Python, the original text files were first parsed and structured into tabular form. For each instance, the main table was converted into a .csv file. The additional parameters (capacities, velocities, swap service times, etc.) were extracted into a separate .csv file.

To ensure consistency and facilitate querying, all parsed data was first stored in a MySQL database. Before modeling, the relevant instance data was exported back into CSV format for direct use in Python. The complete set of processed files can be viewed [here](Dataset/ParsedData), which contains the CSV representations for all six chosen instances.

## Methodology

The optimization model was implemented using Pyomo and solved with Gurobi. The code containing the greedy algorithm and the optimization model used for the instance R104-5 can be viewed [here](R104-5.py). The same code is modified for other instances by importing the appropriate .csv files. 

### Greedy Algorithm

A simple greedy construction heuristic was implemented to generate feasible solutions. It was constructed as follows:

  - The algorithm iteratively assigns customers to the nearest available vehicle within capacity and time-window constraints.
  - When an electric vehicle reaches its fuel limit, a swap is requested and performed before continuing.
  - The greedy solution only uses one BSV for all swaps.

This heuristic is not designed for optimality but instead serves as a baseline for evaluating the optimization model. The greedy solutions for all 6 instances can be viewed [here](Solutions/GreedySolutions).

### Optimization Model

 A full mathematical formulation with detailed descriptions of all sets, parameters, variables, and constraints is provided in the document [EVRPTW-SMBS Optimization model](EVRPTW_SMBS_Optimization_model.pdf). This provides a direct link between the mathematical specification and the Pyomo implementation. The Results of the optimization model code for each instance can be viewed [here](Solutions/ModelSolutions).

## Results

The comparison between the benchmark, greedy, and Pyomo-Gurobi solutions can be explored in the [interactive dashboard](SolutionsComparisonReport.pbix). Greedy solutions are reported only when they are feasible.

The results show that the distance gap is `-1.03%`, meaning the optimization model produced routes that are, on average, slightly shorter than those in the benchmark solutions. In contrast, the cost gap is `+18.93%`. This discrepancy arises because the coefficients used in the benchmark's cost function are not disclosed in the dataset. For this project, the cost coefficients were chosen so that the generated routes and fleet sizes align reasonably with the benchmark, but the resulting cost values are not directly comparable on an absolute scale.

The dashboard also illustrates that greedy solutions consistently require more vehicles (ECVs and BSVs) and yield higher total distances and costs than either the Pyomo-Gurobi solutions or the benchmark. This highlights that, while the greedy algorithm can generate feasible solutions quickly, it systematically sacrifices solution quality and fails to achieve optimality.

These findings reinforce the importance of exact optimization models for routing problems, while also highlighting opportunities to improve computational efficiency.

## Conclusion

This project demonstrates that exact optimization using Pyomo-Gurobi can generate solutions comparable to benchmark results, even when cost coefficients differ. The greedy heuristic provides baseline feasibility checks but consistently underperforms compared to the optimization model.

### Potential Improvements

  - Scalability: Improve solver performance for larger instances using heuristics and advanced arc pruning.
  - Improve Bounds: Use results from greedy algorithm to have better upper bounds for total number of ECVs and BSVs used.
  - Hybrid Methods: Combine greedy construction with local search or metaheuristic.
  - Data Pipeline: Automate parsing and database integration to allow faster switching between instances.
  - Visualization: Extend the Power BI dashboard with route maps for better interpretability of results.
  - Realism: Incorporate stochastic elements such as uncertain demand or travel times.
