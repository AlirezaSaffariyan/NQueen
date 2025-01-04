# N-Queens Problem Solver

## Overview

This project provides a comprehensive solution to the classic N-Queens problem, a well-known problem in the field of computer science and chess. The N-Queens problem is a constraint satisfaction problem where the goal is to place N queens on an NxN chessboard such that no two queens attack each other.

## Features

- **Multiple Algorithms**: The project implements several algorithms to solve the N-Queens problem, including:

  - Hill Climbing
  - Genetic Algorithm
  - Constraint Satisfaction Problem (CSP)
  - CSP with Minimum Remaining Value (MRV) heuristic

- **Visualization**: The project includes a graphical user interface (GUI) built using PyQt5 to visualize the solution process.

- **Step-by-Step Solution**: The project provides the ability to visualize the solution process step-by-step for each algorithm.

- **Customizable Board Size**: The project allows users to customize the size of the chessboard.

- **Random Queen Placement**: The project includes a feature to randomly place queens on the board.

## Requirements

- Python 3.x: The project requires Python 3.x to run.
- PyQt5: The project requires PyQt5 for the GUI.
- Other Dependencies: The project uses several standard libraries, including random, sys, and time.

## Installation

1. Clone the repository using `git clone https://github.com/AlirezaSaffariyan/NQueen.git`
2. Install the required dependencies using `pip install -r requirements.txt`
3. Run the project using `python n_queens.py`

## Usage

1. Run the project using python n_queens.py
2. Select the desired algorithm from the dropdown menu.
3. Customize the board size using the spin box.
4. Click the "Random Queens" button to randomly place queens on the board.
5. Click the "Solve Puzzle" button to start the solution process.
6. The project will visualize the solution process step-by-step.

## Algorithms

### Hill Climbing

- The Hill Climbing algorithm starts with a random initial state and iteratively applies a series of small changes to the state.
- The algorithm evaluates the fitness of each new state and moves to the state with the highest fitness.
- The algorithm repeats this process until a solution is found or a maximum number of iterations is reached.

### Genetic Algorithm

- The Genetic Algorithm is a population-based algorithm that uses principles of natural selection and genetics to search for a solution.
- The algorithm starts with an initial population of random states and iteratively applies selection, crossover, and mutation operators to the population.
- The algorithm evaluates the fitness of each state in the population and selects the fittest states to reproduce.
- The algorithm repeats this process until a solution is found or a maximum number of generations is reached.

### Constraint Satisfaction Problem (CSP)

- The CSP algorithm is a constraint-based algorithm that uses a backtracking search to find a solution.
- The algorithm starts with an empty state and iteratively adds queens to the state, ensuring that each queen does not attack any other queen.
- The algorithm uses a recursive function to explore the search space and backtracks when a dead end is reached.

### CSP with Minimum Remaining Value (MRV) heuristic

- The CSP with MRV heuristic is a variant of the CSP algorithm that uses the MRV heuristic to select the next queen to place.
- The MRV heuristic selects the queen with the fewest remaining values, which reduces the search space and improves the efficiency of the algorithm.

## License

This project is licensed under the MIT License.
