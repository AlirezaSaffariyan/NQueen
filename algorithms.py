import random


class NQueenAlgorithms:
    """Class to handle n-queens algorithms.

    Implemented algorithms:
        - Hill Climbing
        - Genetic
        - CSP
        - CSP with MRV
    """

    def __init__(
        self,
        size,
        queens=None,
        max_iterations=1000,
        population_size=100,
        mutation_rate=0.1,
    ):
        self.size = size
        self.queens = queens if queens is not None else self.generate_random_state()
        self.max_iterations = max_iterations  # Max iterations for hill climbing
        self.population_size = population_size  # Population size for genetic algorithm
        self.mutation_rate = mutation_rate  # Mutation rate for genetic algorithm

    def csp_with_mrv_with_steps(self):
        """Solve the N-Queens problem using CSP with MRV heuristic, showing steps."""
        solution = [None] * self.size  # Initialize with None values
        steps = []

        # Initialize domains for each column
        domains = {col: set(range(self.size)) for col in range(self.size)}

        # Calculate middle positions
        mid_col = self.size // 2
        mid_row = self.size // 2

        # Place the first queen in the middle
        solution[mid_col] = mid_row
        steps.append(solution[:])  # Add initial state

        # Update domains after placing the first queen
        self.update_domains(domains, mid_col, mid_row)

        # Generate column sequence starting from middle
        cols_sequence = self.generate_column_sequence(mid_col)

        if self.solve_csp_mrv_with_steps(1, solution, domains, steps, cols_sequence):
            return steps
        return None

    def update_queen_positions(self):
        """Update the chessboard to display queens."""
        # Clear the board first
        self.clear_board()

        # Update the positions where queens are present
        for col, row in enumerate(self.queens):
            if (
                row is not None and 0 <= row < self.size
            ):  # Check if the row is valid and not None
                button_index = row * self.size + col
                if (
                    0 <= button_index < len(self.buttons)
                ):  # Check if the button index is valid
                    self.buttons[button_index].setText("♛")

    def create_state_representation(self, solution):
        """Create a list representation of the current state with 0s for empty squares."""
        return [0 if x is None else x + 1 for x in solution]

    def generate_column_sequence(self, mid):
        """Generate sequence of columns starting from middle going outwards."""
        sequence = []
        left = mid - 1
        right = mid + 1

        # Alternate between left and right sides
        while left >= 0 or right < self.size:
            if left >= 0:
                sequence.append(left)
            if right < self.size:
                sequence.append(right)
            left -= 1
            right += 1

        return sequence

    def create_clean_solution(self, solution, cols_sequence):
        """Create a solution array that maintains the proper column ordering."""
        clean_solution = [None] * self.size
        for i, col in enumerate(cols_sequence):
            if solution[col] is not None:
                clean_solution[col] = solution[col]
        return [pos for pos in clean_solution if pos is not None]

    def is_safe_mrv(self, row, col, solution):
        """Check if it's safe to place a queen at the given position with partial solution."""
        for c in range(self.size):
            if solution[c] is not None:  # Check only placed queens
                if solution[c] == row or abs(solution[c] - row) == abs(c - col):
                    return False
        return True

    def update_domains(self, domains, col, row):
        """Update domains after placing a queen."""
        # Remove the used row from all unassigned columns
        for other_col in range(self.size):
            if other_col != col:  # Skip the current column
                # Remove the same row
                domains[other_col].discard(row)

                # Remove diagonal positions
                diagonal_up = row + abs(other_col - col)
                diagonal_down = row - abs(other_col - col)

                if diagonal_up < self.size:
                    domains[other_col].discard(diagonal_up)
                if diagonal_down >= 0:
                    domains[other_col].discard(diagonal_down)

    def get_domain_size(self, col, solution):
        """Get the size of the domain for a given column."""
        if not solution:
            return self.size

        valid_positions = set(range(self.size))
        for c, r in enumerate(solution):
            # Remove rows that are already taken
            valid_positions.discard(r)

            # Remove diagonal positions
            diagonal_up = r + (col - c)
            diagonal_down = r - (col - c)

            if diagonal_up < self.size:
                valid_positions.discard(diagonal_up)
            if diagonal_down >= 0:
                valid_positions.discard(diagonal_down)

        return len(valid_positions)

    # New CSP MRV based method
    def csp_with_mrv(self):
        """Solve the N-Queens problem using CSP with Minimum Remaining Value (MRV) heuristic."""
        solution = []
        if self.solve_csp_mrv(0, solution):
            return solution
        return None  # Return None if no solution found

    def solve_csp_mrv(self, col, solution):
        """Recursive function to solve the N-Queens problem using CSP with MRV."""
        if col >= self.size:
            return True  # All queens are placed

        # Get the row options for this column based on MRV
        row_options = self.get_row_options(solution, col)

        # Sort options by MRV (fewest remaining values)
        row_options.sort(key=lambda x: len(self.get_row_options(solution, col, x)))

        for row in row_options:
            if self.is_safe(row, col, solution):
                solution.append(row)  # Place queen
                if self.solve_csp_mrv(col + 1, solution):
                    return True  # Recur to place next queen
                solution.pop()  # Backtrack if no solution found

        return False  # No valid placement found

    def get_row_options(self, solution, col, row=None):
        """Get available rows for a given column considering the current solution."""
        if row is None:  # If no specific row is given, return all valid rows
            return [r for r in range(self.size) if self.is_safe(r, col, solution)]
        else:  # Return the row options for a specific row
            return [
                r for r in range(self.size) if self.is_safe(r, col, solution + [row])
            ]

    # new

    def csp(self):
        """Solve the N-Queens problem using a backtracking approach."""
        solution = []
        if self.solve_csp(0, solution):
            return solution
        return None  # Return None if no solution found

    def csp_with_steps(self):
        """Solve the N-Queens problem using a backtracking approach with steps."""
        solution = []
        steps = []
        if self.solve_csp(0, solution, steps):
            return steps
        return None  # Return None if no solution found

    def solve_csp(self, col, solution, steps):
        """Recursive function to solve the N-Queens problem using backtracking."""
        if col >= self.size:
            steps.append(solution[:])  # Add the complete solution
            return True

        for row in range(self.size):
            if col == 0:
                row = random.randint(0, 7)
            if self.is_safe(row, col, solution):
                solution.append(row)
                steps.append(solution[:])  # Record the current state
                if self.solve_csp(col + 1, solution, steps):
                    return True
                solution.pop()

        return False

    def is_safe(self, row, col, solution):
        """Check if it's safe to place a queen at (row, col)."""
        for c in range(col):
            if solution[c] == row or abs(solution[c] - row) == abs(c - col):
                return False
        return True

    def hill_climbing(self):
        """Perform hill climbing algorithm to solve the n-queens problem."""
        current_state = self.queens
        current_attacks = self.heuristic(current_state)

        neighbor_states = self.get_neighbors(current_state)

        # Find the best neighbor state
        best_state = min(neighbor_states, key=self.heuristic)
        best_attacks = self.heuristic(best_state)

        while best_attacks < current_attacks:
            current_state = best_state
            current_attacks = best_attacks

            neighbor_states = self.get_neighbors(current_state)

            # Find the best neighbor state
            best_state = min(neighbor_states, key=self.heuristic)
            best_attacks = self.heuristic(best_state)

        return current_state

    def hill_climbing_with_steps(self):
        """Perform hill climbing algorithm with steps to solve the n-queens problem."""
        states = []
        current_state = self.queens
        current_attacks = self.heuristic(current_state)

        neighbor_states = self.get_neighbors(current_state)

        # Find the best neighbor state
        best_state = min(neighbor_states, key=self.heuristic)
        best_attacks = self.heuristic(best_state)

        while best_attacks < current_attacks:
            current_state = best_state
            current_attacks = best_attacks

            neighbor_states = self.get_neighbors(current_state)

            # Find the best neighbor state
            best_state = min(neighbor_states, key=self.heuristic)
            best_attacks = self.heuristic(best_state)

            states.append(current_state)
        return states

    def get_neighbors(self, state):
        """Generate all neighbor states by moving each queen to different rows."""
        neighbors = []
        for col in range(self.size):
            for row in range(self.size):
                if row != state[col]:  # Avoid moving to the same row
                    new_state = state[:]
                    new_state[col] = row  # Move queen to new row
                    neighbors.append(new_state)

        return neighbors

    def genetic(self):
        """Perform genetic algorithm to solve the n-queens problem."""
        population = self.initialize_population()

        for iteration in range(self.max_iterations):
            # Sort the population by fitness (ascending)
            population.sort(key=self.heuristic)
            new_population = population[:2]  # Keep the two best solutions

            # Create new individuals using crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = random.choices(
                    population[:20], k=2
                )  # Select the top 20 for crossover
                child = self.crossover(parent1, parent2)

                if random.random() < self.mutation_rate:
                    child = self.mutate(child)

                new_population.append(child)

            population = new_population

            # Check if a solution is found
            if any(self.heuristic(ind) == 0 for ind in population):
                return next(
                    (ind for ind in population if self.heuristic(ind) == 0), None
                )

        return min(
            [ind for ind in population]
        )  # No solution found after max iterations

    def genetic_with_steps(self):
        """Perform genetic algorithm with steps to solve the n-queens problem."""
        states = []
        population = self.initialize_population()

        for iteration in range(self.max_iterations):
            # Sort the population by fitness (ascending)
            population.sort(key=self.heuristic)
            new_population = population[:2]  # Keep the two best solutions

            # Create new individuals using crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = random.choices(
                    population[:20], k=2
                )  # Select the top 20 for crossover
                child = self.crossover(parent1, parent2)

                if random.random() < self.mutation_rate:
                    child = self.mutate(child)

                new_population.append(child)

            population = new_population

            # Check if a solution is found
            if any(self.heuristic(ind) == 0 for ind in population):
                states.append(
                    next((ind for ind in population if self.heuristic(ind) == 0), None)
                )
                return states

        return states

    def initialize_population(self):
        """Generate a population of random solutions."""
        return [self.generate_random_state() for _ in range(self.population_size)]

    def generate_random_state(self):
        """Generate a random state for the N-Queens problem."""
        return [random.randint(0, self.size - 1) for _ in range(self.size)]

    def crossover(self, parent1, parent2):
        """Create a child by combining two parents."""
        crossover_point = random.randint(1, self.size - 2)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return self.ensure_no_duplicate_queens(child)

    def mutate(self, state):
        """Mutate a state by randomly changing the position of one queen."""
        col = random.randint(0, self.size - 1)
        row = random.randint(0, self.size - 1)
        mutated_state = state[:]
        mutated_state[col] = row
        return self.ensure_no_duplicate_queens(mutated_state)

    def ensure_no_duplicate_queens(self, state):
        """Ensure that the generated state has distinct rows for each queen."""
        seen = set()
        for col in range(self.size):
            row = state[col]
            while row in seen:  # If the row is already taken, find a new one
                row = random.randint(0, self.size - 1)
            seen.add(row)
            state[col] = row
        return state

    def heuristic(self, state):
        """Evaluate the state by counting the number of attacking pairs of queens."""
        attacks = 0
        for i in range(self.size):
            # Skip if no queen in this column
            if state[i] is None:
                continue
            for j in range(i + 1, self.size):
                # Skip if no queen in this column
                if state[j] is None:
                    continue
                if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                    attacks += 1
        return attacks

    def solve_csp_mrv_with_steps(
        self, queens_placed, solution, domains, steps, cols_sequence
    ):
        """Recursive function to solve N-Queens using CSP with MRV, recording steps."""
        if queens_placed >= self.size:
            steps.append(solution[:])  # Add final solution
            return True

        # Get next column to fill based on pre-computed sequence
        col = cols_sequence[
            queens_placed - 1
        ]  # -1 because first queen is already placed

        # Try each possible row in the domain of this column
        for row in sorted(domains[col]):
            if self.is_safe_mrv(row, col, solution):
                # Place the queen
                solution[col] = row
                steps.append(solution[:])  # Record the step

                # Save current domains
                old_domains = {k: v.copy() for k, v in domains.items()}

                # Update domains for remaining columns
                self.update_domains(domains, col, row)

                # Recur with next queen
                if self.solve_csp_mrv_with_steps(
                    queens_placed + 1, solution, domains, steps, cols_sequence
                ):
                    return True

                # Backtrack
                solution[col] = None
                steps.append(solution[:])  # Record backtracking
                domains.update(old_domains)  # Restore domains

        return False

    def get_attacking_positions(self, state):
        """Get all positions under attack by queens."""
        attacking_positions = set()

        for col1 in range(self.size):
            row1 = state[col1]
            if row1 is None:  # Skip if no queen in this column
                continue

            # Check horizontal and diagonal attacks
            for col2 in range(self.size):
                if col1 == col2:
                    continue

                row2 = state[col2]
                if row2 is None:  # Skip if no queen in this column
                    continue

                # Check if queens are attacking each other
                if row1 == row2 or abs(row1 - row2) == abs(  # Same row
                    col1 - col2
                ):  # Diagonal
                    attacking_positions.add((row1, col1))
                    attacking_positions.add((row2, col2))

        return attacking_positions
