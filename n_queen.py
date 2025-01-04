import random
import sys
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QLabel,
    QHBoxLayout,
)

from algorithms import NQueenAlgorithms

# List of available algorithms
ALGORITHMS = ["Select an algorithm", "Hill Climbing", "Genetic", "CSP", "CSP with MRV"]


class ThreatMonitor(QThread):
    """Thread to monitor threats on the board."""

    update_threats = pyqtSignal(int, set)  # Signal to emit threat count and positions

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.speed = 400  # Default speed
        self.queens = []
        self.size = 8

    def run(self):
        while self.running:
            if self.queens:
                algorithm = NQueenAlgorithms(self.size, self.queens)
                threats = algorithm.heuristic(self.queens)
                attacking_positions = algorithm.get_attacking_positions(self.queens)
                self.update_threats.emit(threats, attacking_positions)
            self.msleep(self.speed)

    def update_speed(self, speed):
        """Update the monitoring speed."""
        self.speed = speed

    def update_state(self, queens, size):
        """Update the current board state."""
        self.queens = queens
        self.size = size


# Modified NQueensUI class
class NQueensUI(QWidget):
    def __init__(self, size: int):
        super().__init__()

        self.setWindowTitle("N-Queens Problem")
        self.setMinimumSize(800, 600)
        self.size = size
        self.board = []
        self.queens = []
        self.buttons = []
        self.selected_queen = None

        # Main horizontal layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Left panel (sidebar) styling
        left_panel = QWidget()
        left_panel.setFixedWidth(280)  # Slightly wider for better appearance
        # In the left_panel.setStyleSheet section, update the QSpinBox styling:
        left_panel.setStyleSheet(
            """
            QWidget {
                background-color: #2C3E50;
                color: white;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 12px;
                margin: 8px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #2574A9;
            }
            QLabel {
                font-weight: bold;
                margin-top: 15px;
                margin-bottom: 5px;
                color: #ECF0F1;
                font-size: 14px;
            }
            QComboBox, QSpinBox {
                padding: 8px;
                margin: 5px;
                border: 2px solid #3498DB;
                border-radius: 6px;
                background-color: #34495E;
                color: white;
                min-height: 30px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
                background-color: #3498DB;
                #border-radius: 2px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #2980B9;
            }
            QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
            }
            QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
            }
            QSpinBox::up-arrow {
                background-color: transparent;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-bottom: 6px solid white;
                width: 0px;
                height: 0px;
            }
            QSpinBox::down-arrow {
                background-color: transparent;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 6px solid white;
                width: 0px;
                height: 0px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                background-color: transparent;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 6px solid white;
                width: 0px;
                height: 0px;
            }
        """
        )

        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        # Title styling
        title_label = QLabel("N-Queens Solver")
        title_label.setStyleSheet(
            """
            font-size: 20px;
            padding-top: 25px;
            padding-bottom: 40px;
            padding-right: auto;
            padding-left: auto;
            color: #ECF0F1;
            font-weight: bold;
            border-bottom: 2px solid #3498DB;
            margin-bottom: 15px;
        """
        )
        left_layout.addWidget(title_label)

        # Settings sections with improved spacing
        algo_label = QLabel("Select Algorithm:")
        left_layout.addWidget(algo_label)
        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItems(ALGORITHMS)
        left_layout.addWidget(self.algorithm_selector)

        size_label = QLabel("Board Size:")
        left_layout.addWidget(size_label)
        self.size_selector = QSpinBox()
        self.size_selector.setMinimum(4)
        self.size_selector.setMaximum(20)
        self.size_selector.setValue(self.size)
        self.size_selector.valueChanged.connect(self.update_board_size)
        left_layout.addWidget(self.size_selector)

        speed_label = QLabel("Animation Speed:")
        left_layout.addWidget(speed_label)
        self.speed_selector = QSpinBox()
        self.speed_selector.setMinimum(100)
        self.speed_selector.setMaximum(2000)
        self.speed_selector.setValue(400)
        self.speed_selector.setSingleStep(100)
        self.speed_selector.setSuffix(" ms")
        left_layout.addWidget(self.speed_selector)

        # Action buttons with icons or symbols
        self.random_queens_button = QPushButton("🎲 Random Queens")
        self.random_queens_button.clicked.connect(self.place_random_queens)
        left_layout.addWidget(self.random_queens_button)

        self.solve_button = QPushButton("✨ Solve Puzzle")
        self.solve_button.clicked.connect(self.solve_automatically)
        left_layout.addWidget(self.solve_button)

        left_layout.addStretch()

        # Add threat counter to left panel
        self.threat_label = QLabel("Threats: 0")
        self.threat_label.setStyleSheet(
            """
            font-size: 18px;
            padding: 10px;
            margin: 10px;
            background-color: #34495E;
            border-radius: 5px;
            color: #E74C3C;
        """
        )
        left_layout.addWidget(self.threat_label)

        # Initialize threat monitor thread
        self.threat_monitor = ThreatMonitor()
        self.threat_monitor.update_threats.connect(self.update_threat_display)
        self.threat_monitor.start()

        # Right panel (chessboard) styling
        right_panel = QWidget()
        right_panel.setStyleSheet(
            """
            QWidget {
                background-color: #ECF0F1;
            }
        """
        )
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Chessboard container with border
        chessboard_container = QWidget()
        chessboard_container.setStyleSheet(
            """
            QWidget {
                background-color: #34495E;
                border-radius: 10px;
                padding: 20px;
            }
        """
        )
        self.chessboard_frame = QGridLayout()
        chessboard_container.setLayout(self.chessboard_frame)
        right_layout.addWidget(chessboard_container)
        right_layout.addStretch()

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        self.create_board()
        self.draw_board()

        # Connect speed selector to update thread
        self.speed_selector.valueChanged.connect(self.speed_selector_changed)

    def create_board(self):
        """Create and initialize the chessboard."""
        self.board = [[None for _ in range(self.size)] for _ in range(self.size)]

    def draw_board(self):
        """Draw the chessboard with improved styling."""
        self.buttons = []
        button_size = min(500 // self.size, 60)  # Limit maximum button size

        for i in range(self.size):
            for j in range(self.size):
                button = QPushButton("")
                button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                button.setFixedSize(button_size, button_size)

                # Improved chess square styling
                if (i + j) % 2 == 0:
                    style = """
                        background-color: #FFFFFF;
                        border: 1px solid #34495E;
                    """
                else:
                    style = """
                        background-color: #3498DB;
                        border: 1px solid #34495E;
                    """

                # Add hover effect and queen styling
                button.setStyleSheet(
                    f"""
                    QPushButton {{
                        {style}
                        font-size: {button_size * 0.6}px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        border: 2px solid #17089B;
                    }}
                """
                )

                button.clicked.connect(
                    lambda checked, row=i, col=j: self.button_clicked(row, col)
                )
                self.chessboard_frame.addWidget(button, i, j)
                self.buttons.append(button)

    def update_board_size(self):
        """Update the size of the chessboard based on the spin box value."""
        self.size = self.size_selector.value()
        self.create_board()

        # Clear all existing widgets in the chessboard frame
        for i in reversed(range(self.chessboard_frame.count())):
            self.chessboard_frame.itemAt(i).widget().setParent(None)

        self.draw_board()

        # Update threat monitor with new size
        self.threat_monitor.update_state(self.queens, self.size)

    def place_random_queens(self):
        """Place queens randomly in separate columns."""
        self.queens = [random.randint(0, self.size - 1) for _ in range(self.size)]
        self.update_queen_positions()

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

        # Update threat monitor with new state
        self.threat_monitor.update_state(self.queens, self.size)

    def clear_board(self):
        """Clear the chessboard of any queens."""
        for button in self.buttons:
            button.setText("")  # Remove any text from buttons
            button.setStyleSheet(
                button.styleSheet().replace("border: 2px solid rgb(255, 0, 0);", "")
            )  # Remove highlight

        self.selected_queen = None  # Reset the selected queen

    def button_clicked(self, row, col):
        """Handle button clicks on the chessboard."""
        # Check if there is a queen in the clicked position
        if self.queens[col] == row:  # There is a queen in this column
            if self.selected_queen is None:
                # Select the queen
                self.selected_queen = col
                self.highlight_queen(row, col)
            else:
                if self.selected_queen == col:
                    # Deselect the queen if clicking on the same one
                    self.clear_highlight(row, col)
                    self.selected_queen = None
                else:
                    # Try to move the queen to the new position
                    if self.selected_queen == col and self.queens[col] != row:
                        self.move_queen(self.selected_queen, row)
        else:
            # Clicking on an empty square
            if self.selected_queen is not None and self.selected_queen == col:
                # Try to move to this square
                self.move_queen(self.selected_queen, row)

    def clear_highlight(self, row, col):
        """Clear the highlight from the selected queen."""
        self.buttons[row * self.size + col].setStyleSheet(
            self.buttons[row * self.size + col]
            .styleSheet()
            .replace("border: 2px solid rgb(255, 0, 0);", "")
        )

    def move_queen(self, col, new_row):
        """Move the queen to a new row in the same column."""
        if 0 <= col < len(self.queens) and 0 <= new_row < self.size:  # Add validation
            self.queens[col] = new_row
            self.update_queen_positions()

    def highlight_queen(self, row, col):
        """Highlight the selected queen."""
        self.buttons[row * self.size + col].setStyleSheet(
            self.buttons[row * self.size + col].styleSheet()
            + "border: 2px solid rgb(255, 0, 0);"
        )

    def solve_automatically(self):
        """Solve the N-Queens problem automatically using the selected algorithm."""
        selected_algorithm = self.algorithm_selector.currentText()
        algorithm = NQueenAlgorithms(self.size, self.queens)

        if selected_algorithm == "Select an algorithm":
            mbox = QMessageBox()
            mbox.setIcon(QMessageBox.Warning)
            mbox.setWindowTitle("Select Algorithm")
            mbox.setText("Please select a valid algorithm.")
            mbox.addButton(QMessageBox.Ok)
            mbox.exec_()
            return

        # Get solution states based on selected algorithm
        states = None
        if selected_algorithm == "Hill Climbing":
            states = algorithm.hill_climbing_with_steps()
        elif selected_algorithm == "Genetic":
            states = algorithm.genetic_with_steps()
        elif selected_algorithm == "CSP":
            states = algorithm.csp_with_steps()
        elif selected_algorithm == "CSP with MRV":
            states = algorithm.csp_with_mrv_with_steps()

        # Check if a solution was found
        if states and any(state is not None for state in states):
            self.visualize_solution([state for state in states if state is not None])
        else:
            mbox = QMessageBox()
            mbox.setIcon(QMessageBox.Information)
            mbox.setWindowTitle("No Solution")
            mbox.setText("No solution was found for the selected algorithm.")
            mbox.addButton(QMessageBox.Ok)
            mbox.exec_()

    def visualize_solution(self, states):
        """Visualize the solution steps."""
        self.current_step = 0
        self.total_steps = len(states)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_board_with_step)
        self.timer.start(
            self.speed_selector.value()
        )  # Update every second (adjust as necessary)

        self.states = states  # Store the states for visualization

    def update_board_with_step(self):
        """Update the chessboard to reflect the current step in the solution."""
        if self.current_step < self.total_steps:
            self.queens = self.states[self.current_step]
            self.update_queen_positions()  # Update the UI with the current state
            self.current_step += 1
        else:
            self.timer.stop()  # Stop the timer when done
            self.show_completion_message()  # Show the completion message

    def show_completion_message(self):
        """Show a message when the visualization is completed."""
        selected_algorithm = self.algorithm_selector.currentText()
        mbox = QMessageBox()
        mbox.setIcon(QMessageBox.Information)
        mbox.setWindowTitle("Result")
        mbox.setText(f"Most efficient solution found using {selected_algorithm}!")
        mbox.addButton(QMessageBox.Ok)
        mbox.exec_()

    def update_threat_display(self, threat_count, attacking_positions):
        """Update the threat counter and highlight threatening positions."""
        self.threat_label.setText(f"Threats: {threat_count}")

        # First restore all squares to their original colors
        for i in range(self.size):
            for j in range(self.size):
                button = self.buttons[i * self.size + j]
                if (i + j) % 2 == 0:
                    color = "#FFFFFF"
                else:
                    color = "#3498DB"

                # Keep existing style but update background color
                current_style = button.styleSheet()
                new_style = self.update_background_color(current_style, color)
                button.setStyleSheet(new_style)

        # Highlight attacking positions
        for row, col in attacking_positions:
            if 0 <= row < self.size and 0 <= col < self.size:
                button = self.buttons[row * self.size + col]
                current_style = button.styleSheet()
                new_style = self.update_background_color(current_style, "#E74C3C")
                button.setStyleSheet(new_style)

    def update_background_color(self, style, color):
        """Update background-color in stylesheet while preserving other properties."""
        if not style.strip():
            style = "QPushButton {}"  # Set a fallback default style

        style_parts = style.split("}")
        updated_parts = []
        for part in style_parts:
            if "background-color:" in part:
                updated_part = (
                    part.split("background-color:")[0] + f"background-color: {color};"
                )
            else:
                updated_part = part
            if updated_part.strip():
                updated_parts.append(updated_part + "}")

        updated_style = "".join(updated_parts)

        return updated_style

    def closeEvent(self, event):
        """Clean up thread before closing."""
        self.threat_monitor.running = False
        self.threat_monitor.wait()
        super().closeEvent(event)

    def speed_selector_changed(self):
        """Update threat monitor speed when speed selector changes."""
        self.threat_monitor.update_speed(self.speed_selector.value())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(15)
    app.setFont(font)

    # Initialize with a default size for the chessboard
    n_queens_ui = NQueensUI(size=8)
    n_queens_ui.show()

    sys.exit(app.exec_())
