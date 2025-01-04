from typing import List, Optional, Tuple, Dict, Set
import copy
import heapq
import time
import logging

# Set up logging configuration at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

class Board:
    def __init__(self, state: Optional[List[List[Optional[int]]]] = None):
        # Initialize with either provided state or create empty 4x4 board
        if state:
            self.state = copy.deepcopy(state)
        else:
            self.state = [[None] * 4 for _ in range(4)]
        
        # Cache the empty tile position for efficient moves
        self.empty_pos = self._find_empty()
    
    def _find_empty(self) -> Tuple[int, int]:
        """Find the position of the empty tile"""
        for i in range(4):
            for j in range(4):
                if self.state[i][j] is None:
                    return (i, j)
        raise ValueError("Invalid board: No empty tile found")
    
    def is_goal_state(self) -> bool:
        """Check if current state is the goal state"""
        goal = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, None]]
        return self.state == goal
    
    def get_state(self) -> List[List[Optional[int]]]:
        """Return current board state"""
        return copy.deepcopy(self.state)
    
    def __str__(self) -> str:
        """String representation of the board"""
        return '\n'.join([' '.join(str(tile) if tile is not None else '_' 
                                 for tile in row) for row in self.state])
    
    def __eq__(self, other: 'Board') -> bool:
        """Check if two board states are equal"""
        if not isinstance(other, Board):
            return False
        return self.state == other.state
    
    def __hash__(self) -> int:
        """Hash function for board state (needed for sets/dicts)"""
        return hash(str(self.state))
    
    def get_valid_moves(self) -> List[str]:
        """Return list of valid moves from current state"""
        valid_moves = []
        row, col = self.empty_pos
        
        # Check all possible moves
        if row > 0:
            valid_moves.append('up')
        if row < 3:
            valid_moves.append('down')
        if col > 0:
            valid_moves.append('left')
        if col < 3:
            valid_moves.append('right')
            
        return valid_moves
    
    def move(self, direction: str) -> 'Board':
        """
        Make a move in the given direction and return new board state
        Raises ValueError for invalid moves
        """
        if direction not in self.get_valid_moves():
            raise ValueError(f"Invalid move: {direction}")
        
        # Get current empty position
        row, col = self.empty_pos
        
        # Calculate new position based on direction
        new_row, new_col = row, col
        if direction == 'up':
            new_row -= 1
        elif direction == 'down':
            new_row += 1
        elif direction == 'left':
            new_col -= 1
        elif direction == 'right':
            new_col += 1
            
        # Create new board state
        new_state = copy.deepcopy(self.state)
        # Swap empty tile with the tile in the move direction
        new_state[row][col], new_state[new_row][new_col] = \
            new_state[new_row][new_col], new_state[row][col]
            
        return Board(new_state)
    
    def get_successors(self) -> List['Board']:
        """Generate all possible successor states"""
        return [self.move(move) for move in self.get_valid_moves()]
    
    def manhattan_distance(self) -> int:
        """
        Calculate Manhattan distance heuristic
        Sum of distances each tile is from its goal position
        """
        distance = 0
        goal_positions = {
            1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (0, 3),
            5: (1, 0), 6: (1, 1), 7: (1, 2), 8: (1, 3),
            9: (2, 0), 10: (2, 1), 11: (2, 2), 12: (2, 3),
            13: (3, 0), 14: (3, 1), 15: (3, 2)
        }
        
        for i in range(4):
            for j in range(4):
                tile = self.state[i][j]
                if tile is not None:  # Skip empty tile
                    goal_row, goal_col = goal_positions[tile]
                    distance += abs(i - goal_row) + abs(j - goal_col)
        
        return distance
    
    def misplaced_tiles(self) -> int:
        """
        Calculate number of misplaced tiles heuristic
        Count tiles not in their goal position
        """
        count = 0
        goal = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, None]]
                
        for i in range(4):
            for j in range(4):
                if self.state[i][j] != goal[i][j] and self.state[i][j] is not None:
                    count += 1
        
        return count
    
    def linear_conflicts(self) -> int:
        """
        Calculate linear conflicts heuristic
        Adds a penalty when two tiles are in the same row/column but in wrong order
        """
        conflicts = 0
        goal_positions = {
            1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (0, 3),
            5: (1, 0), 6: (1, 1), 7: (1, 2), 8: (1, 3),
            9: (2, 0), 10: (2, 1), 11: (2, 2), 12: (2, 3),
            13: (3, 0), 14: (3, 1), 15: (3, 2)
        }
        
        # Check rows
        for i in range(4):
            for j in range(4):
                tile1 = self.state[i][j]
                if tile1 is not None:
                    goal_row1, _ = goal_positions[tile1]
                    if goal_row1 == i:  # tile1 is in correct row
                        for k in range(j + 1, 4):
                            tile2 = self.state[i][k]
                            if tile2 is not None:
                                goal_row2, _ = goal_positions[tile2]
                                if goal_row2 == i and tile1 > tile2:
                                    conflicts += 2
        
        return conflicts
    
    def calculate_heuristic(self, method: str = 'manhattan') -> int:
        """
        Calculate heuristic value using specified method
        Supports 'manhattan', 'misplaced', or 'combined' methods
        """
        if method == 'manhattan':
            return self.manhattan_distance()
        elif method == 'misplaced':
            return self.misplaced_tiles()
        elif method == 'combined':
            # Combine manhattan distance with linear conflicts
            return self.manhattan_distance() + self.linear_conflicts()
        else:
            raise ValueError(f"Unknown heuristic method: {method}") 

class Node:
    def __init__(self, board: Board, parent: Optional['Node'] = None, 
                 g_cost: int = 0, move: str = ''):
        self.board = board
        self.parent = parent
        self.g_cost = g_cost
        # Modify heuristic calculation to be more aggressive
        self.h_cost = board.calculate_heuristic('combined') * 5  # Weight the heuristic more
        self.f_cost = self.g_cost + self.h_cost
        self.move = move
    
    def __lt__(self, other: 'Node') -> bool:
        # Tiebreaking: prefer nodes with lower h_cost when f_costs are equal
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost
    
    def __eq__(self, other: 'Node') -> bool:
        return isinstance(other, Node) and self.board == other.board
    
    def get_path(self) -> List[str]:
        """Reconstruct the path from start to this node"""
        path = []
        current = self
        while current.parent:
            path.append(current.move)
            current = current.parent
        return path[::-1]
    
    def get_path_cost(self) -> int:
        """Return the cost of the path to this node"""
        return self.g_cost

class PuzzleVisualizer:
    def __init__(self, output_file: str = "puzzle_solution.txt"):
        self.output_file = output_file
        self.moves_history = []
    
    def add_state(self, board: Board, move: str = "", step: int = 0) -> None:
        """Record a board state with its move and step number"""
        state_str = (
            f"\nStep {step}: {move}\n"
            f"{'=' * 17}\n"
            f"{self._format_board(board)}\n"
            f"{'=' * 17}"
        )
        self.moves_history.append(state_str)
    
    def _format_board(self, board: Board) -> str:
        """Format board state for visualization"""
        rows = []
        for row in board.state:
            row_str = " ".join(f"{tile:2}" if tile is not None else " _" for tile in row)
            rows.append("|" + row_str + "|")
        return "\n".join(rows)
    
    def save_solution(self) -> None:
        """Save the complete solution to a file"""
        with open(self.output_file, "w") as f:
            f.write("\n".join(self.moves_history))
            f.write(f"\nTotal moves: {len(self.moves_history)-1}\n")  # -1 for initial state

def solve_puzzle(initial_board: Board, solver_context: Optional[dict] = None) -> Tuple[List[str], int, int]:
    """
    Solve the 15-puzzle using A* algorithm
    Returns: (move_sequence, nodes_explored, path_cost)
    """
    logging.info("Starting puzzle solution...")
    start_time = time.time()
    
    # Initialize visualizer
    visualizer = PuzzleVisualizer()
    visualizer.add_state(initial_board, "Initial state", 0)
    
    start_node = Node(initial_board)
    open_set: List[Node] = [start_node]
    closed_set: Set[Board] = set()
    came_from: Dict[Board, Node] = {}
    g_score: Dict[Board, int] = {initial_board: 0}
    
    nodes_explored = 0
    last_log_time = time.time()
    log_interval = 5
    
    while open_set:
        current = heapq.heappop(open_set)
        nodes_explored += 1
        
        # Update solver statistics if context is provided
        if solver_context:
            solver_context['update_stats'](open_set, current)
        
        # Periodic logging
        current_time = time.time()
        if current_time - last_log_time >= log_interval:
            logging.info(
                f"Exploring node {nodes_explored}: "
                f"f_cost={current.f_cost}, "
                f"g_cost={current.g_cost}, "
                f"h_cost={current.h_cost}, "
                f"frontier_size={len(open_set)}, "
                f"closed_set_size={len(closed_set)}"
            )
            last_log_time = current_time
        
        if current.board.is_goal_state():
            elapsed_time = time.time() - start_time
            logging.info(
                f"Solution found! "
                f"Nodes explored: {nodes_explored}, "
                f"Time taken: {elapsed_time:.2f}s, "
                f"Path cost: {current.g_cost}"
            )
            
            # Reconstruct and visualize the solution path
            moves = []
            node = current
            step = 1
            path_states = []
            while node.parent:
                path_states.append((node.board, node.move))
                moves.append(node.move)
                node = node.parent
            
            # Add states in correct order
            for board, move in reversed(path_states):
                visualizer.add_state(board, move, step)
                step += 1
            
            # Save the complete solution
            visualizer.save_solution()
            return moves[::-1], nodes_explored, g_score[current.board]
        
        closed_set.add(current.board)
        
        # Log every 10000 nodes explored
        if nodes_explored % 10000 == 0:
            logging.info(
                f"Explored {nodes_explored} nodes. "
                f"Current f_cost: {current.f_cost}, "
                f"Frontier size: {len(open_set)}"
            )
        
        # Early stopping if the frontier gets too large
        if len(open_set) > 50000:
            logging.warning("Frontier size exceeded limit. Puzzle might be too complex.")
            raise ValueError("Search space too large")
            
        # Modify successor handling to be more selective
        successor_count = 0
        best_successor = None
        best_h_cost = float('inf')
        
        for move in current.board.get_valid_moves():
            successor_board = current.board.move(move)
            if successor_board in closed_set:
                continue
                
            tentative_g = g_score[current.board] + 1
            successor_node = Node(successor_board, current, tentative_g, move)
            
            # Keep track of the best successor
            if successor_node.h_cost < best_h_cost:
                best_h_cost = successor_node.h_cost
                best_successor = successor_node
            
            if (successor_board not in g_score or 
                tentative_g < g_score[successor_board]):
                came_from[successor_board] = current
                g_score[successor_board] = tentative_g
                
                if not any(node.board == successor_board for node in open_set):
                    heapq.heappush(open_set, successor_node)
                    successor_count += 1
        
        # If we're not making progress, prioritize the best successor
        if current.h_cost <= best_h_cost and best_successor:
            heapq.heappush(open_set, best_successor)
        
        # Log when many successors are generated
        if successor_count > 2:
            logging.debug(
                f"Generated {successor_count} successors for node {nodes_explored}"
            )
    
    logging.warning(
        f"No solution found after exploring {nodes_explored} nodes. "
        f"Time elapsed: {time.time() - start_time:.2f}s"
    )
    raise ValueError("No solution found")

class PuzzleSolver:
    def __init__(self, heuristic_method: str = 'combined'):
        self.heuristic_method = heuristic_method
        self.nodes_explored = 0
        self.max_search_depth = 0
        self.max_frontier_size = 0
        self.last_log_time = time.time()
        self.log_interval = 5  # Log every 5 seconds
        self.start_time = None
    
    def _update_statistics(self, open_set: List[Node], current_node: Node) -> None:
        """Update solver statistics"""
        # Update frontier size
        self.max_frontier_size = max(self.max_frontier_size, len(open_set))
        
        # Update search depth (path length to current node)
        current_depth = 0
        node = current_node
        while node.parent:
            current_depth += 1
            node = node.parent
        self.max_search_depth = max(self.max_search_depth, current_depth)
    
    def solve(self, initial_board: Board) -> dict:
        """Solve the puzzle and return detailed statistics"""
        self.start_time = time.time()
        
        try:
            # Create solver context to pass to solve_puzzle
            solver_context = {
                'solver': self,
                'update_stats': self._update_statistics
            }
            moves, nodes, cost = solve_puzzle(initial_board, solver_context)
            success = True
        except ValueError as e:
            moves, nodes, cost = [], self.nodes_explored, 0
            success = False
            
        end_time = time.time()
        
        return {
            'success': success,
            'solution': moves,
            'nodes_explored': nodes,
            'path_cost': cost,
            'max_search_depth': self.max_search_depth,
            'max_frontier_size': self.max_frontier_size,
            'runtime': end_time - self.start_time
        }
    
    @staticmethod
    def is_solvable(board: Board) -> bool:
        """
        Check if the given board state is solvable
        Uses the inversion count method
        """
        # Flatten the board into a 1D list, removing None
        flat_board = []
        empty_row = 0
        for i, row in enumerate(board.state):
            for tile in row:
                if tile is not None:
                    flat_board.append(tile)
                else:
                    empty_row = i
        
        # Count inversions
        inversions = 0
        for i in range(len(flat_board)):
            for j in range(i + 1, len(flat_board)):
                if flat_board[i] > flat_board[j]:
                    inversions += 1
        
        # For a 4x4 puzzle:
        # If width is odd, puzzle is solvable if inversions is even
        # If width is even, puzzle is solvable if:
        #   (blank on even row from bottom + inversions odd) or
        #   (blank on odd row from bottom + inversions even)
        board_width = 4
        if board_width % 2 == 0:
            if (board_width - empty_row) % 2 == 0:
                return inversions % 2 == 1
            else:
                return inversions % 2 == 0
        else:
            return inversions % 2 == 0
