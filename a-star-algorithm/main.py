from a_star_algorithm import Board, PuzzleSolver

def create_random_board() -> Board:
    """Create a random solvable board state"""
    import random
    
    # Create solved state
    numbers = list(range(1, 16)) + [None]
    
    while True:
        # Shuffle numbers
        random.shuffle(numbers)
        
        # Create board state
        state = [numbers[i:i+4] for i in range(0, 16, 4)]
        board = Board(state)
        
        # Check if solvable
        if PuzzleSolver.is_solvable(board):
            return board

# Create a puzzle instance
# initial_board = create_random_board()

initial_board_matrix = [
    [6, 13, 7 ,10],
    [8, 9, 11, None],
    [15, 2 , 12, 5],
    [14, 3, 1, 4]
]

initial_board = Board(initial_board_matrix)

print(initial_board)
print("Is solvable: ", PuzzleSolver.is_solvable(Board(initial_board_matrix)))

# Create solver and solve puzzle
solver = PuzzleSolver(heuristic_method='combined')
result = solver.solve(initial_board)

# Print results
if result['success']:
    print(f"Solution found in {result['runtime']:.2f} seconds")
    print(f"Moves: {result['solution']}")
    print(f"Path cost: {result['path_cost']}")
    print(f"Nodes explored: {result['nodes_explored']}")
    print(f"Max search depth: {result['max_search_depth']}")
    print(f"Max frontier size: {result['max_frontier_size']}")
else:
    print("No solution found")