# Description of the project

The project is a implementation of the A* algorithm to solve the 15-puzzle game.

# 15-puzzle game description

The 15-puzzle game is a game where the player has to move the numbers from 1 to 15 to the goal state. The goal state is a 4x4 grid with the numbers from 1 to 15 in a specific order. The player can move the numbers in the grid by sliding them to an empty space. The empty space is represented by the type `None`.

# A* algorithm description

The A* algorithm is a search algorithm that finds the shortest path from the start state to the goal state. The algorithm uses a heuristic function to estimate the cost of the path from the start state to the goal state. The heuristic function is the sum of the cost of the path from the start state to the current state and the estimated cost of the path from the current state to the goal state. The algorithm uses a priority queue to store the states to be explored. The algorithm explores the states in the priority queue, starting with the state with the lowest cost. The algorithm stops when the goal state is reached.

# Implementation

The implementation is a Python implementation of the A* algorithm. The implementation uses the `heapq` module to implement the priority queue. The implementation uses the `math` module to implement the heuristic function. The implementation uses the `copy` module to implement the deepcopy function. The implementation uses the `time` module to measure the time of the algorithm.


**Key concepts to consider:**
- A* Algorithm core components
- 15-puzzle game mechanics
- State representation
- Heuristic function
- Priority queue management

1. **State Representation**
   - Create a class to represent the puzzle board (4x4 grid)
   - Implement methods to manipulate the board state
   - Add functionality to check if a state is the goal state

2. **Move Generation**
   - Implement logic to find valid moves (up, down, left, right)
   - Create functions to generate successor states
   - Ensure moves are only made to adjacent empty spaces

3. **Heuristic Function**
   - Implement Manhattan distance calculation
   - Calculate misplaced tiles
   - Combine heuristics for better estimation

4. **A* Algorithm Core**
   - Implement priority queue using heapq
   - Create Node class to track:
     - Current state
     - Parent state
     - g-cost (path cost from start)
     - h-cost (heuristic estimate)
     - f-cost (g + h)

5. **Search Implementation**
   - Maintain open and closed sets
   - Implement state exploration logic
   - Track path to solution
   - Handle edge cases and invalid states

