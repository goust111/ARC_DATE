
import numpy as np
from typing import *
import time
import random
from arc_utils import Color, draw_line, paste, random_free_location, random_sprite

# concepts:
# path finding


# description:
# In the input you will see teal pixels and a short green line and a short red line.
# Find a path starting from the green line and ending at the red line and color that path green, with the following constraints:
# You can't go through a teal pixel; you can only change direction when you hit a teal pixel; you have to start in the direction of the green line.

def main(input_grid):
    """
    Find and color a constrained path from the green start line to the red end line.

    Input structure:
    - A black grid contains teal barrier pixels.
    - A short green horizontal or vertical line gives the starting location and initial travel direction.
    - A short red line marks the destination.

    Rule:
    - Run breadth-first search over position plus direction states.
    - Move straight through black cells and turn only when a teal pixel blocks the next step.
    - Stop when the next cell in the current direction is red, then color the recovered path green.

    Key insight:
    - Teal pixels are not passable walls; they are turn triggers that define where the path may change direction.
    """
    # Plan:
    # 1. Find the start and end points of the pathfinding problem
    # 2. Define the state space, initial state(s), successor function, and goal test
    # 3. Run bfs to find the shortest path from start to end
    # 4. Color the path green

    # 1. Parse the input, based on color
    # There is the start object, end object, and barriers object
    background = Color.BLACK
    start_object = input_grid.copy()
    start_object[start_object != Color.GREEN] = background
    end_object = input_grid.copy()
    end_object[end_object != Color.RED] = background
    barriers_object = input_grid.copy()
    barriers_object[barriers_object != Color.TEAL] = background

    # Determine the orientation of the start object
    r_coordinates = {r for r, c in np.argwhere(start_object == Color.GREEN)}
    c_coordinates = {c for r, c in np.argwhere(start_object == Color.GREEN)}
    # vertical line?
    if len(r_coordinates) == 1:
        possible_orientations = [(0, 1), (0, -1)]
    # horizontal line?
    elif len(c_coordinates) == 1:
        possible_orientations = [(1, 0), (-1, 0)]
    else:
        assert False, "Start object is not horizontal/vertical"
    
    # 2. Define the state space, initial state(s), successor function, and goal test
    # A state is a tuple of (x, y, orientation)
    # orientation is a tuple of (dx, dy)
        
    # Initially we begin at a point on the line, along the orientation of the line
    initial_states = [(r, c, orientation)
                      for r, c in np.argwhere(start_object == Color.GREEN)
                      for orientation in possible_orientations]
    

    def successors(state):
        r, c, orientation = state
        dr, dc = orientation

        if not (0 <= r + dr < input_grid.shape[0] and 0 <= c + dc < input_grid.shape[1]):
            return

        if barriers_object[r + dr, c + dc] == background:
            yield (r + dr, c + dc, orientation)
        if barriers_object[r + dr, c + dc] != background:
            # right angle turns
            new_orientations = [(dc, dr), (-dc, -dr)]
            for new_orientation in new_orientations:
                yield (r, c, new_orientation)
    
    def is_goal(state):
        r, c, (dr, dc) = state
        if not (0 <= r + dr < end_object.shape[0] and 0 <= c + dc < end_object.shape[1]):
            return False
        return end_object[r + dr, c + dc] == Color.RED
    
    # 3. Run bfs to find the shortest path from start to end
    queue = list(initial_states)
    visited = set(initial_states)
    parent = {}
    while queue:
        state = queue.pop(0)        
        if is_goal(state):
            break        
        for successor in successors(state):
            if successor not in visited:
                visited.add(successor)
                parent[successor] = state
                queue.append(successor)

    assert is_goal(state), "No path found"
    
    path = []
    while state in parent:
        path.append(state)
        state = parent[state]

    # 4. Color the path green
    # draw on top of the input grid
    output_grid = input_grid.copy()
    for r, c, _ in path:
        output_grid[r, c] = Color.GREEN

    return output_grid

def generate_input():
    # We want to first generate a successful path and add some noise teal pixels.
    # Finally, we will remove the intermediate pixels in the path.
    # Because the problem never uses the color 42, we will draw the path in that color, but finally erase it before returning the grid.

    # Initialize grid
    n = random.randint(14, 20)
    grid = np.zeros((n, n), dtype=int)

    # Start sprite: a 1x2 green line (we rotate the final grid for variety).
    start_sprite = np.full((1, 2), Color.GREEN, dtype=int)

    # Draw start sprite
    start_r, start_c = random_free_location(grid, start_sprite, transparent=Color.BLACK, border=4, padding=0, padding_connectivity=8)
    paste(grid, start_sprite, start_r, start_c, transparent=Color.BLACK, clip=True)

    # Make a random path from the start to the end, leaving color 42 along the path, and leaving teal pixels at each turn
    r,c = start_r, start_c-1
    orientation = (0, -1)
    target_length = random.randint(10, 20)
    for _ in range(target_length):
        # Draw the path
        if grid[r, c] != Color.BLACK:
            return generate_input()
        grid[r, c] = 42        

        if random.random() < 0.2:
            # right angle turn
            dr, dc = orientation
            new_orientation = random.choice([(dc, dr), (-dc, -dr)])
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= n or cc < 0 or cc >= n:
                return generate_input()
            if grid[rr, cc] != Color.BLACK:
                return generate_input()
            grid[rr, cc] = Color.TEAL
            orientation = new_orientation
        
        dr, dc = orientation
        r += dr
        c += dc

        if r < 0 or r >= n or c < 0 or c >= n: return generate_input()
    
    # Color the ending red
    grid[r, c] = Color.RED
    grid[r-dr, c-dc] = Color.RED

    # randomly sprinkle teal in unoccupied locations
    for r, c in np.argwhere(grid == Color.BLACK):
        if random.random() < 0.3:
            grid[r, c] = Color.TEAL

    # Replace the path with black
    grid[grid == 42] = Color.BLACK

    # Randomly rotate
    grid = np.rot90(grid, k=-(random.randint(0, 3)))

    return grid
