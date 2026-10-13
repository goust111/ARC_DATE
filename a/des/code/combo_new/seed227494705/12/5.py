import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror source-room stamps through maroon divider holes into adjacent rooms.

    Input structure: grey background split by maroon horizontal and vertical
    partition lines into rectangular rooms; source rooms contain a colored stamp
    and one red marker, and maroon dividers may contain single grey transfer holes.
    Rule: in source-room reading order, use the red marker's direction from the
    stamp bbox center to choose a neighboring divider; if that divider has a
    hole, mirror every stamp pixel across the divider into the adjacent room.
    Key insight: direction marker, transfer hole, and mirror axis are independent
    visible cues that must all agree before copying.
    """
    out = grid.copy()
    bg, wall, red = 5, 9, 2
    row_lines = [r for r in range(grid.shape[0]) if np.sum(grid[r] == wall) >= grid.shape[1] // 2]
    col_lines = [c for c in range(grid.shape[1]) if np.sum(grid[:, c] == wall) >= grid.shape[0] // 2]
    rooms = []
    for i, (r1, r2) in enumerate(zip(row_lines, row_lines[1:])):
        for j, (c1, c2) in enumerate(zip(col_lines, col_lines[1:])):
            rooms.append((r1 + 1, r2 - 1, c1 + 1, c2 - 1))
    for r1, r2, c1, c2 in rooms:
        sub = grid[r1:r2 + 1, c1:c2 + 1]
        stamps = [(r1 + r, c1 + c) for r, c in np.argwhere((sub != bg) & (sub != wall) & (sub != red))]
        marks = [(r1 + r, c1 + c) for r, c in np.argwhere(sub == red)]
        if not stamps or not marks:
            continue
        rs = [p[0] for p in stamps]
        cs = [p[1] for p in stamps]
        center = ((min(rs) + max(rs)) / 2, (min(cs) + max(cs)) / 2)
        mr, mc = marks[0]
        if abs(mr - center[0]) >= abs(mc - center[1]):
            direction = "down" if mr > center[0] else "up"
        else:
            direction = "right" if mc > center[1] else "left"
        if direction == "right":
            wall_c = c2 + 1
            holes = [(r, wall_c) for r in range(r1, r2 + 1) if grid[r, wall_c] == bg]
            for r, c in stamps:
                if holes:
                    out[r, wall_c + (wall_c - c)] = grid[r, c]
        elif direction == "left":
            wall_c = c1 - 1
            holes = [(r, wall_c) for r in range(r1, r2 + 1) if grid[r, wall_c] == bg]
            for r, c in stamps:
                if holes:
                    out[r, wall_c - (c - wall_c)] = grid[r, c]
        elif direction == "down":
            wall_r = r2 + 1
            holes = [(wall_r, c) for c in range(c1, c2 + 1) if grid[wall_r, c] == bg]
            for r, c in stamps:
                if holes:
                    out[wall_r + (wall_r - r), c] = grid[r, c]
        else:
            wall_r = r1 - 1
            holes = [(wall_r, c) for c in range(c1, c2 + 1) if grid[wall_r, c] == bg]
            for r, c in stamps:
                if holes:
                    out[wall_r - (r - wall_r), c] = grid[r, c]
    return out


def generate_input() -> np.ndarray:
    rows, cols = random.choice([(2, 2), (2, 3)])
    rh, rw = 6, 6
    h, w = rows * rh + rows + 1, cols * rw + cols + 1
    grid = np.full((h, w), 5, dtype=int)
    for r in range(0, h, rh + 1):
        grid[r, :] = 9
    for c in range(0, w, rw + 1):
        grid[:, c] = 9
    rooms = [(1 + i * (rh + 1), 1 + j * (rw + 1)) for i in range(rows) for j in range(cols)]
    random.shuffle(rooms)
    for idx, (r, c) in enumerate(rooms[:random.randint(1, min(3, len(rooms)))]):
        color = random.choice([1, 3, 4, 6, 7, 8])
        stamp = [(1, 1), (1, 2), (2, 1), (3, 1)]
        for a, b in stamp:
            grid[r + a, c + b] = color
        possible = []
        i, j = (r - 1) // (rh + 1), (c - 1) // (rw + 1)
        if j < cols - 1:
            possible.append("right")
        if j > 0:
            possible.append("left")
        if i < rows - 1:
            possible.append("down")
        if i > 0:
            possible.append("up")
        direction = random.choice(possible)
        if direction == "right":
            grid[r + 2, c + rw] = 5
            grid[r + 3, c + 4] = 2
        elif direction == "left":
            grid[r + 2, c - 1] = 5
            grid[r + 3, c + 0] = 2
        elif direction == "down":
            grid[r + rh, c + 2] = 5
            grid[r + 4, c + 3] = 2
        else:
            grid[r - 1, c + 2] = 5
            grid[r + 0, c + 3] = 2
    return grid
