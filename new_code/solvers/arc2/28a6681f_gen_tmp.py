from __future__ import annotations
import numpy as np
import random

from solvers.arc2 import load_pid
_mod = load_pid('28a6681f')
main = _mod.main


def _build_wall(base_width: int, height: int, direction: str) -> np.ndarray:
    """Build a staircase wall as a local grid.
    direction='left' means constant edge on left, steps widen right toward bottom.
    direction='right' means constant edge on right, steps widen left toward bottom.
    Returns (height, base_width) array with 1s where wall exists."""
    wall = np.zeros((height, base_width), dtype=int)

    # Build widths from bottom (full) to top (narrow)
    # Each level can reduce width by 0 or 1
    widths = [base_width]
    for i in range(1, height):
        reduce = random.choice([0, 1])
        new_w = max(1, widths[-1] - reduce)
        widths.append(new_w)
    widths.reverse()  # now top to bottom

    for row_i, w in enumerate(widths):
        if direction == 'left':
            # Constant left edge, fill from left
            wall[row_i, :w] = 1
        else:
            # Constant right edge, fill from right
            wall[row_i, base_width - w:] = 1

    return wall


def _gen_attempt() -> np.ndarray | None:
    H, W = 10, 10
    grid = np.zeros((H, W), dtype=int)

    # Pick 2 structure colors
    struct_colors = random.sample([2, 3, 4, 5, 6, 7, 8, 9], 2)

    # Base widths for each wall
    left_base_w = random.randint(3, 5)
    right_base_w = random.randint(3, 5)
    combined = left_base_w + right_base_w
    if combined < 6 or combined > 10:
        return None

    # Position on bottom row
    max_start = W - combined
    if max_start < 0:
        return None
    base_start = random.randint(max(0, 0), max_start)

    # Left wall: constant edge on left, steps widen right
    left_height = random.randint(3, min(8, H - 1))
    left_wall = _build_wall(left_base_w, left_height, 'left')

    # Right wall: constant edge on right, steps widen left
    right_height = random.randint(3, min(8, H - 1))
    right_wall = _build_wall(right_base_w, right_height, 'right')

    # Place left wall
    left_col_start = base_start
    for r_off in range(left_height):
        r = H - left_height + r_off
        for c_off in range(left_base_w):
            c = left_col_start + c_off
            if left_wall[r_off, c_off] and 0 <= c < W:
                grid[r, c] = struct_colors[0]

    # Place right wall adjacent
    right_col_start = base_start + left_base_w
    for r_off in range(right_height):
        r = H - right_height + r_off
        for c_off in range(right_base_w):
            c = right_col_start + c_off
            if right_wall[r_off, c_off] and 0 <= c < W:
                grid[r, c] = struct_colors[1]

    # Optionally add guide
    has_guide = random.random() < 0.4
    guide_color = None
    if has_guide:
        remaining = [c for c in [2, 3, 4, 5, 6, 7, 8, 9] if c not in struct_colors]
        guide_color = random.choice(remaining)

        # Find top of structures
        struct_rows = np.where(np.any(grid != 0, axis=1))[0]
        if len(struct_rows) > 0:
            top_struct = struct_rows[0]
            guide_bottom = top_struct - 1
            if guide_bottom >= 4:
                guide_height = random.randint(4, min(5, guide_bottom + 1))
                guide_top = guide_bottom - guide_height + 1
                if guide_top >= 0:
                    # V-funnel
                    if random.random() < 0.5:
                        # Find center of basin
                        basin_center = base_start + left_base_w
                        bottom_c = min(max(basin_center, 2), W - 3)
                        for i in range(guide_height):
                            r = guide_bottom - i
                            cl = bottom_c - i - 1
                            cr = bottom_c + i
                            if 0 <= cl < W and grid[r, cl] == 0:
                                grid[r, cl] = guide_color
                            if 0 <= cr < W and grid[r, cr] == 0 and cr != cl:
                                grid[r, cr] = guide_color
                    else:
                        # Diagonal ramp
                        start_c = random.randint(1, W - 2)
                        direction = random.choice([-1, 1])
                        for i in range(guide_height):
                            r = guide_top + i
                            c = start_c + i * direction
                            if 0 <= c < W and grid[r, c] == 0:
                                grid[r, c] = guide_color

    # Now place water on top of structures (upper area)
    # Find all bg cells that have a non-bg cell directly below
    surfaces = []
    for r in range(H - 1):
        for c in range(W):
            if grid[r, c] == 0 and grid[r + 1, c] != 0:
                surfaces.append((r, c))

    if not surfaces:
        return None

    # Pick water count
    water_count = random.randint(3, min(25, len(surfaces) * 3))

    # Place water as a contiguous block resting on surfaces
    # Start from a surface cell and grow upward
    random.shuffle(surfaces)

    # Try each surface as starting point
    for start_r, start_c in surfaces:
        placed = {(start_r, start_c)}
        frontier = [(start_r, start_c)]

        while len(placed) < water_count and frontier:
            pos = random.choice(frontier)
            r, c = pos

            # Try extending: up, left, right (not down into structures)
            neighbors = []
            for dr, dc in [(-1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in placed and grid[nr, nc] == 0:
                    neighbors.append((nr, nc))

            if neighbors:
                new_pos = random.choice(neighbors)
                placed.add(new_pos)
                frontier.append(new_pos)
            else:
                frontier.remove(pos)

        if len(placed) < 3:
            continue

        # Verify support: each water pixel must be supported by structure or water below
        # Sort bottom-up for checking
        placed_sorted = sorted(placed, key=lambda rc: -rc[0])
        valid = True
        for r, c in placed_sorted:
            if r == H - 1:
                continue
            below = grid[r + 1, c]
            if below == 0 and (r + 1, c) not in placed:
                valid = False
                break

        if not valid:
            continue

        # Place water on grid
        input_grid = grid.copy()
        for r, c in placed:
            input_grid[r, c] = 1

        # Run solver and verify
        output = main(input_grid)

        # Check: water count preserved, structures unchanged
        if np.sum(output == 1) != len(placed):
            continue
        struct_mask = grid != 0
        if not np.array_equal(output[struct_mask], grid[struct_mask]):
            continue

        # Check water actually moved (some pixels changed position)
        water_in = set(zip(*np.where(input_grid == 1)))
        water_out = set(zip(*np.where(output == 1)))
        if water_in == water_out:
            # Water didn't move -- not interesting, but still valid
            # Prefer cases where water moves
            continue

        # Check output water is in basin area (lower portion)
        out_water_rows = [r for r, c in water_out]
        if min(out_water_rows) < 2:
            # Water still in upper area -- not great
            pass

        return input_grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 300 attempts")
