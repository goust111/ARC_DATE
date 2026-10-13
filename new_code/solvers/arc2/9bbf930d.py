from __future__ import annotations
import numpy as np

def _get_corridor_color(grid: np.ndarray, r: int, c: int, direction: str) -> int | None:
    """
    Get corridor color if position is in a corridor for given direction.

    A corridor is the space between two parallel bars of the same color.
    """
    H, W = grid.shape
    rr = int(r)
    cc = int(c)
    if rr < 0 or rr >= H or cc < 0 or cc >= W:
        return None

    # Particle + background are hard-coded for this puzzle family.
    PARTICLE = 6
    BG = 7

    if direction in {"left", "right"}:  # moving horizontally -> need bars above/below
        if rr == 0 or rr == H - 1:
            return None
        above = int(grid[rr - 1, cc])
        below = int(grid[rr + 1, cc])
        if above == below and above not in {PARTICLE, BG}:
            return int(above)
    else:  # moving vertically -> need bars left/right
        if cc == 0 or cc == W - 1:
            return None
        left = int(grid[rr, cc - 1])
        right = int(grid[rr, cc + 1])
        if left == right and left not in {PARTICLE, BG}:
            return int(left)
    return None


def _trace_particle(grid: np.ndarray, start_r: int, start_c: int, initial_dir: str) -> list[tuple[int, int]]:
    """
    Trace a particle's path with inertia.

    The particle:
    - Moves in current direction through empty cells and corridors
    - Turns 90 degrees when blocked and a corridor is available
    - Continues with inertia even when exiting a corridor
    - Stops at grid edge or when completely blocked
    """
    H, W = grid.shape
    r = int(start_r)
    c = int(start_c)
    direction = str(initial_dir)
    path = [(r, c)]

    DIR_DELTA = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
    TURN_OPTIONS = {
        "right": ["up", "down"],
        "left": ["up", "down"],
        "up": ["left", "right"],
        "down": ["left", "right"],
    }

    PARTICLE = 6
    BG = 7

    in_corridor = False

    for _ in range(300):
        dr, dc = DIR_DELTA[direction]
        nr = int(r + dr)
        nc = int(c + dc)

        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break

        next_cell = int(grid[nr, nc])

        if next_cell not in {PARTICLE, BG}:
            # Blocked by colored cell - try to turn into a corridor.
            turned = False
            for turn_dir in TURN_OPTIONS[direction]:
                tdr, tdc = DIR_DELTA[turn_dir]
                tr = int(r + tdr)
                tc = int(c + tdc)
                if 0 <= tr < H and 0 <= tc < W:
                    if int(grid[tr, tc]) in {PARTICLE, BG} and _get_corridor_color(grid, tr, tc, turn_dir) is not None:
                        r, c, direction = tr, tc, turn_dir
                        in_corridor = True
                        path.append((r, c))
                        turned = True
                        break
            if not turned:
                break
            continue

        # Can potentially move forward.
        next_corridor = _get_corridor_color(grid, nr, nc, direction)

        if in_corridor and next_corridor is None:
            # Exiting corridor - check if we can turn into another corridor.
            turned = False
            for turn_dir in TURN_OPTIONS[direction]:
                tdr, tdc = DIR_DELTA[turn_dir]
                tr = int(r + tdr)
                tc = int(c + tdc)
                if 0 <= tr < H and 0 <= tc < W:
                    if int(grid[tr, tc]) in {PARTICLE, BG} and _get_corridor_color(grid, tr, tc, turn_dir) is not None:
                        r, c, direction = tr, tc, turn_dir
                        in_corridor = True
                        path.append((r, c))
                        turned = True
                        break
            if not turned:
                # Continue with inertia.
                r, c = nr, nc
                in_corridor = False
                path.append((r, c))
        else:
            r, c = nr, nc
            if next_corridor is not None:
                in_corridor = True
            path.append((r, c))

    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move magenta particles along corridors formed by parallel colored bars.

    Input structure:
    - bg = 7 (orange). Particles = 6 (magenta) in the leftmost column.
    - Colored bars form parallel pairs defining corridors between them.

    Rule:
    - A particle enters a corridor whose bars start near the left edge at its row.
    - Moves right with inertia: continues until blocked, then turns 90° into an available corridor.
    - At corridor exit, prefers turning into another corridor; otherwise continues straight.
    - Stops at grid edge or when fully blocked; output shows final position only.

    Key insight: Inertia — the particle maintains direction through open space, only turning when forced by obstacles or corridor junctions.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    BG = 7
    PARTICLE = 6

    # Find particles that can move.
    particles_to_move: list[int] = []
    for r in range(H):
        if int(g[r, 0]) != int(PARTICLE):
            continue

        # Find the first corridor cell in this row.
        first_corridor_col = None
        corridor_color = None
        for c in range(W):
            if int(g[r, c]) == int(BG) and 0 < r < H - 1:
                above = int(g[r - 1, c])
                below = int(g[r + 1, c])
                if above == below and above not in {int(PARTICLE), int(BG)}:
                    first_corridor_col = int(c)
                    corridor_color = int(above)
                    break

        if first_corridor_col is None or corridor_color is None:
            continue

        # Check if corridor bars start at the same column near left edge.
        bar_above_start = W
        bar_below_start = W
        for c in range(W):
            if int(g[r - 1, c]) == int(corridor_color):
                bar_above_start = int(c)
                break
        for c in range(W):
            if int(g[r + 1, c]) == int(corridor_color):
                bar_below_start = int(c)
                break

        MAX_BAR_START = 3
        if bar_above_start == bar_below_start and bar_above_start <= int(MAX_BAR_START):
            particles_to_move.append(int(r))

    # Move each particle.
    for start_r in particles_to_move:
        path = _trace_particle(g, int(start_r), 0, "right")
        if len(path) > 1:
            out[int(start_r), 0] = int(BG)
            end_r, end_c = path[-1]
            out[int(end_r), int(end_c)] = int(PARTICLE)

    return out

