from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg, in_bounds
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the edge-touching green start bar through the corridor, turning at each teal blocker until the path leaves the grid.

    Input structure:
    - The background is the most common color.
    - Exactly two foreground colors are present: a 2-pixel 4-connected start bar and many blocker pixels.
    - Exactly one pixel of the start bar lies on the grid border.

    Rule:
    - Start from the border pixel of the bar and move through the other bar pixel into the grid.
    - Paint every traversed non-blocker pixel with the start color.
    - When the next cell is a blocker, turn to the unique perpendicular non-blocker neighbor; when the next cell is outside the grid, stop.

    Key insight: The blocker pixels define a one-cell-wide corridor, so every interior collision has exactly one valid 90-degree turn.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected a path color and a blocker color"

    counts = {c: int(np.count_nonzero(g == c)) for c in fg_colors}
    path_colors = [c for c, n in counts.items() if n == 2]
    assert len(path_colors) == 1, "expected exactly one 2-pixel start bar"
    path_color = int(path_colors[0])
    blocker_color = int([c for c in fg_colors if c != path_color][0])

    start_pixels = [tuple(map(int, rc)) for rc in np.argwhere(g == path_color)]
    assert len(start_pixels) == 2
    (r0, c0), (r1, c1) = start_pixels
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "start bar must be 4-connected"

    def is_border(pixel: tuple[int, int]) -> bool:
        r, c = pixel
        return r == 0 or r == H - 1 or c == 0 or c == W - 1

    border_pixels = [p for p in start_pixels if is_border(p)]
    assert len(border_pixels) == 1, "exactly one start pixel must touch the border"
    border_pixel = border_pixels[0]
    inner_pixel = start_pixels[1] if start_pixels[0] == border_pixel else start_pixels[0]

    dr = int(inner_pixel[0] - border_pixel[0])
    dc = int(inner_pixel[1] - border_pixel[1])
    assert abs(dr) + abs(dc) == 1

    out = g.copy()
    r, c = inner_pixel
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "path walk should not cycle"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(g.shape, nr, nc):
            break

        if int(g[nr, nc]) != blocker_color:
            out[nr, nc] = path_color
            r, c = int(nr), int(nc)
            continue

        choices: list[tuple[int, int]] = []
        for tdr, tdc in ((-dc, dr), (dc, -dr)):
            tr, tc = r + tdr, c + tdc
            if in_bounds(g.shape, tr, tc) and int(g[tr, tc]) != blocker_color:
                choices.append((int(tdr), int(tdc)))

        assert len(choices) == 1, "each blocker collision must have one valid turn"
        dr, dc = choices[0]

    return out

# --------------- generator (att1) ---------------

# Fixed palette from the observed task examples.
BG_COLOR, PATH_COLOR, BLOCKER_COLOR = 7, 3, 8

# Canvas and trajectory bounds.
MIN_GRID_SIZE, MAX_GRID_SIZE = 5, 8
MIN_DIM_BASE, MIN_DIM_TURN_DIVISOR = 3, 2
MIN_TURNS, MAX_TURNS = 3, 7
MIN_UNIQUE_PATH, MAX_UNIQUE_PATH = 8, 22

# Segment, blocker, and retry limits.
MAX_SEGMENT_LEN = 5
MIN_FWD_FOR_TURN, MIN_FWD_AFTER_TURN, MIN_FWD_WITH_REMAINING_TURNS = 2, 1, 2
MIN_BLOCKERS, MAX_BLOCKERS = 6, 14
MIN_EXTRA_BLOCKERS, MAX_EXTRA_BLOCKERS = 0, 2
GEN_RETRIES = 300

ENTRY_SIDES = ['top', 'bottom', 'left', 'right']
NEIGHBORS_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def _max_forward(r, c, dr, dc, H, W):
    """Return the number of in-bounds cells before leaving the grid."""
    steps = 0
    while True:
        r, c = r + dr, c + dc
        if r < 0 or r >= H or c < 0 or c >= W:
            return steps
        steps += 1


def _choose_entry_bar(H, W):
    """Choose the border start bar and its inward direction."""
    side = random.choice(ENTRY_SIDES)
    if side == 'top':
        c = random.randint(0, W - 1)
        return (0, c), (1, c), 1, 0
    if side == 'bottom':
        c = random.randint(0, W - 1)
        return (H - 1, c), (H - 2, c), -1, 0
    if side == 'left':
        r = random.randint(0, H - 1)
        return (r, 0), (r, 1), 0, 1
    r = random.randint(0, H - 1)
    return (r, W - 1), (r, W - 2), 0, -1


def _extend_path(grid, path_cells, r, c, dr, dc, seg_len=None, allow_path_color=False):
    """Append a bounded segment, or walk until exit when seg_len is None."""
    H, W = grid.shape
    steps = 0
    while True:
        if seg_len is not None and steps == seg_len:
            return r, c
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            return (r, c) if seg_len is None else None
        if grid[nr, nc] != BG_COLOR and not (allow_path_color and grid[nr, nc] == PATH_COLOR):
            return None
        path_cells.append((nr, nc))
        r, c = nr, nc
        steps += 1


def _place_turn_blockers(grid, path_cells, blocker_cells, r, c, dr, dc, num_turns, seg_i):
    """Place the ahead and wrong-side blockers that force one valid turn."""
    H, W = grid.shape
    ahead_r, ahead_c = r + dr, c + dc
    if grid[ahead_r, ahead_c] != BG_COLOR:
        return None

    perps = [(-dc, dr), (dc, -dr)]
    random.shuffle(perps)
    for chosen_dir in perps:
        other_dir = perps[1] if chosen_dir is perps[0] else perps[0]
        cr, cc = r + chosen_dir[0], c + chosen_dir[1]
        otr, otc = r + other_dir[0], c + other_dir[1]
        c_in = 0 <= cr < H and 0 <= cc < W
        o_in = 0 <= otr < H and 0 <= otc < W

        if not c_in or grid[cr, cc] != BG_COLOR:
            continue

        fwd_room = _max_forward(r, c, chosen_dir[0], chosen_dir[1], H, W)
        if fwd_room < MIN_FWD_AFTER_TURN:
            continue
        remaining_turns = num_turns - seg_i - 1
        if remaining_turns > 0 and fwd_room < MIN_FWD_WITH_REMAINING_TURNS:
            continue

        if o_in:
            if grid[otr, otc] == BG_COLOR:
                if (otr, otc) in path_cells:
                    continue
                grid[otr, otc] = BLOCKER_COLOR
                blocker_cells.append((otr, otc))
            elif grid[otr, otc] != BLOCKER_COLOR:
                continue

        grid[ahead_r, ahead_c] = BLOCKER_COLOR
        blocker_cells.append((ahead_r, ahead_c))
        return chosen_dir
    return None


def _scatter_extra_blockers(grid, path_cells, bar_cells, blocker_cells):
    """Add optional distractor blockers that are not adjacent to the path/start."""
    H, W = grid.shape
    occupied = set(path_cells) | bar_cells
    for _ in range(random.randint(MIN_EXTRA_BLOCKERS, MAX_EXTRA_BLOCKERS)):
        free = [
            (rr, cc) for rr in range(H) for cc in range(W)
            if grid[rr, cc] == BG_COLOR and (rr, cc) not in occupied
            and not any((rr + ddr, cc + ddc) in occupied for ddr, ddc in NEIGHBORS_4)
        ]
        if free:
            cell = random.choice(free)
            grid[cell] = BLOCKER_COLOR
            blocker_cells.append(cell)


def _build_validated_input(grid, path_cells) -> np.ndarray | None:
    """Erase sampled path cells and require the solver to reconstruct them."""
    inp = grid.copy()
    for pr, pc in path_cells:
        inp[pr, pc] = BG_COLOR

    try:
        out = main(inp)
    except Exception:
        return None

    expected = inp.copy()
    for pr, pc in path_cells:
        expected[pr, pc] = PATH_COLOR
    return inp if np.array_equal(out, expected) else None


def _gen_attempt() -> np.ndarray | None:
    num_turns = random.randint(MIN_TURNS, MAX_TURNS)
    min_dim = max(MIN_GRID_SIZE, MIN_DIM_BASE + num_turns // MIN_DIM_TURN_DIVISOR)
    H = random.randint(min_dim, MAX_GRID_SIZE)
    W = random.randint(min_dim, MAX_GRID_SIZE)
    grid = np.full((H, W), BG_COLOR, dtype=int)

    border_pixel, inner_pixel, dr, dc = _choose_entry_bar(H, W)
    grid[border_pixel] = PATH_COLOR
    grid[inner_pixel] = PATH_COLOR

    r, c = inner_pixel
    path_cells = []
    blocker_cells = []

    for seg_i in range(num_turns + 1):
        is_final = (seg_i == num_turns)

        if is_final:
            pos = _extend_path(grid, path_cells, r, c, dr, dc)
            if pos is None:
                return None
            r, c = pos
            break

        max_fwd = _max_forward(r, c, dr, dc, H, W)
        if max_fwd < MIN_FWD_FOR_TURN:
            return None

        max_seg = min(MAX_SEGMENT_LEN, max_fwd - 1)
        seg_len = random.choices(range(1, max_seg + 1), weights=range(1, max_seg + 1))[0]
        pos = _extend_path(grid, path_cells, r, c, dr, dc, seg_len, allow_path_color=True)
        if pos is None:
            return None
        r, c = pos

        chosen_dir = _place_turn_blockers(grid, path_cells, blocker_cells, r, c, dr, dc,
                                          num_turns, seg_i)
        if chosen_dir is None:
            return None
        dr, dc = chosen_dir

    unique_path = len(set(path_cells))
    if not (MIN_UNIQUE_PATH <= unique_path <= MAX_UNIQUE_PATH):
        return None

    _scatter_extra_blockers(grid, path_cells, {border_pixel, inner_pixel}, blocker_cells)
    if not (MIN_BLOCKERS <= len(blocker_cells) <= MAX_BLOCKERS):
        return None

    return _build_validated_input(grid, path_cells)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
