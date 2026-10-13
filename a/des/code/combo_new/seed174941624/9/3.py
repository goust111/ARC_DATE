from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
WALL = 5
COLOR_POOL = (1, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
SYSTEM_RETRIES = 160


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_axis(grid: np.ndarray) -> tuple[str, int]:
    g = np.asarray(grid, dtype=int)
    row_axes = [r for r in range(g.shape[0]) if np.all(g[r, :] == RED)]
    col_axes = [c for c in range(g.shape[1]) if np.all(g[:, c] == RED)]
    assert len(row_axes) + len(col_axes) == 1, "expected exactly one full red mirror axis"
    if row_axes:
        return ("row", int(row_axes[0]))
    return ("col", int(col_axes[0]))


def _is_passable(grid: np.ndarray, r: int, c: int) -> bool:
    return _in_bounds(grid.shape, r, c) and int(grid[r, c]) not in {WALL, RED}


def _is_wall_like(grid: np.ndarray, r: int, c: int) -> bool:
    return not _in_bounds(grid.shape, r, c) or int(grid[r, c]) == WALL


def _infer_orientation(grid: np.ndarray, r: int, c: int) -> str:
    horizontal = (
        (_is_passable(grid, r, c - 1) or _is_passable(grid, r, c + 1))
        and _is_wall_like(grid, r - 1, c)
        and _is_wall_like(grid, r + 1, c)
    )
    vertical = (
        (_is_passable(grid, r - 1, c) or _is_passable(grid, r + 1, c))
        and _is_wall_like(grid, r, c - 1)
        and _is_wall_like(grid, r, c + 1)
    )
    assert horizontal ^ vertical, "colored pixel must lie in exactly one straight channel"
    return "h" if horizontal else "v"


def _walk_to_terminal(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> tuple[int, int]:
    r, c = start
    dr, dc = step
    while _is_passable(grid, r + dr, c + dc):
        r += dr
        c += dc
    return (int(r), int(c))


def _is_closed_end(grid: np.ndarray, end: tuple[int, int], step: tuple[int, int], orient: str) -> bool:
    r, c = end
    dr, dc = step
    fr, fc = r + dr, c + dc
    if not _in_bounds(grid.shape, fr, fc) or int(grid[fr, fc]) != WALL:
        return False
    perp_dirs = ((-1, 0), (1, 0)) if orient == "h" else ((0, -1), (0, 1))
    return not any(_is_passable(grid, r + pr, c + pc) for pr, pc in perp_dirs)


def _slide_destination(grid: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    r, c = start
    orient = _infer_orientation(grid, r, c)
    steps = ((0, -1), (0, 1)) if orient == "h" else ((-1, 0), (1, 0))
    terminals = [_walk_to_terminal(grid, start, step) for step in steps]
    closed = [_is_closed_end(grid, end, step, orient) for end, step in zip(terminals, steps)]
    assert sum(closed) == 1, "channel must have exactly one gray-blocked end"
    return terminals[int(closed[1])]


def _reflect(axis: tuple[str, int], cell: tuple[int, int]) -> tuple[int, int]:
    kind, fixed = axis
    r, c = cell
    if kind == "row":
        return (int(2 * fixed - r), int(c))
    return (int(r), int(2 * fixed - c))


def _token_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    mask = (grid != BG) & (grid != RED) & (grid != WALL)
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored channel token to its dead end and add its mirror image across the red axis.

    Input structure:
    - The background is black.
    - Gray pixels form one-cell-wide corridor walls, leaving black corridor paths that are straight
      horizontal or vertical channels with exactly one colored pixel in each channel.
    - A channel has one open end, which reaches the grid edge or a perpendicular corridor junction,
      and one closed end, which is blocked by a gray wall cell.
    - A full red row or full red column acts as the mirror axis.

    Rule:
    - For each colored pixel, infer whether its local channel is horizontal or vertical from the gray
      side walls around it.
    - Follow that channel in both directions and identify the unique end whose next cell is a gray wall
      and whose endpoint is not also open into a perpendicular corridor; move the colored pixel there.
    - Clear the original colored position to black.
    - Reflect the moved endpoint across the red axis and place a same-colored pixel at that symmetric
      cell only if the reflected cell is inside the grid and currently black.

    Key insight:
    - The colored pixels do not move toward the nearest edge or toward the red line; they move to the
      unique dead end determined by corridor geometry, and only after that final position is known is
      the same color mirrored across the red axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    axis = _find_axis(g)
    tokens = sorted(_token_cells(g))

    base = g.copy()
    if tokens:
        rows = np.array([r for r, _ in tokens], dtype=int)
        cols = np.array([c for _, c in tokens], dtype=int)
        base[rows, cols] = BG

    out = base.copy()
    kind, fixed = axis
    if kind == "row":
        out[fixed, :] = RED
    else:
        out[:, fixed] = RED

    moved: list[tuple[tuple[int, int], int]] = []
    for r, c in tokens:
        color = int(g[r, c])
        dst = _slide_destination(base, (r, c))
        assert int(out[dst]) == BG, "slid channels must end at distinct black cells"
        out[dst] = color
        moved.append((dst, color))

    for dst, color in moved:
        mr, mc = _reflect(axis, dst)
        if _in_bounds(out.shape, mr, mc) and int(out[mr, mc]) == BG:
            out[mr, mc] = color

    return out


def _reflect_coords(coords: set[tuple[int, int]], height: int) -> set[tuple[int, int]]:
    return {(int(height - 1 - r), int(c)) for r, c in coords}


def _reflect_list(coords: list[tuple[int, int]], height: int) -> list[tuple[int, int]]:
    return [(int(height - 1 - r), int(c)) for r, c in coords]


def _buffer_cells(coords: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    buffered: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    buffered.add((int(nr), int(nc)))
    return buffered


def _sample_top_system(width: int, half_h: int, branch: bool) -> tuple[set[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(SYSTEM_RETRIES):
        trunk_end = random.randint(4 if branch else 3, half_h - 2)

        if branch:
            branch_row = random.randint(2, trunk_end - 2)
            candidates: list[tuple[int, int]] = []
            for col in range(4, width - 4):
                left_max = col - 1
                right_max = width - 2 - col
                if left_max >= 3:
                    candidates.append((col, -1))
                if right_max >= 3:
                    candidates.append((col, 1))
            if not candidates:
                return None
            col, side = random.choice(candidates)
            max_len = (col - 1) if side < 0 else (width - 2 - col)
            branch_len = random.randint(3, min(max_len, 7))
            trunk_token_choices = [row for row in range(1, trunk_end) if row != branch_row]
            if not trunk_token_choices:
                continue
            trunk_token = (int(random.choice(trunk_token_choices)), int(col))
            branch_offset = random.randint(1, branch_len - 1)
            branch_token = (int(branch_row), int(col + side * branch_offset))

            passable = {(int(r), int(col)) for r in range(trunk_end + 1)}
            for offset in range(1, branch_len + 1):
                passable.add((int(branch_row), int(col + side * offset)))
            return passable, [trunk_token, branch_token]

        col = random.randint(2, width - 3)
        trunk_token = (int(random.randint(1, trunk_end - 1)), int(col))
        passable = {(int(r), int(col)) for r in range(trunk_end + 1)}
        return passable, [trunk_token]

    return None


def _place_system(
    width: int,
    height: int,
    half_h: int,
    branch: bool,
    paired: bool,
    reserve: set[tuple[int, int]],
    reflect_only: bool = False,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(SYSTEM_RETRIES):
        sampled = _sample_top_system(width, half_h, branch)
        if sampled is None:
            continue
        top_passable, top_tokens = sampled

        if paired:
            global_passable = set(top_passable) | _reflect_coords(top_passable, height)
            global_tokens = list(top_tokens) + _reflect_list(top_tokens, height)
        elif reflect_only:
            global_passable = _reflect_coords(top_passable, height)
            global_tokens = _reflect_list(top_tokens, height)
        else:
            global_passable = set(top_passable)
            global_tokens = list(top_tokens)

        footprint = _buffer_cells(global_passable, height, width)
        if footprint & reserve:
            continue
        return global_passable, global_tokens, footprint

    return None


def _derive_walls(passable: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    walls: set[tuple[int, int]] = set()
    for r, c in passable:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in passable:
                walls.add((int(nr), int(nc)))
    return walls


def _build_candidate() -> np.ndarray | None:
    half_h = random.randint(7, 12)
    height = 2 * half_h + 1
    width = random.randint(18, 28)
    axis_r = half_h

    reserve = {(axis_r, c) for c in range(width)}
    passable_all: set[tuple[int, int]] = set()
    tokens_all: list[tuple[int, int]] = []

    paired_branch = random.random() < 0.7
    unpaired_branch = random.random() < 0.7
    if not (paired_branch or unpaired_branch):
        if random.random() < 0.5:
            paired_branch = True
        else:
            unpaired_branch = True

    paired = _place_system(width, height, half_h, paired_branch, paired=True, reserve=reserve)
    if paired is None:
        return None
    paired_passable, paired_tokens, paired_footprint = paired
    passable_all |= paired_passable
    tokens_all.extend(paired_tokens)
    reserve |= paired_footprint

    reflect_unpaired = random.random() < 0.5
    unpaired = _place_system(
        width,
        height,
        half_h,
        unpaired_branch,
        paired=False,
        reserve=reserve,
        reflect_only=reflect_unpaired,
    )
    if unpaired is None:
        return None
    unpaired_passable, unpaired_tokens, unpaired_footprint = unpaired
    passable_all |= unpaired_passable
    tokens_all.extend(unpaired_tokens)
    reserve |= unpaired_footprint

    if len(tokens_all) > len(COLOR_POOL):
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[axis_r, :] = RED

    for r, c in _derive_walls(passable_all, height, width):
        if int(grid[r, c]) == BG:
            grid[r, c] = WALL

    colors = list(COLOR_POOL)
    random.shuffle(colors)
    for (r, c), color in zip(tokens_all, colors):
        grid[r, c] = int(color)

    if random.random() < 0.5:
        grid = np.rot90(grid)
    if random.random() < 0.5:
        grid = np.fliplr(grid)
    if random.random() < 0.5:
        grid = np.flipud(grid)

    return np.asarray(grid, dtype=int)


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    input_tokens = int(np.count_nonzero((grid != BG) & (grid != RED) & (grid != WALL)))
    output_tokens = int(np.count_nonzero((out != BG) & (out != RED) & (out != WALL)))
    mirrors_added = output_tokens - input_tokens
    if input_tokens < 3:
        return False
    if mirrors_added <= 0 or mirrors_added >= input_tokens:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if grid is None:
            continue
        if max(grid.shape) > 30:
            continue
        if _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid mirrored-channel input")
