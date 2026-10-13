import numpy as np


BLACK = 0
BLUE = 1
RED = 2
TOKEN_COLORS = (3, 4, 5, 6, 7, 8, 9)
RNG = np.random.default_rng()
GEN_RETRIES = 400


def _start_column(grid: np.ndarray) -> int:
    red = np.argwhere(grid == RED)
    assert red.shape == (1, 2), "expected exactly one red start pixel"
    r, c = map(int, red[0])
    assert r == grid.shape[0] - 1, "red start must be on the bottom row"
    return c


def _token_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BLACK) & (grid != RED)


def _simulate_path(
    height: int,
    width: int,
    start_c: int,
    token_positions: set[tuple[int, int]],
) -> dict:
    r = height - 1
    c = int(start_c)
    dc = -1 if c == width - 1 else 1
    path: list[tuple[int, int]] = []
    hit_coords: list[tuple[int, int]] = []
    wall_bounces = 0
    events: list[dict] = []

    for step in range(1, height):
        next_c = c + dc
        bounced = False
        if next_c < 0 or next_c >= width:
            dc *= -1
            next_c = c + dc
            bounced = True
            wall_bounces += 1

        r -= 1
        c = next_c
        coord = (r, c)
        path.append(coord)

        hit = coord in token_positions
        if hit:
            hit_coords.append(coord)
            dc *= -1

        events.append(
            {
                "step": step,
                "coord": coord,
                "bounced": bounced,
                "hit": hit,
                "dc_after": dc,
            }
        )

    return {
        "path": path,
        "hit_coords": hit_coords,
        "wall_bounces": wall_bounces,
        "events": events,
    }


def _simulate_from_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert BLUE not in set(map(int, np.unique(g))), "input may not contain blue"
    assert not np.isin(g[-1], TOKEN_COLORS).any(), "bottom row may only contain the red start"
    start_c = _start_column(g)
    tokens = {tuple(map(int, rc)) for rc in np.argwhere(_token_mask(g))}
    return _simulate_path(g.shape[0], g.shape[1], start_c, tokens)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the bouncing diagonal ray from the red start and color its visited cells blue.

    Input structure:
    - The grid is a black background with exactly one red start pixel on the bottom row.
    - The remaining foreground cells are isolated single-pixel tokens whose colors are
      neither black, red, nor blue.
    - The left and right grid borders act as reflecting walls.

    Rule:
    - Start from the red pixel with horizontal direction to the right, except when the
      start is already in the rightmost column, in which case the initial direction is left.
    - Each move goes one row upward and one column sideways; if the sideways move would
      leave the grid, reverse the horizontal direction first and then take the step.
    - Whenever the path lands on a token, that token is absorbed into the path and the
      horizontal direction reverses immediately for the following step.
    - Stop after reaching the top row. Keep the red start red, recolor every visited
      path cell blue, preserve only the tokens that were never visited, and leave all
      other cells black.

    Key insight:
    - Tokens are active prisms rather than passive collectibles, so the path must be
      simulated in order because each earlier collision changes the later trajectory.
    """
    g = np.asarray(grid, dtype=int)
    sim = _simulate_from_grid(g)

    out = np.zeros_like(g)
    token_mask = _token_mask(g)
    out[token_mask] = g[token_mask]

    for r, c in sim["path"]:
        out[r, c] = BLUE

    out[g == RED] = RED
    return out


def _simulate_hit_steps(height: int, width: int, start_c: int, hit_steps: set[int]) -> dict:
    r = height - 1
    c = int(start_c)
    dc = -1 if c == width - 1 else 1
    path: list[tuple[int, int]] = []
    hit_coords: list[tuple[int, int]] = []
    wall_bounces = 0

    for step in range(1, height):
        next_c = c + dc
        if next_c < 0 or next_c >= width:
            dc *= -1
            next_c = c + dc
            wall_bounces += 1

        r -= 1
        c = next_c
        coord = (r, c)
        path.append(coord)
        if step in hit_steps:
            hit_coords.append(coord)
            dc *= -1

    return {"path": path, "hit_coords": hit_coords, "wall_bounces": wall_bounces}


def _choose_hit_steps(height: int) -> set[int]:
    hit_count = int(RNG.choice([2, 2, 3]))
    candidates = list(range(2, height - 1))
    if len(candidates) < hit_count:
        return set()

    chosen: list[int] = []
    shuffled = list(RNG.permutation(candidates))
    for step in shuffled:
        if all(abs(step - prev) >= 2 for prev in chosen):
            chosen.append(int(step))
        if len(chosen) == hit_count:
            break

    if len(chosen) != hit_count:
        return set()
    return set(chosen)


def _build_grid(
    height: int,
    width: int,
    start_c: int,
    hit_coords: list[tuple[int, int]],
    actual_path: list[tuple[int, int]],
    empty_path: list[tuple[int, int]],
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    grid[height - 1, start_c] = RED

    palette_size = int(RNG.integers(3, min(6, len(TOKEN_COLORS)) + 1))
    palette = list(RNG.choice(TOKEN_COLORS, size=palette_size, replace=False))

    for idx, (r, c) in enumerate(hit_coords):
        grid[r, c] = palette[idx % palette_size]

    actual_set = set(actual_path)
    empty_only = [coord for coord in empty_path if coord not in actual_set]
    candidates = [
        (r, c)
        for r in range(height - 1)
        for c in range(width)
        if (r, c) not in actual_set and grid[r, c] == BLACK
    ]

    if not empty_only:
        return None

    off_path: list[tuple[int, int]] = []
    decoy = empty_only[int(RNG.integers(len(empty_only)))]
    off_path.append(decoy)

    same_row_choices = [
        (r, c)
        for r, c in candidates
        if any(r == pr and abs(c - pc) >= 1 for pr, pc in actual_path)
        and (r, c) != decoy
    ]
    if same_row_choices:
        off_path.append(same_row_choices[int(RNG.integers(len(same_row_choices)))])

    RNG.shuffle(candidates)
    target_off_path = int(RNG.integers(3, min(8, len(candidates) + 1)))
    for coord in candidates:
        if coord in off_path:
            continue
        off_path.append(coord)
        if len(off_path) >= target_off_path:
            break

    off_path = list(dict.fromkeys(off_path))
    if len(off_path) < 3:
        return None

    visible_colors = list(RNG.choice(palette, size=len(off_path), replace=True))
    if len(set(visible_colors)) == 1 and len(palette) > 1:
        visible_colors[-1] = palette[1]

    for coord, color in zip(off_path, visible_colors):
        r, c = coord
        grid[r, c] = int(color)

    return grid


def _is_valid_example(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    if np.count_nonzero(grid == RED) != 1:
        return False

    sim = _simulate_from_grid(grid)
    start_c = _start_column(grid)
    empty = _simulate_path(grid.shape[0], grid.shape[1], start_c, set())
    out = main(grid)

    actual_path = sim["path"]
    hit_coords = sim["hit_coords"]
    token_coords = {tuple(map(int, rc)) for rc in np.argwhere(_token_mask(grid))}
    off_path_tokens = token_coords - set(actual_path)
    empty_only = set(empty["path"]) - set(actual_path)

    if len(hit_coords) < 2:
        return False
    if not off_path_tokens:
        return False
    if not empty_only:
        return False
    if sim["wall_bounces"] < 1:
        return False
    if set(hit_coords).issubset(set(empty["path"])):
        return False
    if actual_path == empty["path"]:
        return False
    if not any(coord in off_path_tokens for coord in empty_only):
        return False

    red_r, red_c = map(int, np.argwhere(grid == RED)[0])
    expected_nonblack = set(actual_path) | off_path_tokens | {(red_r, red_c)}
    out_nonblack = {tuple(map(int, rc)) for rc in np.argwhere(out != BLACK)}
    if out[red_r, red_c] != RED:
        return False
    if any(out[r, c] != BLUE for r, c in actual_path):
        return False
    if any(out[r, c] != grid[r, c] for r, c in off_path_tokens):
        return False
    if out_nonblack != expected_nonblack:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(9, 19))
    width = int(RNG.integers(5, 13))
    if RNG.random() < 0.4:
        start_c = width - 1
    else:
        start_c = int(RNG.integers(0, width - 1))

    hit_steps = _choose_hit_steps(height)
    if not hit_steps:
        return None

    actual = _simulate_hit_steps(height, width, start_c, hit_steps)
    empty = _simulate_hit_steps(height, width, start_c, set())

    if actual["wall_bounces"] < 1:
        return None
    if len(actual["hit_coords"]) < 2:
        return None
    if set(actual["hit_coords"]).issubset(set(empty["path"])):
        return None

    grid = _build_grid(
        height,
        width,
        start_c,
        actual["hit_coords"],
        actual["path"],
        empty["path"],
    )
    if grid is None:
        return None
    if not _is_valid_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid prism-bounce input")
