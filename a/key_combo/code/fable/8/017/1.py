import numpy as np


CONTROL_COLOR = 2
MAX_GENERATION_ATTEMPTS = 400
GRID_SIDE_MIN = 14
GRID_SIDE_MAX = 30
WINDOW_SIZE_RANGE = (5, 12)
TILE_SIZE_RANGE = (2, 4)
REPEAT_COUNT_RANGE = (2, 4)
MIN_CLEARANCE = 2
BACKGROUND_CHOICES = (0, 1, 3, 4, 5, 6, 7, 8, 9)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _smallest_period_1d(length: int, value_at) -> int:
    for period in range(1, length + 1):
        if length % period != 0:
            continue
        ok = True
        for idx in range(length):
            if value_at(idx) != value_at(idx % period):
                ok = False
                break
        if ok:
            return period
    return length


def _tile_periods(patch: np.ndarray) -> tuple[int, int]:
    h, w = patch.shape
    period_h = _smallest_period_1d(
        h,
        lambda r: tuple(map(int, patch[r, :])),
    )
    period_w = _smallest_period_1d(
        w,
        lambda c: tuple(map(int, patch[:, c])),
    )
    return period_h, period_w


def _find_control_points(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    candidates: list[tuple[int, np.ndarray]] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        coords = np.argwhere(grid == color)
        if coords.shape[0] != 4:
            continue

        rows = coords[:, 0]
        cols = coords[:, 1]
        if np.sum(rows == rows.min()) != 1 or np.sum(rows == rows.max()) != 1:
            continue
        if np.sum(cols == cols.min()) != 1 or np.sum(cols == cols.max()) != 1:
            continue

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        rect = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = np.zeros(rect.shape, dtype=bool)
        for r, c in coords:
            mask[r - r0, c - c0] = True
        if np.all(rect[~mask] == bg):
            candidates.append((color, coords))

    assert len(candidates) == 1, "expected exactly one valid control-point color"
    color, coords = candidates[0]
    return int(color), coords.astype(int)


def _find_seed_bbox(grid: np.ndarray, bg: int, control_coords: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid != bg
    mask[control_coords[:, 0], control_coords[:, 1]] = False
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty seed block"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert mask[r0 : r1 + 1, c0 : c1 + 1].all(), "seed block must be a solid rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the globally phased wallpaper inside the rectangle implied by four control pixels.

    Input structure:
    - A dominant background contains one solid rectangular seed block whose colors already form repeated rows and columns.
    - Four detached single-pixel control markers share one color and uniquely define the top, bottom, left, and right extremes.
    - The closed rectangle spanned by those extreme rows and columns is otherwise all background and does not overlap the seed block.

    Rule:
    - Recover the minimal repeating tile from the seed block's rectangular contents.
    - Treat that tile as part of an infinite wallpaper anchored to the seed block's absolute top-left coordinate.
    - Fill every background cell inside the control-point rectangle with the wallpaper color at the same absolute coordinate.
    - Leave the four control pixels and everything outside the window unchanged.

    Key insight:
    - The window does not start a fresh local tiling; it exposes a wallpaper whose phase is fixed by where the seed block already sits on the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    _, control_coords = _find_control_points(g, bg)
    sr0, sc0, sr1, sc1 = _find_seed_bbox(g, bg, control_coords)
    seed_patch = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    tile_h, tile_w = _tile_periods(seed_patch)
    tile = seed_patch[:tile_h, :tile_w]

    rows = control_coords[:, 0]
    cols = control_coords[:, 1]
    wr0, wr1 = int(rows.min()), int(rows.max())
    wc0, wc1 = int(cols.min()), int(cols.max())

    out = g.copy()
    for r in range(wr0, wr1 + 1):
        for c in range(wc0, wc1 + 1):
            if out[r, c] != bg:
                continue
            out[r, c] = tile[(r - sr0) % tile_h, (c - sc0) % tile_w]
    return out


def _sample_tile(rng: np.random.Generator, colors: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    while True:
        tile = rng.choice(colors, size=(tile_h, tile_w), replace=True)
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        period_h, period_w = _tile_periods(tile)
        if period_h == tile_h and period_w == tile_w:
            return tile.astype(int)


def _rectangles_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], clearance: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + clearance < br0
        or br1 + clearance < ar0
        or ac1 + clearance < bc0
        or bc1 + clearance < ac0
    )


def _choose_window_origin(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    win_h: int,
    win_w: int,
    seed_bbox: tuple[int, int, int, int],
    seed_origin: tuple[int, int],
    tile_shape: tuple[int, int],
    phase_case: int,
) -> tuple[int, int] | None:
    sr0, sc0 = seed_origin
    tile_h, tile_w = tile_shape
    row_modes = {
        0: lambda r: (r - sr0) % tile_h != 0,
        1: lambda r: (r - sr0) % tile_h == 0,
        2: lambda r: (r - sr0) % tile_h != 0,
    }
    col_modes = {
        0: lambda c: (c - sc0) % tile_w == 0,
        1: lambda c: (c - sc0) % tile_w != 0,
        2: lambda c: (c - sc0) % tile_w != 0,
    }

    row_candidates = [r for r in range(0, grid_h - win_h + 1) if row_modes[phase_case](r)]
    col_candidates = [c for c in range(0, grid_w - win_w + 1) if col_modes[phase_case](c)]
    rng.shuffle(row_candidates)
    rng.shuffle(col_candidates)

    for r0 in row_candidates:
        for c0 in col_candidates:
            bbox = (r0, c0, r0 + win_h - 1, c0 + win_w - 1)
            if _rectangles_clear(bbox, seed_bbox, MIN_CLEARANCE):
                return r0, c0
    return None


def _place_control_points(
    rng: np.random.Generator,
    win_r0: int,
    win_c0: int,
    win_h: int,
    win_w: int,
) -> list[tuple[int, int]]:
    r1 = win_r0 + win_h - 1
    c1 = win_c0 + win_w - 1
    for _ in range(50):
        top = (win_r0, int(rng.integers(win_c0 + 1, c1)))
        bottom = (r1, int(rng.integers(win_c0 + 1, c1)))
        left = (int(rng.integers(win_r0 + 1, r1)), win_c0)
        right = (int(rng.integers(win_r0 + 1, r1)), c1)
        points = [top, bottom, left, right]
        if len(set(points)) == 4:
            return points
    raise AssertionError("failed to place distinct control points")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.choice(BACKGROUND_CHOICES))
    palette = np.array([c for c in range(10) if c not in {bg, CONTROL_COLOR}], dtype=int)
    rng.shuffle(palette)

    tile_h = int(rng.integers(TILE_SIZE_RANGE[0], TILE_SIZE_RANGE[1] + 1))
    tile_w = int(rng.integers(TILE_SIZE_RANGE[0], TILE_SIZE_RANGE[1] + 1))
    reps_h = int(rng.integers(REPEAT_COUNT_RANGE[0], REPEAT_COUNT_RANGE[1] + 1))
    reps_w = int(rng.integers(REPEAT_COUNT_RANGE[0], REPEAT_COUNT_RANGE[1] + 1))
    seed_h, seed_w = tile_h * reps_h, tile_w * reps_w

    win_h = int(rng.integers(WINDOW_SIZE_RANGE[0], WINDOW_SIZE_RANGE[1] + 1))
    win_w = int(rng.integers(WINDOW_SIZE_RANGE[0], WINDOW_SIZE_RANGE[1] + 1))
    if win_h < 5 or win_w < 5:
        return None

    min_h = max(GRID_SIDE_MIN, seed_h + 2, win_h + 2, seed_h + win_h + MIN_CLEARANCE + 1)
    min_w = max(GRID_SIDE_MIN, seed_w + 2, win_w + 2, seed_w + win_w + MIN_CLEARANCE + 1)
    if min_h > GRID_SIDE_MAX or min_w > GRID_SIDE_MAX:
        return None

    grid_h = int(rng.integers(min_h, GRID_SIDE_MAX + 1))
    grid_w = int(rng.integers(min_w, GRID_SIDE_MAX + 1))
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    seed_r0 = int(rng.integers(0, grid_h - seed_h + 1))
    seed_c0 = int(rng.integers(0, grid_w - seed_w + 1))
    seed_bbox = (seed_r0, seed_c0, seed_r0 + seed_h - 1, seed_c0 + seed_w - 1)

    color_count = int(rng.integers(2, 4))
    seed_colors = palette[:color_count]
    tile = _sample_tile(rng, seed_colors, tile_h, tile_w)
    seed_patch = np.tile(tile, (reps_h, reps_w))
    grid[seed_r0 : seed_r0 + seed_h, seed_c0 : seed_c0 + seed_w] = seed_patch

    phase_case = int(rng.integers(0, 3))
    win_origin = _choose_window_origin(
        rng,
        grid_h,
        grid_w,
        win_h,
        win_w,
        seed_bbox,
        (seed_r0, seed_c0),
        (tile_h, tile_w),
        phase_case,
    )
    if win_origin is None:
        return None
    win_r0, win_c0 = win_origin

    for r, c in _place_control_points(rng, win_r0, win_c0, win_h, win_w):
        grid[r, c] = CONTROL_COLOR

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    filled = out[win_r0 : win_r0 + win_h, win_c0 : win_c0 + win_w]
    if np.count_nonzero(filled == bg) > 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wallpaper-window input")
