import numpy as np


_RNG = np.random.default_rng()
_PALETTE = tuple(range(10))

_GRID_MIN = 7
_GRID_MAX = 19
_ACTIVE_MIN = 2
_ACTIVE_MAX = 6
_GEN_RETRIES = 300
_FULL_GROWTH_PROB = 0.6


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _parse_top_band(grid: np.ndarray, bg: int):
    h, w = grid.shape
    band = grid[0]
    active_cols = [c for c in range(w) if int(band[c]) != bg]
    if len(active_cols) < 2:
        return None

    colors = [int(band[c]) for c in active_cols]
    if len(set(colors)) != len(colors):
        return None

    rest = np.argwhere(grid[1:, :] != bg)
    if rest.shape[0] != len(active_cols):
        return None

    rank_by_color = {}
    rank = 0
    for c in range(w):
        color = int(band[c])
        if color == bg:
            continue
        rank += 1
        rank_by_color[color] = rank

    seeds = {}
    for c in active_cols:
        color = int(band[c])
        locs = np.argwhere(grid[1:, c] == color)
        if locs.shape[0] != 1:
            return None
        if int(np.count_nonzero(grid[1:, :] == color)) != 1:
            return None
        seeds[color] = (int(locs[0, 0]) + 1, int(c))

    active_col_set = set(active_cols)
    for r, c in rest:
        rr = int(r) + 1
        cc = int(c)
        if cc not in active_col_set:
            return None
        if int(grid[rr, cc]) != int(band[cc]):
            return None

    return {
        "axis": "top",
        "bg": int(bg),
        "rank_by_color": rank_by_color,
        "seeds": seeds,
    }


def _parse_left_band(grid: np.ndarray, bg: int):
    h, w = grid.shape
    band = grid[:, 0]
    active_rows = [r for r in range(h) if int(band[r]) != bg]
    if len(active_rows) < 2:
        return None

    colors = [int(band[r]) for r in active_rows]
    if len(set(colors)) != len(colors):
        return None

    rest = np.argwhere(grid[:, 1:] != bg)
    if rest.shape[0] != len(active_rows):
        return None

    rank_by_color = {}
    rank = 0
    for r in range(h):
        color = int(band[r])
        if color == bg:
            continue
        rank += 1
        rank_by_color[color] = rank

    seeds = {}
    for r in active_rows:
        color = int(band[r])
        locs = np.argwhere(grid[r, 1:] == color)
        if locs.shape[0] != 1:
            return None
        if int(np.count_nonzero(grid[:, 1:] == color)) != 1:
            return None
        seeds[color] = (int(r), int(locs[0, 0]) + 1)

    active_row_set = set(active_rows)
    for r, c in rest:
        rr = int(r)
        cc = int(c) + 1
        if rr not in active_row_set:
            return None
        if int(grid[rr, cc]) != int(band[rr]):
            return None

    return {
        "axis": "left",
        "bg": int(bg),
        "rank_by_color": rank_by_color,
        "seeds": seeds,
    }


def _parse_instance(grid: np.ndarray):
    bg = _get_bg(grid)
    candidates = [cand for cand in (_parse_top_band(grid, bg), _parse_left_band(grid, bg)) if cand is not None]
    assert len(candidates) == 1, "expected exactly one valid control-band orientation"
    return candidates[0]


def _transform_from_spec(grid: np.ndarray, spec) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    bg = int(spec["bg"])

    for color, (r, c) in spec["seeds"].items():
        steps = int(spec["rank_by_color"][color])
        if spec["axis"] == "top":
            for dr in range(1, steps + 1):
                rr = int(r) + dr
                if rr >= out.shape[0] or int(out[rr, c]) != bg:
                    break
                out[rr, c] = int(color)
        else:
            for dc in range(1, steps + 1):
                cc = int(c) + dc
                if cc >= out.shape[1] or int(out[r, cc]) != bg:
                    break
                out[r, cc] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow color-matched seeds by distances encoded in an edge control band.

    Input structure:
    - The background is the most common color.
    - Either the top row or the left column is a one-cell-wide control band.
    - The control band contains several unique non-background colors, with background gaps allowed at inactive positions.
    - Outside that edge band, each active column or row contains exactly one seed pixel whose color matches the band entry on that column or row.

    Rule:
    - Determine whether the control band is the top row or the left column.
    - Read only the non-background band entries in visible order; the first active color has growth length 1, the second has length 2, and so on, ignoring background gaps.
    - With a top-row band, find the matching seed in the same column and grow that color downward for its length, starting one cell below the seed.
    - With a left-column band, find the matching seed in the same row and grow that color rightward for its length, starting one cell to the right of the seed.
    - Stop a growth if it would leave the grid or would hit a non-background cell. Keep the control band and all seed pixels in place.

    Key insight:
    - The edge band is both a color lookup table and an ordered ruler: color matching identifies which seed to extend, while the seed's extension length comes from the color's position among the active band entries.
    """
    g = np.asarray(grid, dtype=int)
    spec = _parse_instance(g)
    return _transform_from_spec(g, spec)


def _sample_colors(bg: int, count: int) -> list[int]:
    choices = np.array([color for color in _PALETTE if color != int(bg)], dtype=int)
    return [int(color) for color in _RNG.choice(choices, size=count, replace=False)]


def _choose_seed_coord(limit: int, steps: int) -> int:
    if limit <= 1:
        return 1

    full_max = limit - steps - 1
    if full_max >= 1 and float(_RNG.random()) < _FULL_GROWTH_PROB:
        return int(_RNG.integers(1, full_max + 1))

    low = max(1, limit - max(steps, 2))
    return int(_RNG.integers(low, limit))


def _build_case(axis: str) -> np.ndarray:
    h = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    w = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    bg = int(_RNG.integers(0, len(_PALETTE)))

    if axis == "top":
        band_len = w
    else:
        band_len = h

    max_active = min(_ACTIVE_MAX, band_len - 1)
    active_count = int(_RNG.integers(_ACTIVE_MIN, max_active + 1))
    positions = sorted(int(pos) for pos in _RNG.choice(np.arange(1, band_len), size=active_count, replace=False))
    colors = _sample_colors(bg, active_count)

    grid = np.full((h, w), bg, dtype=int)

    if axis == "top":
        for idx, (col, color) in enumerate(zip(positions, colors), start=1):
            grid[0, col] = int(color)
            row = _choose_seed_coord(h, idx)
            grid[row, col] = int(color)
    else:
        for idx, (row, color) in enumerate(zip(positions, colors), start=1):
            grid[row, 0] = int(color)
            col = _choose_seed_coord(w, idx)
            grid[row, col] = int(color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        axis = "top" if int(_RNG.integers(0, 2)) == 0 else "left"
        grid = _build_case(axis)
        spec = _parse_instance(grid)
        out = _transform_from_spec(grid, spec)
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a non-degenerate puzzle instance")
