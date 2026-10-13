import numpy as np


BG = 0
MARKER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_TOWER_HEIGHT = 2
MAX_TOWER_HEIGHT = 8
GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 24)
TOWER_COUNT_OPTIONS = (2, 3, 4, 5)
MIN_COL_GAP = 3
MIN_ROW_GAP = 2
GEN_RETRIES = 400

_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _extract_towers(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    towers = []
    marker_positions = np.argwhere(g == MARKER)
    for mr_raw, mc_raw in marker_positions:
        mr = int(mr_raw)
        mc = int(mc_raw)
        assert mr + 1 < h, "marker must sit above a tower"
        color = int(g[mr + 1, mc])
        if color in (bg, MARKER):
            continue

        height = 0
        row = mr + 1
        while row < h and int(g[row, mc]) == color:
            height += 1
            row += 1

        if height < MIN_TOWER_HEIGHT:
            continue

        towers.append(
            {
                "marker": (mr, mc),
                "color": color,
                "height": int(height),
                "top": int(mr + 1),
                "bottom": int(mr + height),
                "col": mc,
            }
        )

    assert towers, "expected at least one tower"

    body_mask = np.zeros_like(g, dtype=bool)
    for tower in towers:
        body_mask[tower["top"] : tower["bottom"] + 1, tower["col"]] = True

    expected_non_bg = body_mask.copy()
    expected_non_bg[g == MARKER] = True
    assert np.array_equal(g != bg, expected_non_bg), "input must contain only towers and gray markers"

    marker_rows = [int(t["marker"][0]) for t in towers]
    marker_cols = [int(t["col"]) for t in towers]
    assert len(set(marker_rows)) == len(marker_rows), "same-row towers create ambiguous horizontal overlaps"
    assert len(set(marker_cols)) == len(marker_cols), "same-column towers create ambiguous vertical overlaps"

    towers.sort(key=lambda t: (int(t["marker"][0]), int(t["col"])))
    return towers, body_mask


def _cast_ray(start_r: int, start_c: int, dr: int, dc: int, body_mask: np.ndarray):
    h, w = body_mask.shape
    cells = []
    row = int(start_r)
    col = int(start_c)
    while True:
        row += int(dr)
        col += int(dc)
        if not (0 <= row < h and 0 <= col < w):
            break
        if bool(body_mask[row, col]):
            break
        cells.append((int(row), int(col)))
    return cells


def _build_ray_maps(towers, body_mask: np.ndarray):
    vertical = {}
    horizontal = {}

    for idx, tower in enumerate(towers):
        mr, mc = tower["marker"]

        for cell in _cast_ray(mr, mc, -1, 0, body_mask):
            assert cell not in vertical, "multiple vertical rays at one cell are unsupported"
            vertical[cell] = idx

        for dr, dc in ((0, -1), (0, 1)):
            for cell in _cast_ray(mr, mc, dr, dc, body_mask):
                assert cell not in horizontal, "multiple horizontal rays at one cell are unsupported"
                horizontal[cell] = idx

    return vertical, horizontal


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    towers, body_mask = _extract_towers(g)
    vertical, horizontal = _build_ray_maps(towers, body_mask)

    out = g.copy()
    for cell in set(vertical) | set(horizontal):
        row, col = cell
        if int(g[row, col]) == MARKER:
            continue
        if bool(body_mask[row, col]):
            continue

        v_idx = vertical.get(cell)
        h_idx = horizontal.get(cell)

        if v_idx is not None and h_idx is not None:
            v_tower = towers[int(v_idx)]
            h_tower = towers[int(h_idx)]
            if int(v_tower["height"]) > int(h_tower["height"]):
                color = int(v_tower["color"])
            else:
                color = int(h_tower["color"])
        elif v_idx is not None:
            color = int(towers[int(v_idx)]["color"])
        else:
            assert h_idx is not None
            color = int(towers[int(h_idx)]["color"])

        out[row, col] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend tower-colored rays from gray tower-top markers.

    Each tower emits one upward ray and two horizontal rays from its gray marker.
    Rays stop before tower bodies. At vertical-horizontal intersections, the
    taller source tower wins; ties go to the horizontal ray.
    """
    return _solve(grid)


def _sample_spaced_columns(width: int, count: int):
    usable = np.arange(1, width - 1, dtype=int)
    if usable.size < count:
        return None

    for _ in range(80):
        cols = sorted(int(x) for x in _RNG.choice(usable, size=count, replace=False))
        if all(cols[i + 1] - cols[i] >= MIN_COL_GAP for i in range(len(cols) - 1)):
            return cols
    return None


def _sample_pair_heights(branch: str):
    pairs = []
    for vertical_height in range(MIN_TOWER_HEIGHT, MAX_TOWER_HEIGHT + 1):
        for horizontal_height in range(MIN_TOWER_HEIGHT, MAX_TOWER_HEIGHT + 1):
            if branch == "vertical" and vertical_height > horizontal_height:
                pairs.append((vertical_height, horizontal_height))
            elif branch == "horizontal" and vertical_height < horizontal_height:
                pairs.append((vertical_height, horizontal_height))
            elif branch == "tie" and vertical_height == horizontal_height:
                pairs.append((vertical_height, horizontal_height))

    assert pairs
    return pairs[int(_RNG.integers(len(pairs)))]


def _sample_row(max_row: int, used_rows):
    candidates = [
        row
        for row in range(1, max_row + 1)
        if all(abs(int(row) - int(prev)) >= MIN_ROW_GAP for prev in used_rows)
    ]
    if not candidates:
        return None
    return int(_RNG.choice(candidates))


def _gen_attempt():
    tower_count = int(_RNG.choice(TOWER_COUNT_OPTIONS))
    min_width = max(GRID_W_RANGE[0], 3 + (tower_count - 1) * MIN_COL_GAP)
    max_width = GRID_W_RANGE[1]
    if min_width > max_width:
        return None

    branch = str(_RNG.choice(["vertical", "horizontal", "tie"]))
    pair_vertical_height, pair_horizontal_height = _sample_pair_heights(branch)

    heights = [int(_RNG.integers(MIN_TOWER_HEIGHT, MAX_TOWER_HEIGHT + 1)) for _ in range(tower_count)]
    width = int(_RNG.integers(min_width, max_width + 1))
    cols = _sample_spaced_columns(width, tower_count)
    if cols is None:
        return None

    pair_left = int(_RNG.integers(0, tower_count - 1))
    pair_right = pair_left + 1
    vertical_idx, horizontal_idx = (
        (pair_left, pair_right) if bool(_RNG.integers(2)) else (pair_right, pair_left)
    )
    heights[vertical_idx] = int(pair_vertical_height)
    heights[horizontal_idx] = int(pair_horizontal_height)

    min_height = max(
        GRID_H_RANGE[0],
        max(heights) + 3,
        2 + (tower_count - 1) * MIN_ROW_GAP,
    )
    max_height = GRID_H_RANGE[1]
    if min_height > max_height:
        return None
    height = int(_RNG.integers(min_height, max_height + 1))

    rows = [None] * tower_count
    max_row_horizontal = height - heights[horizontal_idx] - 1
    max_row_vertical = height - heights[vertical_idx] - 1
    upper_limit = min(max_row_horizontal, max_row_vertical - MIN_ROW_GAP)
    if upper_limit < 1:
        return None

    rows[horizontal_idx] = int(_RNG.integers(1, upper_limit + 1))
    vertical_low = rows[horizontal_idx] + MIN_ROW_GAP
    if vertical_low > max_row_vertical:
        return None
    rows[vertical_idx] = int(_RNG.integers(vertical_low, max_row_vertical + 1))

    used_rows = [rows[horizontal_idx], rows[vertical_idx]]
    other_indices = [idx for idx in range(tower_count) if idx not in (horizontal_idx, vertical_idx)]
    _RNG.shuffle(other_indices)
    for idx in other_indices:
        row = _sample_row(height - heights[idx] - 1, used_rows)
        if row is None:
            return None
        rows[idx] = row
        used_rows.append(row)

    colors = list(_RNG.choice(np.array(PALETTE, dtype=int), size=tower_count, replace=False))
    grid = np.full((height, width), BG, dtype=int)
    towers = []
    for idx in range(tower_count):
        marker_row = int(rows[idx])
        col = int(cols[idx])
        tower_height = int(heights[idx])
        color = int(colors[idx])
        grid[marker_row, col] = MARKER
        grid[marker_row + 1 : marker_row + tower_height + 1, col] = color
        towers.append(
            {
                "marker": (marker_row, col),
                "color": color,
                "height": tower_height,
                "top": marker_row + 1,
                "bottom": marker_row + tower_height,
                "col": col,
            }
        )

    out = _solve(grid)

    intersection = (int(rows[horizontal_idx]), int(cols[vertical_idx]))
    if int(grid[intersection]) != BG:
        return None

    expected = int(colors[vertical_idx]) if branch == "vertical" else int(colors[horizontal_idx])
    if int(out[intersection]) != expected:
        return None

    towers_by_row = sorted(towers, key=lambda t: int(t["marker"][0]))
    if len(towers_by_row) >= 3:
        rays = sum(
            len(_cast_ray(t["marker"][0], t["marker"][1], dr, dc, np.zeros_like(grid, dtype=bool)))
            for t in towers_by_row
            for dr, dc in ((-1, 0), (0, -1), (0, 1))
        )
        if rays <= 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            _ = main(grid)
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle input")
