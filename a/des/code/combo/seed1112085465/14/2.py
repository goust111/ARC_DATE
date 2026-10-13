from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
MARKER = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_col(g: np.ndarray) -> int:
    sep_cols = [c for c in range(g.shape[1]) if bool(np.all(g[:, c] == SEP))]
    assert len(sep_cols) == 1, "expected exactly one full-height grey separator"
    return int(sep_cols[0])


def _find_marker(panel: np.ndarray) -> tuple[np.ndarray, str]:
    candidates = []
    for coords in _components(panel == MARKER):
        if int(coords.shape[0]) != 2:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == r1 and c1 == c0 + 1:
            candidates.append((coords, "horizontal"))
        elif c0 == c1 and r1 == r0 + 1:
            candidates.append((coords, "vertical"))
    assert len(candidates) == 1, "expected one maroon two-pixel direction marker"
    return candidates[0]


def _extract_pattern(panel: np.ndarray, marker_coords: np.ndarray) -> np.ndarray:
    mask = panel != BG
    mask[marker_coords[:, 0], marker_coords[:, 1]] = False
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty source pattern"
    r0, c0, r1, c1 = _bbox(coords)
    return panel[r0 : r1 + 1, c0 : c1 + 1].copy()


def _read_table(panel: np.ndarray) -> list[tuple[int, int]]:
    rows = [r for r in range(panel.shape[0]) if bool(np.any(panel[r] != BG))]
    assert rows, "expected at least one mapping row"

    pairs: list[tuple[int, int]] = []
    table_cols: np.ndarray | None = None
    for r in rows:
        cols = np.flatnonzero(panel[r] != BG)
        assert len(cols) == 2, "each mapping row must contain exactly two colored cells"
        if table_cols is None:
            table_cols = cols
        else:
            assert np.array_equal(cols, table_cols), "mapping rows must align into two columns"
        pairs.append((int(panel[r, cols[0]]), int(panel[r, cols[1]])))

    left_colors = [a for a, _b in pairs]
    right_colors = [b for _a, b in pairs]
    assert len(set(left_colors)) == len(left_colors), "left mapping column must be unique"
    assert len(set(right_colors)) == len(right_colors), "right mapping column must be unique"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the source-panel colors using the right-hand lookup table, with the
    table direction chosen by the maroon two-pixel marker.

    Input structure
    - Background is black.
    - A full-height grey separator splits the grid into a left source panel and
      a right mapping table.
    - The left panel contains a colored pattern and a maroon two-pixel line
      marker that is either horizontal or vertical.
    - The right panel contains an Nx2 array of color pairs.

    Rule
    - Horizontal marker: read each table row left -> input, right -> output.
    - Vertical marker: read each table row right -> input, left -> output.
    - Apply the resulting color mapping to the source pattern.
    - Return only the transformed source pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_c = _find_separator_col(g)
    assert 0 < sep_c < g.shape[1] - 1, "separator must split the grid into two non-empty panels"

    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]
    marker_coords, orientation = _find_marker(left)
    pattern = _extract_pattern(left, marker_coords)
    pairs = _read_table(right)

    mapping: dict[int, int] = {}
    for left_color, right_color in pairs:
        src, dst = (left_color, right_color) if orientation == "horizontal" else (right_color, left_color)
        assert src not in mapping, "mapping inputs must be unique"
        mapping[int(src)] = int(dst)

    source_colors = {int(v) for v in np.unique(pattern) if int(v) != BG}
    assert source_colors <= set(mapping), "every source color must appear in the mapping table"

    out = pattern.copy()
    for src, dst in mapping.items():
        out[pattern == int(src)] = int(dst)
    return out


# --------------- generator (o5x) ---------------

COLOR_COUNT_RANGE = (3, 5)
PATTERN_SIDE_RANGE = (3, 7)
PATTERN_AREA_RANGE = (6, 18)
LEFT_EXTRA_W_RANGE = (4, 8)
RIGHT_W_RANGE = (4, 7)
HEIGHT_EXTRA_RANGE = (2, 6)
TABLE_ROW_STRIDES = (1, 2)
FRONTIER_COMPACT_PROB = 0.7
FRONTIER_TOP_K = 3


def _grow_shape(height: int, width: int, area: int) -> np.ndarray | None:
    if area < 1 or area > height * width:
        return None

    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                frontier.add((rr, cc))

    add_frontier(*start)
    while len(cells) < area:
        if not frontier:
            return None
        if random.random() < FRONTIER_COMPACT_PROB:
            ranked = sorted(
                frontier,
                key=lambda rc: sum((rc[0] + dr, rc[1] + dc) in cells for dr, dc in NEIGHBORS4),
                reverse=True,
            )
            pick = random.choice(ranked[: min(FRONTIER_TOP_K, len(ranked))])
        else:
            pick = random.choice(tuple(frontier))
        frontier.remove(pick)
        cells.add(pick)
        add_frontier(*pick)

    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    if min(mask.shape) < 2:
        return None
    return mask


def _shifted_pairs(colors: list[int]) -> list[tuple[int, int]]:
    n = len(colors)
    shifts = [k for k in range(1, n) if not (n % 2 == 0 and k == n // 2)]
    shift = random.choice(shifts)
    rotated = colors[shift:] + colors[:shift]
    return list(zip(colors, rotated))


def _make_pattern(source_colors: list[int]) -> np.ndarray | None:
    base_h = random.randint(*PATTERN_SIDE_RANGE)
    base_w = random.randint(*PATTERN_SIDE_RANGE)
    min_area = max(len(source_colors) + 1, PATTERN_AREA_RANGE[0])
    max_area = min(base_h * base_w, PATTERN_AREA_RANGE[1])
    if min_area > max_area:
        return None

    area = random.randint(min_area, max_area)
    mask = _grow_shape(base_h, base_w, area)
    if mask is None or int(mask.sum()) < len(source_colors):
        return None

    patch = np.zeros(mask.shape, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    for color, (r, c) in zip(source_colors, coords):
        patch[r, c] = int(color)
    for r, c in coords[len(source_colors) :]:
        patch[r, c] = int(random.choice(source_colors))
    return patch


def _marker_cells(anchor_r: int, anchor_c: int, orientation: str) -> list[tuple[int, int]]:
    if orientation == "horizontal":
        return [(anchor_r, anchor_c), (anchor_r, anchor_c + 1)]
    return [(anchor_r, anchor_c), (anchor_r + 1, anchor_c)]


def _place_marker(
    height: int,
    left_w: int,
    pat_r: int,
    pat_c: int,
    pat_h: int,
    pat_w: int,
    orientation: str,
) -> list[tuple[int, int]] | None:
    forbidden = np.zeros((height, left_w), dtype=bool)
    r0 = max(0, pat_r - 1)
    c0 = max(0, pat_c - 1)
    r1 = min(height, pat_r + pat_h + 1)
    c1 = min(left_w, pat_c + pat_w + 1)
    forbidden[r0:r1, c0:c1] = True

    candidates: list[list[tuple[int, int]]] = []
    row_limit = height if orientation == "horizontal" else height - 1
    col_limit = left_w - 1 if orientation == "horizontal" else left_w
    for r in range(row_limit):
        for c in range(col_limit):
            cells = _marker_cells(r, c, orientation)
            if all(not bool(forbidden[rr, cc]) for rr, cc in cells):
                candidates.append(cells)
    if not candidates:
        return None
    return random.choice(candidates)


def _apply_mapping(pattern: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = pattern.copy()
    for src, dst in mapping.items():
        out[pattern == int(src)] = int(dst)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(PALETTE, n_colors)
    pairs = _shifted_pairs(colors)
    orientation = random.choice(("horizontal", "vertical"))

    mapping = {
        int(src): int(dst)
        for src, dst in (pairs if orientation == "horizontal" else [(b, a) for a, b in pairs])
    }
    source_colors = list(mapping)

    pattern = _make_pattern(source_colors)
    if pattern is None:
        return None
    expected = _apply_mapping(pattern, mapping)

    pat_h, pat_w = map(int, pattern.shape)
    row_stride = random.choice(TABLE_ROW_STRIDES)
    table_span = 1 + (n_colors - 1) * row_stride

    min_height = max(pat_h + 2, table_span + 1)
    height = random.randint(min_height, min_height + HEIGHT_EXTRA_RANGE[1])
    left_w = random.randint(pat_w + LEFT_EXTRA_W_RANGE[0], pat_w + LEFT_EXTRA_W_RANGE[1])
    right_w = random.randint(*RIGHT_W_RANGE)
    if right_w < 2:
        return None

    pat_r = random.randint(0, height - pat_h)
    pat_c = random.randint(0, left_w - pat_w)
    marker = _place_marker(height, left_w, pat_r, pat_c, pat_h, pat_w, orientation)
    if marker is None:
        return None

    table_top = random.randint(0, height - table_span)
    table_c = random.randint(0, right_w - 2)

    grid = np.zeros((height, left_w + 1 + right_w), dtype=int)
    grid[:, left_w] = SEP
    grid[pat_r : pat_r + pat_h, pat_c : pat_c + pat_w] = pattern
    for rr, cc in marker:
        grid[rr, cc] = MARKER
    for i, (left_color, right_color) in enumerate(pairs):
        r = table_top + i * row_stride
        grid[r, left_w + 1 + table_c] = int(left_color)
        grid[r, left_w + 1 + table_c + 1] = int(right_color)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
