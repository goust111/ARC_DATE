from __future__ import annotations

import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_ORDER = ("up", "right", "down", "left")
PALETTE = tuple(range(10))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _intervals_from_separator_lines(lines: list[int], size: int) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(set(int(x) for x in lines)) + [int(size)]
    intervals: list[tuple[int, int]] = []
    for lo, hi in zip(bounds, bounds[1:]):
        start = int(lo + 1)
        end = int(hi - 1)
        if start <= end:
            intervals.append((start, end))
    return intervals


def _find_separator_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_lines: dict[int, list[int]] = {}
    col_lines: dict[int, list[int]] = {}

    for r in range(h):
        row = g[r]
        if np.all(row == row[0]):
            row_lines.setdefault(int(row[0]), []).append(int(r))
    for c in range(w):
        col = g[:, c]
        if np.all(col == col[0]):
            col_lines.setdefault(int(col[0]), []).append(int(c))

    candidates = set(row_lines) & set(col_lines)
    assert candidates, "expected a separator color with full rows and full columns"

    best: tuple[tuple[int, int, int], int, list[tuple[int, int]], list[tuple[int, int]]] | None = None
    for color in candidates:
        row_ranges = _intervals_from_separator_lines(row_lines[color], h)
        col_ranges = _intervals_from_separator_lines(col_lines[color], w)
        if not row_ranges or not col_ranges:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_ranges}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_ranges}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        score = (
            int(len(row_ranges) * len(col_ranges)),
            int(len(row_lines[color]) + len(col_lines[color])),
            int(len(row_lines[color]) * len(col_lines[color])),
        )
        candidate = (score, int(color), row_ranges, col_ranges)
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None, "failed to infer equal-sized panels from separator lines"
    _score, color, row_ranges, col_ranges = best
    return int(color), row_ranges, col_ranges


def _stamp_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = mask.shape
    region = canvas[top : top + rows, left : left + cols]
    region[mask] = int(color)


def _solve_panel(panel: np.ndarray, bg: int) -> np.ndarray:
    sub = np.asarray(panel, dtype=int)
    ph, pw = sub.shape

    colors = sorted(set(map(int, sub.ravel())) - {int(bg)})
    assert len(colors) == 1, "each panel must contain exactly one non-background color"
    color = int(colors[0])
    mask = sub == color

    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    seed_mask = mask & ~border
    assert seed_mask.any(), "seed pattern must lie strictly inside the panel"

    sr0, sc0, sr1, sc1 = _bbox(seed_mask)
    seed_patch = seed_mask[sr0 : sr1 + 1, sc0 : sc1 + 1]
    seed_h = int(sr1 - sr0 + 1)
    seed_w = int(sc1 - sc0 + 1)

    out = np.full_like(sub, int(bg))
    _stamp_mask(out, seed_patch, sr0, sc0, color)

    if np.any(mask[0, :]):
        step = 1
        while sr0 - step * seed_h >= 0:
            _stamp_mask(out, seed_patch, sr0 - step * seed_h, sc0, color)
            step += 1
    if np.any(mask[-1, :]):
        step = 1
        while sr1 + step * seed_h < ph:
            _stamp_mask(out, seed_patch, sr0 + step * seed_h, sc0, color)
            step += 1
    if np.any(mask[:, 0]):
        step = 1
        while sc0 - step * seed_w >= 0:
            _stamp_mask(out, seed_patch, sr0, sc0 - step * seed_w, color)
            step += 1
    if np.any(mask[:, -1]):
        step = 1
        while sc1 + step * seed_w < pw:
            _stamp_mask(out, seed_patch, sr0, sc0 + step * seed_w, color)
            step += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    In equal-sized separator-defined panels, replicate each seed pattern in the
    directions indicated by same-colored edge markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator, row_ranges, col_ranges = _find_separator_structure(g)
    bg = _most_common_color(g[g != separator])
    out = g.copy()

    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            out[r0 : r1 + 1, c0 : c1 + 1] = _solve_panel(g[r0 : r1 + 1, c0 : c1 + 1], bg)

    return out


# --------------- generator ---------------

PANEL_GRID_OPTIONS = ((2, 2), (2, 3), (3, 2))
PANEL_SPAN_OPTIONS = (6, 8, 9, 10, 12)
MAX_GRID_SIDE = 30
MAX_GENERATION_ATTEMPTS = 500


def _seed_spans_for(panel_span: int) -> list[int]:
    return [
        span
        for span in range(2, panel_span)
        if panel_span % span == 0 and 3 <= panel_span // span <= 5
    ]


LAYOUT_OPTIONS = [
    (panel_rows, panel_cols, panel_h, panel_w)
    for panel_rows, panel_cols in PANEL_GRID_OPTIONS
    for panel_h in PANEL_SPAN_OPTIONS
    for panel_w in PANEL_SPAN_OPTIONS
    if _seed_spans_for(panel_h)
    and _seed_spans_for(panel_w)
    and panel_rows * panel_h + panel_rows + 1 <= MAX_GRID_SIDE
    and panel_cols * panel_w + panel_cols + 1 <= MAX_GRID_SIDE
]


def _choose_primary_directions(panel_count: int) -> list[str]:
    counts = {direction: 0 for direction in DIR_ORDER}
    primary = list(DIR_ORDER)
    for direction in primary:
        counts[direction] += 1
    while len(primary) < panel_count:
        min_count = min(counts.values())
        options = [direction for direction, count in counts.items() if count == min_count]
        picked = random.choice(options)
        primary.append(picked)
        counts[picked] += 1
    random.shuffle(primary)
    return primary


def _make_seed_mask(height: int, width: int) -> np.ndarray | None:
    area = height * width
    min_cells = max(3, (area + 2) // 3)
    max_cells = max(min_cells, (2 * area + 2) // 3)

    for _ in range(200):
        target = random.randint(min_cells, min(area, max_cells))
        mask = np.zeros((height, width), dtype=bool)
        start = (random.randrange(height), random.randrange(width))
        mask[start] = True

        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                for dr, dc in DIRS4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break

            frontier = list(dict.fromkeys(frontier))
            weights = []
            for r, c in frontier:
                weight = 1
                if r in (0, height - 1):
                    weight += 3
                if c in (0, width - 1):
                    weight += 3
                weights.append(weight)
            nr, nc = random.choices(frontier, weights=weights, k=1)[0]
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue
        if not mask[0, :].any():
            continue
        if not mask[-1, :].any():
            continue
        if not mask[:, 0].any():
            continue
        if not mask[:, -1].any():
            continue
        if mask.all() and random.random() < 0.8:
            continue
        return mask

    return None


def _paint_seed(panel: np.ndarray, seed_mask: np.ndarray, top: int, left: int, color: int) -> None:
    region = panel[top : top + seed_mask.shape[0], left : left + seed_mask.shape[1]]
    region[seed_mask] = int(color)


def _marker_positions(
    seed_mask: np.ndarray,
    panel_h: int,
    panel_w: int,
    seed_top: int,
    seed_left: int,
    directions: set[str],
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []

    if "up" in directions:
        cols = [int(c) for c in np.flatnonzero(seed_mask[0, :])]
        marker_col = seed_left + random.choice(cols)
        positions.append((0, marker_col))
    if "down" in directions:
        cols = [int(c) for c in np.flatnonzero(seed_mask[-1, :])]
        marker_col = seed_left + random.choice(cols)
        positions.append((panel_h - 1, marker_col))
    if "left" in directions:
        rows = [int(r) for r in np.flatnonzero(seed_mask[:, 0])]
        marker_row = seed_top + random.choice(rows)
        positions.append((marker_row, 0))
    if "right" in directions:
        rows = [int(r) for r in np.flatnonzero(seed_mask[:, -1])]
        marker_row = seed_top + random.choice(rows)
        positions.append((marker_row, panel_w - 1))

    assert len(set(positions)) == len(positions), "direction markers must occupy distinct border cells"
    return positions


def _build_panel(panel_h: int, panel_w: int, bg: int, color: int, primary_dir: str) -> tuple[np.ndarray, np.ndarray]:
    seed_h = random.choice(_seed_spans_for(panel_h))
    seed_w = random.choice(_seed_spans_for(panel_w))
    tile_rows = panel_h // seed_h
    tile_cols = panel_w // seed_w
    assert tile_rows >= 3 and tile_cols >= 3

    seed_mask = _make_seed_mask(seed_h, seed_w)
    if seed_mask is None:
        raise RuntimeError("failed to sample a non-degenerate seed mask")

    seed_row = random.randint(1, tile_rows - 2)
    seed_col = random.randint(1, tile_cols - 2)
    seed_top = seed_row * seed_h
    seed_left = seed_col * seed_w

    size = random.randint(1, 4)
    others = [direction for direction in DIR_ORDER if direction != primary_dir]
    directions = {primary_dir, *random.sample(others, size - 1)}

    panel = np.full((panel_h, panel_w), int(bg), dtype=int)
    _paint_seed(panel, seed_mask, seed_top, seed_left, color)
    markers = _marker_positions(seed_mask, panel_h, panel_w, seed_top, seed_left, directions)
    for r, c in markers:
        panel[r, c] = int(color)

    solved = _solve_panel(panel, bg)
    for r, c in markers:
        assert int(solved[r, c]) == int(color), "markers must be covered by grown copies"
    return panel, solved


def _dominant_non_separator_color(grid: np.ndarray, separator: int) -> int:
    return _most_common_color(grid[grid != separator])


def _gen_attempt() -> np.ndarray | None:
    panel_rows, panel_cols, panel_h, panel_w = random.choice(LAYOUT_OPTIONS)
    height = panel_rows * panel_h + panel_rows + 1
    width = panel_cols * panel_w + panel_cols + 1

    separator, bg = random.sample(PALETTE, 2)
    panel_count = panel_rows * panel_cols
    panel_colors = random.sample([color for color in PALETTE if color not in {separator, bg}], panel_count)
    primary_dirs = _choose_primary_directions(panel_count)

    inp = np.full((height, width), int(separator), dtype=int)
    expected = np.full((height, width), int(separator), dtype=int)

    for idx in range(panel_count):
        pr = idx // panel_cols
        pc = idx % panel_cols
        r0 = 1 + pr * (panel_h + 1)
        c0 = 1 + pc * (panel_w + 1)
        panel, solved = _build_panel(panel_h, panel_w, bg, panel_colors[idx], primary_dirs[idx])
        inp[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
        expected[r0 : r0 + panel_h, c0 : c0 + panel_w] = solved

    if _dominant_non_separator_color(inp, separator) != bg:
        return None
    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-growth ARC input")
