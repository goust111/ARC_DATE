import random
from collections import deque

import numpy as np


GREY = 5
MAX_RETRIES = 400
PANEL_ROWS_CHOICES = (2, 3)
PANEL_COLS_CHOICES = (2, 3, 4)
PANEL_HEIGHT_CHOICES = (10, 12, 14)
PANEL_WIDTH_CHOICES = (10, 12, 14)
MIN_OBJECT_SIZE = 4
COLORS = tuple(range(10))
CORNER_NAMES = ("tl", "tr", "bl", "br")
CORNER_SLICES = {
    "tl": (slice(0, 2), slice(0, 2)),
    "tr": (slice(0, 2), slice(-2, None)),
    "bl": (slice(-2, None), slice(0, 2)),
    "br": (slice(-2, None), slice(-2, None)),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    start = tuple(map(int, points[0]))
    seen = {start}
    queue = deque([start])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(mask.sum())


def _quadrants_nonempty(mask: np.ndarray) -> bool:
    h, w = mask.shape
    mid_r = h // 2
    mid_c = w // 2
    return all(
        (
            bool(mask[:mid_r, :mid_c].any()),
            bool(mask[:mid_r, mid_c:].any()),
            bool(mask[mid_r:, :mid_c].any()),
            bool(mask[mid_r:, mid_c:].any()),
        )
    )


def _make_object_mask(rng: random.Random, height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    min_cells = max(8, (height * width) // 2)
    target_removals = rng.randint(0, max(1, (height * width) // 3))
    removed = 0
    attempts = 0

    while removed < target_removals and attempts < 200:
        attempts += 1
        cells = np.argwhere(mask)
        r, c = map(int, cells[rng.randrange(len(cells))])
        candidate = mask.copy()
        candidate[r, c] = False
        if int(candidate.sum()) < min_cells:
            continue
        if not candidate[0, :].any() or not candidate[-1, :].any():
            continue
        if not candidate[:, 0].any() or not candidate[:, -1].any():
            continue
        if not _quadrants_nonempty(candidate):
            continue
        if not _connected(candidate):
            continue
        mask = candidate
        removed += 1

    return mask


def _line_ranges(indices: list[int], size: int) -> list[tuple[int, int]]:
    assert indices
    ordered = sorted(set(int(v) for v in indices))
    runs: list[tuple[int, int]] = []
    start = ordered[0]
    prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))

    ranges: list[tuple[int, int]] = []
    if runs[0][0] > 0:
        ranges.append((0, runs[0][0]))
    for (_, end_a), (start_b, _) in zip(runs, runs[1:]):
        if end_a + 1 < start_b:
            ranges.append((end_a + 1, start_b))
    if runs[-1][1] < size - 1:
        ranges.append((runs[-1][1] + 1, size))
    return ranges


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    best = None
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(rows) < 2 or len(cols) < 2:
            continue
        score = len(rows) + len(cols)
        if best is None or score > best[0]:
            best = (score, color, rows, cols)
    assert best is not None, "could not locate panel separators"
    return int(best[1]), best[2], best[3]


def _corner_color(panel: np.ndarray, name: str, bg: int, sep: int) -> int | None:
    rs, cs = CORNER_SLICES[name]
    block = panel[rs, cs]
    values = set(map(int, np.unique(block)))
    if len(values) != 1:
        return None
    color = next(iter(values))
    if color in {bg, sep}:
        return None
    return color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    sep, sep_rows, sep_cols = _find_separator(g, bg)
    row_ranges = _line_ranges(sep_rows, g.shape[0])
    col_ranges = _line_ranges(sep_cols, g.shape[1])

    assert row_ranges and col_ranges
    panel_heights = {r1 - r0 for r0, r1 in row_ranges}
    panel_widths = {c1 - c0 for c0, c1 in col_ranges}
    assert len(panel_heights) == 1 and len(panel_widths) == 1

    out = g.copy()
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            panel = g[r0:r1, c0:c1]
            corner_colors = {
                name: _corner_color(panel, name, bg, sep) for name in CORNER_NAMES
            }
            legend_mask = np.zeros(panel.shape, dtype=bool)
            for name, color in corner_colors.items():
                if color is None:
                    continue
                rs, cs = CORNER_SLICES[name]
                legend_mask[rs, cs] = True

            object_mask = (~legend_mask) & (panel != bg) & (panel != sep)
            assert object_mask.any(), "each panel must contain one foreground object"
            coords = np.argwhere(object_mask)
            rmin, cmin = map(int, coords.min(axis=0))
            rmax, cmax = map(int, coords.max(axis=0))
            mid_r = (rmin + rmax + 1) // 2
            mid_c = (cmin + cmax + 1) // 2

            repainted = panel.copy()
            for rr, cc in coords:
                rr = int(rr)
                cc = int(cc)
                if rr < mid_r:
                    corner = "tl" if cc < mid_c else "tr"
                else:
                    corner = "bl" if cc < mid_c else "br"
                color = corner_colors[corner]
                if color is not None:
                    repainted[rr, cc] = color

            out[r0:r1, c0:c1] = repainted

    return out


def _build_panel(
    rng: random.Random,
    height: int,
    width: int,
    bg: int,
    corner_set: set[str],
    sep: int,
    legend_palette: list[int],
) -> np.ndarray:
    panel = np.full((height, width), bg, dtype=int)
    color_iter = iter(legend_palette)
    for name in CORNER_NAMES:
        if name not in corner_set:
            continue
        color = next(color_iter)
        rs, cs = CORNER_SLICES[name]
        panel[rs, cs] = color

    max_h = min(height - 4, 8)
    max_w = min(width - 4, 8)
    obj_h = rng.choice([v for v in range(MIN_OBJECT_SIZE, max_h + 1, 2)])
    obj_w = rng.choice([v for v in range(MIN_OBJECT_SIZE, max_w + 1, 2)])
    top = rng.randint(2, height - obj_h - 2)
    left = rng.randint(2, width - obj_w - 2)
    mask = _make_object_mask(rng, obj_h, obj_w)
    panel[top : top + obj_h, left : left + obj_w][mask] = GREY

    assert not np.any(panel == sep)
    return panel


def _generate_case(rng: random.Random) -> np.ndarray | None:
    bg_choices = [c for c in COLORS if c != GREY]
    bg = rng.choice(bg_choices)
    sep_choices = [c for c in COLORS if c not in {bg, GREY}]
    sep = rng.choice(sep_choices)

    panel_rows = rng.choice(PANEL_ROWS_CHOICES)
    panel_cols = rng.choice(PANEL_COLS_CHOICES)
    panel_h = rng.choice(PANEL_HEIGHT_CHOICES)
    panel_w = rng.choice(PANEL_WIDTH_CHOICES)

    total_h = panel_rows * panel_h + panel_rows + 1
    total_w = panel_cols * panel_w + panel_cols + 1
    grid = np.full((total_h, total_w), sep, dtype=int)

    corner_sets: list[set[str]] = []
    for _ in range(panel_rows * panel_cols):
        present = {name for name in CORNER_NAMES if rng.random() < 0.5}
        if not present:
            present = {rng.choice(CORNER_NAMES)}
        corner_sets.append(present)

    present_union = set().union(*corner_sets)
    missing_union: set[str] = set()
    for present in corner_sets:
        missing_union |= set(CORNER_NAMES) - present
    if present_union != set(CORNER_NAMES):
        return None
    if not missing_union:
        return None

    palette = [c for c in COLORS if c not in {bg, sep, GREY}]
    panel_index = 0
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            r0 = 1 + pr * (panel_h + 1)
            c0 = 1 + pc * (panel_w + 1)
            present = corner_sets[panel_index]
            panel_index += 1
            colors = rng.sample(palette, k=len(present))
            panel = _build_panel(rng, panel_h, panel_w, bg, present, sep, colors)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
