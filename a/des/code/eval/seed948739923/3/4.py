from __future__ import annotations

import random
from typing import Iterable

import numpy as np


SEED_COLOR = 2
GENERATOR_SEPARATOR = 3
GENERATION_RETRIES = 300


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _interior_spans(lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    best: tuple[tuple[int, int, int], int, list[tuple[int, int]], list[tuple[int, int]]] | None = None

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(rows) < 2 or len(cols) < 2:
            continue
        if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
            continue
        row_spans = _interior_spans(rows)
        col_spans = _interior_spans(cols)
        if not row_spans or not col_spans:
            continue
        if len({r1 - r0 + 1 for r0, r1 in row_spans}) != 1:
            continue
        if len({c1 - c0 + 1 for c0, c1 in col_spans}) != 1:
            continue
        score = (len(row_spans) * len(col_spans), len(rows) + len(cols), -color)
        if best is None or score > best[0]:
            best = (score, color, row_spans, col_spans)

    assert best is not None, "failed to find separator grid"
    return best[1], best[2], best[3]


def _crop_non_bg(patch: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(patch != bg)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _round_half_up(values: np.ndarray) -> int:
    return int(np.floor(float(values.mean()) + 0.5))


def _template_like(cell: np.ndarray, bg: int, seed: int) -> bool:
    return any(int(v) not in {bg, seed} for v in np.unique(cell))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pick a legend template by seed count, then center that template on the seed centroid inside each boxed cell.

    Input structure:
    - Background plus a 1-pixel separator grid partitioning the canvas into equal cells.
    - One top row or left column of cells is a legend strip containing numbered templates in order.
    - Every non-legend cell contains 1..N red seed pixels, where N is the number of legend templates.

    Rule:
    - Count the red seeds in a cell to choose legend template N.
    - Compute the rough visual center of those seeds.
    - Place the chosen template centered on that point, shifting inward if needed to fit the cell.
    - Keep separator lines and the legend strip unchanged; clear the rest of each data cell to background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _mode_color(g)
    _sep, row_spans, col_spans = _find_separator(g, bg)
    seed = SEED_COLOR if np.any(g == SEED_COLOR) else SEED_COLOR

    cell_rows = len(row_spans)
    cell_cols = len(col_spans)
    cells = [[g[r0 : r1 + 1, c0 : c1 + 1] for c0, c1 in col_spans] for r0, r1 in row_spans]

    template_mask = np.zeros((cell_rows, cell_cols), dtype=bool)
    for i in range(cell_rows):
        for j in range(cell_cols):
            template_mask[i, j] = _template_like(cells[i][j], bg, seed)

    row_candidates = [i for i in range(cell_rows) if bool(template_mask[i].all())]
    col_candidates = [j for j in range(cell_cols) if bool(template_mask[:, j].all())]
    template_total = int(template_mask.sum())

    orientation: str
    legend_index: int
    if template_total == cell_cols and len(row_candidates) == 1:
        orientation = "top"
        legend_index = row_candidates[0]
    elif template_total == cell_rows and len(col_candidates) == 1:
        orientation = "left"
        legend_index = col_candidates[0]
    else:
        raise AssertionError("failed to identify legend strip")

    templates: list[np.ndarray] = []
    if orientation == "top":
        for j in range(cell_cols):
            templates.append(_crop_non_bg(cells[legend_index][j], bg))
    else:
        for i in range(cell_rows):
            templates.append(_crop_non_bg(cells[i][legend_index], bg))

    assert len({tpl.shape for tpl in templates}) == 1, "templates must have equal size"
    tpl_h, tpl_w = templates[0].shape

    out = g.copy()
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            if orientation == "top" and i == legend_index:
                continue
            if orientation == "left" and j == legend_index:
                continue

            cell = cells[i][j]
            out[r0 : r1 + 1, c0 : c1 + 1] = bg

            seeds = np.argwhere(cell == seed)
            if len(seeds) == 0:
                continue
            assert 1 <= len(seeds) <= len(templates)

            center_r = _round_half_up(seeds[:, 0])
            center_c = _round_half_up(seeds[:, 1])
            top = min(max(center_r - tpl_h // 2, 0), cell.shape[0] - tpl_h)
            left = min(max(center_c - tpl_w // 2, 0), cell.shape[1] - tpl_w)
            out[r0 + top : r0 + top + tpl_h, c0 + left : c0 + left + tpl_w] = templates[len(seeds) - 1]

    return out


# --------------- generator ---------------


def _odd_sizes(limit_exclusive: int) -> list[int]:
    return [n for n in range(3, limit_exclusive, 2)]


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    rr, cc = b
    pts = [(r, c)]
    while c != cc:
        c += 1 if cc > c else -1
        pts.append((r, c))
    while r != rr:
        r += 1 if rr > r else -1
        pts.append((r, c))
    return pts


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _random_template(rng: random.Random, h: int, w: int, bg: int, colors: list[int]) -> np.ndarray:
    center = (h // 2, w // 2)
    top = (0, rng.randrange(w))
    bottom = (h - 1, rng.randrange(w))
    left = (rng.randrange(h), 0)
    right = (rng.randrange(h), w - 1)

    mask: set[tuple[int, int]] = set()
    for endpoint in (top, bottom, left, right):
        mask.update(_manhattan_path(center, endpoint))

    target_area = rng.randint(max(5, h + w - 3), min(h * w, h * w - 1))
    while len(mask) < target_area:
        frontier = []
        for r, c in list(mask):
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in mask:
                    frontier.append((nr, nc))
        if not frontier:
            break
        mask.add(rng.choice(frontier))

    arr = np.full((h, w), bg, dtype=int)
    palette = rng.sample(colors, k=rng.randint(2, min(3, len(colors))))
    for r, c in mask:
        arr[r, c] = rng.choice(palette)
    if len({int(arr[r, c]) for r, c in mask}) == 1:
        r, c = rng.choice(list(mask))
        arr[r, c] = palette[-1]
    return arr


def _would_clamp(center_r: int, center_c: int, cell_h: int, cell_w: int, tpl_h: int, tpl_w: int) -> bool:
    top = center_r - tpl_h // 2
    left = center_c - tpl_w // 2
    return top < 0 or left < 0 or top + tpl_h > cell_h or left + tpl_w > cell_w


def _candidate_centers(cell_h: int, cell_w: int, tpl_h: int, tpl_w: int, want_clamp: bool) -> list[tuple[int, int]]:
    centers = []
    for r in range(cell_h):
        for c in range(cell_w):
            if _would_clamp(r, c, cell_h, cell_w, tpl_h, tpl_w) == want_clamp:
                centers.append((r, c))
    return centers


def _place_seeds(
    rng: random.Random,
    cell_h: int,
    cell_w: int,
    count: int,
    tpl_h: int,
    tpl_w: int,
    want_clamp: bool,
) -> list[tuple[int, int]]:
    centers = _candidate_centers(cell_h, cell_w, tpl_h, tpl_w, want_clamp)
    assert centers
    center_r, center_c = rng.choice(centers)

    for radius in (1, 2):
        pool = [
            (r, c)
            for r in range(max(0, center_r - radius), min(cell_h, center_r + radius + 1))
            for c in range(max(0, center_c - radius), min(cell_w, center_c + radius + 1))
        ]
        for _ in range(200):
            order = pool[:]
            rng.shuffle(order)
            order.sort(key=lambda p: (abs(p[0] - center_r) + abs(p[1] - center_c), rng.random()))
            pts = order[:count]
            rows = np.array([r for r, _ in pts])
            cols = np.array([c for _, c in pts])
            mean_r = _round_half_up(rows)
            mean_c = _round_half_up(cols)
            if _would_clamp(mean_r, mean_c, cell_h, cell_w, tpl_h, tpl_w) == want_clamp:
                return pts

    ordered = sorted(
        [(r, c) for r in range(cell_h) for c in range(cell_w)],
        key=lambda p: (abs(p[0] - center_r) + abs(p[1] - center_c), p[0], p[1]),
    )
    fallback = ordered[:count]
    return fallback


def _grid_shape_ok(total_rows: int, total_cols: int, cell_h: int, cell_w: int) -> bool:
    h = total_rows * cell_h + total_rows + 1
    w = total_cols * cell_w + total_cols + 1
    return 15 <= h <= 30 and 15 <= w <= 30


def _gen_attempt(rng: random.Random, orientation: str) -> np.ndarray | None:
    template_count = rng.randint(2, 4)

    if orientation == "top":
        data_rows = rng.randint(2, 4)
        data_cols = template_count
        total_rows = data_rows + 1
        total_cols = data_cols
    else:
        data_rows = template_count
        data_cols = rng.randint(2, 4)
        total_rows = data_rows
        total_cols = data_cols + 1

    size_options = [(h, w) for h in (5, 6, 7) for w in (5, 6, 7) if _grid_shape_ok(total_rows, total_cols, h, w)]
    if not size_options:
        return None
    cell_h, cell_w = rng.choice(size_options)

    tpl_h_choices = _odd_sizes(cell_h)
    tpl_w_choices = _odd_sizes(cell_w)
    if not tpl_h_choices or not tpl_w_choices:
        return None
    tpl_h = rng.choice(tpl_h_choices)
    tpl_w = rng.choice(tpl_w_choices)

    bg = rng.choice([0, 1, 5, 6])
    sep = GENERATOR_SEPARATOR
    color_pool = [c for c in range(10) if c not in {bg, SEED_COLOR, sep}]
    rng.shuffle(color_pool)
    template_colors = color_pool[: max(4, min(6, len(color_pool)))]

    templates: list[np.ndarray] = []
    seen_templates: set[bytes] = set()
    for _ in range(template_count):
        for _ in range(100):
            tpl = _random_template(rng, tpl_h, tpl_w, bg, template_colors)
            sig = tpl.tobytes()
            if sig not in seen_templates:
                seen_templates.add(sig)
                templates.append(tpl)
                break
        else:
            return None

    h = total_rows * cell_h + total_rows + 1
    w = total_cols * cell_w + total_cols + 1
    grid = np.full((h, w), bg, dtype=int)

    sep_rows = [r * (cell_h + 1) for r in range(total_rows + 1)]
    sep_cols = [c * (cell_w + 1) for c in range(total_cols + 1)]
    grid[sep_rows, :] = sep
    grid[:, sep_cols] = sep

    row_spans = [(sep_rows[i] + 1, sep_rows[i + 1] - 1) for i in range(total_rows)]
    col_spans = [(sep_cols[j] + 1, sep_cols[j + 1] - 1) for j in range(total_cols)]

    if orientation == "top":
        legend_cells = [(0, j) for j in range(total_cols)]
        data_cells = [(i, j) for i in range(1, total_rows) for j in range(total_cols)]
    else:
        legend_cells = [(i, 0) for i in range(total_rows)]
        data_cells = [(i, j) for i in range(total_rows) for j in range(1, total_cols)]

    for idx, (i, j) in enumerate(legend_cells):
        r0, r1 = row_spans[i]
        c0, c1 = col_spans[j]
        y = rng.randint(0, cell_h - tpl_h)
        x = rng.randint(0, cell_w - tpl_w)
        grid[r0 + y : r0 + y + tpl_h, c0 + x : c0 + x + tpl_w] = templates[idx]

    counts = [1 + (k % template_count) for k in range(len(data_cells))]
    rng.shuffle(counts)
    clamp_flags = [True] * (len(data_cells) // 2) + [False] * (len(data_cells) - len(data_cells) // 2)
    rng.shuffle(clamp_flags)

    has_clamped = False
    has_unclamped = False
    for (i, j), count, want_clamp in zip(data_cells, counts, clamp_flags):
        r0, r1 = row_spans[i]
        c0, c1 = col_spans[j]
        pts = _place_seeds(rng, cell_h, cell_w, count, tpl_h, tpl_w, want_clamp)
        rows = np.array([r for r, _ in pts])
        cols = np.array([c for _, c in pts])
        clamped = _would_clamp(_round_half_up(rows), _round_half_up(cols), cell_h, cell_w, tpl_h, tpl_w)
        has_clamped |= clamped
        has_unclamped |= not clamped
        for rr, cc in pts:
            grid[r0 + rr, c0 + cc] = SEED_COLOR

    if not has_clamped or not has_unclamped:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    preferred = rng.choice(["top", "left"])
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng, preferred)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
