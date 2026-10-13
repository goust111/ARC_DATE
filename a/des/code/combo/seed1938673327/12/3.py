from __future__ import annotations

import numpy as np


def _panel_spans(lines: list[int], total: int) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(lines), total]
    spans: list[tuple[int, int]] = []
    for i in range(len(bounds) - 1):
        start = bounds[i] + 1
        end = bounds[i + 1]
        if start >= end:
            return []
        spans.append((start, end))
    return spans


def _find_layout(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    h, w = g.shape
    candidates: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], int]] = []

    for color in map(int, np.unique(g)):
        sep_rows = [r for r in range(h) if np.all(g[r, :] == color)]
        sep_cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not sep_rows or not sep_cols:
            continue

        row_spans = _panel_spans(sep_rows, h)
        col_spans = _panel_spans(sep_cols, w)
        if not row_spans or not col_spans:
            continue

        row_sizes = {r1 - r0 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 for c0, c1 in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue

        side = next(iter(row_sizes))
        if side < 2 or side != next(iter(col_sizes)):
            continue

        score = len(sep_rows) + len(sep_cols)
        candidates.append((score, color, row_spans, col_spans, side))

    assert candidates, "failed to find separator layout"
    best_score = max(score for score, *_ in candidates)
    best = [item for item in candidates if item[0] == best_score]
    assert len(best) == 1, "separator layout must be unambiguous"
    _, sep_color, row_spans, col_spans, side = best[0]
    return sep_color, row_spans, col_spans, side


def _infer_bg(
    g: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    sep_color: int,
) -> int:
    corners: list[int] = []
    panel_cells: list[np.ndarray] = []

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panel = g[r0:r1, c0:c1]
            corners.extend(
                [
                    int(panel[0, 0]),
                    int(panel[0, -1]),
                    int(panel[-1, 0]),
                    int(panel[-1, -1]),
                ]
            )
            panel_cells.append(panel.reshape(-1))

    vals, counts = np.unique(np.array(corners, dtype=int), return_counts=True)
    order = np.argsort(-counts)
    if len(order) > 0 and (len(order) == 1 or counts[order[0]] > counts[order[1]]):
        bg = int(vals[order[0]])
        if bg != sep_color:
            return bg

    vals, counts = np.unique(np.concatenate(panel_cells), return_counts=True)
    mask = vals != sep_color
    vals = vals[mask]
    counts = counts[mask]
    assert len(vals) > 0
    return int(vals[np.argmax(counts)])


def _scale_up_panel(panel: np.ndarray) -> np.ndarray:
    side = panel.shape[0]
    enlarged = np.repeat(np.repeat(panel, 2, axis=0), 2, axis=1)
    return enlarged[:side, :side].copy()


def _scale_down_panel(panel: np.ndarray, bg: int) -> np.ndarray:
    side = panel.shape[0]
    limit = side - (side % 2)
    if limit == 0:
        reduced = panel[:1, :1].copy()
    else:
        reduced = panel[:limit:2, :limit:2].copy()

    out = np.full(panel.shape, bg, dtype=int)
    rh, rw = reduced.shape
    r0 = (side - rh) // 2
    c0 = (side - rw) // 2
    out[r0 : r0 + rh, c0 : c0 + rw] = reduced
    return out


def _transform_panel(panel: np.ndarray, bg: int) -> np.ndarray:
    fill_ratio = float(np.count_nonzero(panel != bg)) / float(panel.size)
    if fill_ratio < 0.3:
        return _scale_up_panel(panel)
    if fill_ratio > 0.7:
        return _scale_down_panel(panel, bg)
    return panel.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """Resize each separator-delimited panel according to how full its foreground pattern is.

    Sparse panels are enlarged 2x and clipped back to the panel size. Dense panels
    are reduced by taking the top-left cell of each 2x2 block, then centering the
    smaller result inside a background-filled panel. Medium-density panels stay as-is.
    """
    g = np.asarray(grid, dtype=int)
    sep_color, row_spans, col_spans, _side = _find_layout(g)
    bg = _infer_bg(g, row_spans, col_spans, sep_color)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panel = g[r0:r1, c0:c1]
            out[r0:r1, c0:c1] = _transform_panel(panel, bg)
    return out


# --------------- generator ---------------

_MAX_ATTEMPTS = 300
_SIDE_CHOICES = (6, 8, 10)
_BRANCHES = ("sparse", "medium", "dense")


def _neighbors4(r: int, c: int, side: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < side and 0 <= cc < side:
            out.append((rr, cc))
    return out


def _fill_count(branch: str, side: int, rng: np.random.Generator) -> int:
    total = side * side
    if branch == "sparse":
        lo = max(3, int(np.ceil(total * 0.08)))
        hi = max(lo, int(np.floor(total * 0.24)))
    elif branch == "medium":
        lo = int(np.ceil(total * 0.40))
        hi = int(np.floor(total * 0.60))
    else:
        lo = int(np.ceil(total * 0.78))
        hi = min(total - 4, int(np.floor(total * 0.88)))
    return int(rng.integers(lo, hi + 1))


def _grow_mask(side: int, count: int, rng: np.random.Generator) -> np.ndarray:
    corners = {
        (0, 0),
        (0, side - 1),
        (side - 1, 0),
        (side - 1, side - 1),
    }
    available = [(r, c) for r in range(side) for c in range(side) if (r, c) not in corners]
    assert count <= len(available)

    start = available[int(rng.integers(len(available)))]
    mask = np.zeros((side, side), dtype=bool)
    mask[start] = True
    frontier = {
        cell
        for cell in _neighbors4(start[0], start[1], side)
        if cell not in corners
    }

    while int(mask.sum()) < count:
        pick = list(frontier)[int(rng.integers(len(frontier)))]
        frontier.remove(pick)
        if mask[pick]:
            continue
        mask[pick] = True
        for neighbor in _neighbors4(pick[0], pick[1], side):
            if neighbor not in corners and not mask[neighbor]:
                frontier.add(neighbor)

    return mask


def _colorize_mask(
    mask: np.ndarray,
    bg: int,
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    n_colors = 2 if len(coords) < 5 else 2 + int(rng.integers(0, 2))
    n_colors = min(n_colors, len(coords), len(palette))
    chosen = [int(x) for x in rng.choice(np.array(palette, dtype=int), size=n_colors, replace=False)]
    seed_ids = rng.choice(len(coords), size=n_colors, replace=False)
    seeds = [coords[int(i)] for i in np.atleast_1d(seed_ids)]

    panel = np.full(mask.shape, bg, dtype=int)
    metric = int(rng.integers(3))

    def dist(a: tuple[int, int], b: tuple[int, int]) -> int:
        if metric == 0:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        if metric == 1:
            return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        return 2 * abs(a[0] - b[0]) + abs(a[1] - b[1])

    for color, seed in zip(chosen, seeds):
        panel[seed] = color

    for cell in coords:
        if panel[cell] != bg:
            continue
        best_idx = min(range(n_colors), key=lambda i: (dist(cell, seeds[i]), i))
        panel[cell] = chosen[best_idx]

    return panel


def _make_panel(branch: str, side: int, bg: int, palette: list[int], rng: np.random.Generator) -> np.ndarray:
    count = _fill_count(branch, side, rng)
    mask = _grow_mask(side, count, rng)
    panel = _colorize_mask(mask, bg, palette, rng)
    assert int(panel[0, 0]) == bg
    assert int(panel[0, side - 1]) == bg
    assert int(panel[side - 1, 0]) == bg
    assert int(panel[side - 1, side - 1]) == bg
    return panel


def _branch_assignment(panel_count: int, rng: np.random.Generator) -> list[str]:
    offset = int(rng.integers(len(_BRANCHES)))
    seq = [_BRANCHES[(offset + i) % len(_BRANCHES)] for i in range(panel_count)]
    rng.shuffle(seq)
    return seq


def _expected_output(
    grid: np.ndarray,
    bg: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            out[r0:r1, c0:c1] = _transform_panel(grid[r0:r1, c0:c1], bg)
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.integers(2, 5))
    cols = int(rng.integers(2, 5))
    side = int(rng.choice(np.array(_SIDE_CHOICES, dtype=int)))

    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=6, replace=False)]
    bg, sep_color = palette[0], palette[1]
    fg_palette = palette[2:]

    h = rows * side + (rows - 1)
    w = cols * side + (cols - 1)
    grid = np.full((h, w), bg, dtype=int)

    sep_rows = [side + i * (side + 1) for i in range(rows - 1)]
    sep_cols = [side + i * (side + 1) for i in range(cols - 1)]
    for r in sep_rows:
        grid[r, :] = sep_color
    for c in sep_cols:
        grid[:, c] = sep_color

    row_spans = _panel_spans(sep_rows, h)
    col_spans = _panel_spans(sep_cols, w)
    branches = _branch_assignment(rows * cols, rng)

    idx = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panel = _make_panel(branches[idx], side, bg, fg_palette, rng)
            grid[r0:r1, c0:c1] = panel
            idx += 1

    expected = _expected_output(grid, bg, row_spans, col_spans)
    try:
        out = main(grid)
    except Exception:
        return None

    if not np.array_equal(out, expected):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_MAX_ATTEMPTS} attempts")
