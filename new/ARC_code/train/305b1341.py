from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, get_bg
import random


def _legend_pairs(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        left = int(g[r, 0])
        right = int(g[r, 1])
        if left == bg or right == bg:
            break
        pairs.append((left, right))
    assert pairs, "expected a non-empty two-column legend at the upper left"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each sparse color object into a filled rectangular region using its legend-paired fill color.

    Input structure
    A background-color grid has a two-column legend in columns 0 and 1, starting at row 0: each row gives
    a source color in the left column and that source color's fill color in the right column.  Outside the
    legend, the foreground pixels are sparse dot objects whose colors are exactly the legend source colors.

    Rule
    Clear the legend.  For each source color, take the bbox of all of its non-legend dots, expand that bbox
    by one pixel in every direction, and fill background cells inside the expanded rectangle with the paired
    legend fill color while preserving every original non-legend dot.

    Key insight
    The legend order is also the layer order: earlier legend rows are in front, so fill rectangles are painted
    from the last legend row back to the first and higher-priority fills overwrite lower-priority fills.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    H, W = g.shape
    assert W >= 2, "expected room for a two-column legend"

    pairs = _legend_pairs(g, bg)
    source_colors = {source for source, _fill in pairs}
    assert len(source_colors) == len(pairs), "legend source colors must be unique"

    legend_mask = np.zeros(g.shape, dtype=bool)
    legend_mask[: len(pairs), :2] = True
    dot_mask = (g != bg) & ~legend_mask
    assert dot_mask.any(), "expected source dots outside the legend"
    assert set(map(int, g[dot_mask])).issubset(source_colors), "outside dots must use legend source colors"

    dots_by_color: dict[int, np.ndarray] = {}
    all_dots = np.zeros(g.shape, dtype=bool)
    for source, _fill in pairs:
        coords = np.argwhere(dot_mask & (g == source))
        assert coords.size, "each legend source color must appear outside the legend"
        dots_by_color[source] = coords
        all_dots[coords[:, 0], coords[:, 1]] = True

    out = np.full_like(g, bg)
    for source, fill in reversed(pairs):
        r0, c0, r1, c1 = bbox_coords(dots_by_color[source])
        r0 = max(0, r0 - 1)
        c0 = max(0, c0 - 1)
        r1 = min(H - 1, r1 + 1)
        c1 = min(W - 1, c1 + 1)

        region = out[r0 : r1 + 1, c0 : c1 + 1]
        background_inside_region = ~all_dots[r0 : r1 + 1, c0 : c1 + 1]
        region[background_inside_region] = fill

    out[all_dots] = g[all_dots]
    return out

# --------------- generator (o5x) ---------------

# Canvas and legend geometry.
GRID_HEIGHT, GRID_WIDTH = 20, 20
BACKGROUND_COLOR = 0
LEGEND_COLS = 2
DOT_STRIDE, BBOX_PAD, NON_OVERLAP_GAP = 2, 1, 1

# Legend palette sampling.
COLORS = tuple(range(1, 10))
PAIR_COUNT_CHOICES = (2, 3)
ROTATED_SOURCE_FILL_PROB = 0.45

# Dot-grid shape sampling.
DOT_ROWS_MIN, DOT_ROWS_MAX = 2, 8
DOT_COLS_MIN, DOT_COLS_MAX = 2, 5

# Layout search.
THREE_PAIR_FORCED_OVERLAP_STEPS = (1, 2)
MODE_OVERLAP, MODE_OVERLAP_ANY, MODE_SEPARATE, MODE_ANY = "overlap", "overlap_any", "separate", "any"
POST_FORCED_LAYOUT_MODES = (MODE_SEPARATE, MODE_OVERLAP_ANY, MODE_ANY)
POST_FORCED_LAYOUT_WEIGHTS = (3, 3, 1)

# Generation retry budget.
GENERATION_RETRIES = 300


def _origin_bbox(origin: tuple[int, int], dot_rows: int, dot_cols: int) -> tuple[int, int, int, int]:
    r, c = origin
    return (
        r - BBOX_PAD,
        c - BBOX_PAD,
        r + DOT_STRIDE * (dot_rows - 1) + BBOX_PAD,
        c + DOT_STRIDE * (dot_cols - 1) + BBOX_PAD,
    )


def _hits_legend(box: tuple[int, int, int, int], n_pairs: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= n_pairs - 1 and r1 >= 0 and c0 <= LEGEND_COLS - 1 and c1 >= 0


def _legal_origins(n_pairs: int, dot_rows: int, dot_cols: int) -> list[tuple[int, int]]:
    origins: list[tuple[int, int]] = []
    max_r = GRID_HEIGHT - 1 - BBOX_PAD - DOT_STRIDE * (dot_rows - 1)
    max_c = GRID_WIDTH - 1 - BBOX_PAD - DOT_STRIDE * (dot_cols - 1)
    for r in range(BBOX_PAD, max_r + 1):
        for c in range(BBOX_PAD, max_c + 1):
            box = _origin_bbox((r, c), dot_rows, dot_cols)
            if not _hits_legend(box, n_pairs):
                origins.append((r, c))
    return origins


def _dot_coords(origin: tuple[int, int], dot_rows: int, dot_cols: int) -> np.ndarray:
    rows = origin[0] + DOT_STRIDE * np.arange(dot_rows)
    cols = origin[1] + DOT_STRIDE * np.arange(dot_cols)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1).astype(int)


def _expanded_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox_coords(coords)
    return (
        max(0, r0 - BBOX_PAD),
        max(0, c0 - BBOX_PAD),
        min(GRID_HEIGHT - 1, r1 + BBOX_PAD),
        min(GRID_WIDTH - 1, c1 + BBOX_PAD),
    )


def _inside_box(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return (coords[:, 0] >= r0) & (coords[:, 0] <= r1) & (coords[:, 1] >= c0) & (coords[:, 1] <= c1)


def _separated_by_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return (
        a[2] + NON_OVERLAP_GAP < b[0]
        or b[2] + NON_OVERLAP_GAP < a[0]
        or a[3] + NON_OVERLAP_GAP < b[1]
        or b[3] + NON_OVERLAP_GAP < a[1]
    )


def _intersection_bounds(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0, c0 = max(a[0], b[0]), max(a[1], b[1])
    r1, c1 = min(a[2], b[2]), min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _intersection_has_background(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], dot_mask: np.ndarray
) -> bool:
    bounds = _intersection_bounds(a, b)
    if bounds is None:
        return False
    r0, c0, r1, c1 = bounds
    return bool((~dot_mask[r0 : r1 + 1, c0 : c1 + 1]).any())


def _remove_dots_covered_by_higher_priority(
    coords: np.ndarray, higher_boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    keep = np.ones(len(coords), dtype=bool)
    for box in higher_boxes:
        keep &= ~_inside_box(coords, box)
    return coords[keep]


def _scene_data(
    sizes: list[tuple[int, int]], origins: list[tuple[int, int]]
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]], np.ndarray] | None:
    remaining: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []
    dot_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)

    for (dot_rows, dot_cols), origin in zip(sizes, origins):
        coords = _remove_dots_covered_by_higher_priority(_dot_coords(origin, dot_rows, dot_cols), boxes)
        if coords.size == 0:
            return None
        if len(set(map(int, coords[:, 0]))) != dot_rows or len(set(map(int, coords[:, 1]))) != dot_cols:
            return None
        if dot_mask[coords[:, 0], coords[:, 1]].any():
            return None
        dot_mask[coords[:, 0], coords[:, 1]] = True
        remaining.append(coords)
        boxes.append(_expanded_bbox(coords))

    return remaining, boxes, dot_mask


def _render_output(
    pairs: list[tuple[int, int]], remaining: list[np.ndarray], boxes: list[tuple[int, int, int, int]], dot_mask: np.ndarray
) -> np.ndarray:
    out = np.full((GRID_HEIGHT, GRID_WIDTH), BACKGROUND_COLOR, dtype=int)
    for idx in range(len(pairs) - 1, -1, -1):
        _source, fill = pairs[idx]
        r0, c0, r1, c1 = boxes[idx]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[~dot_mask[r0 : r1 + 1, c0 : c1 + 1]] = fill
    for (source, _fill), coords in zip(pairs, remaining):
        out[coords[:, 0], coords[:, 1]] = source
    return out


def _build_input(pairs: list[tuple[int, int]], remaining: list[np.ndarray]) -> np.ndarray:
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), BACKGROUND_COLOR, dtype=int)
    grid[: len(pairs), :LEGEND_COLS] = np.asarray(pairs, dtype=int)
    for (source, _fill), coords in zip(pairs, remaining):
        grid[coords[:, 0], coords[:, 1]] = source
    return grid


def _visible_overlap(
    pairs: list[tuple[int, int]], boxes: list[tuple[int, int, int, int]], dot_mask: np.ndarray, out: np.ndarray
) -> bool:
    for hi in range(len(pairs)):
        for lo in range(hi + 1, len(pairs)):
            bounds = _intersection_bounds(boxes[hi], boxes[lo])
            if bounds is None:
                continue
            r0, c0, r1, c1 = bounds
            overlap_fill = (~dot_mask[r0 : r1 + 1, c0 : c1 + 1]) & (
                out[r0 : r1 + 1, c0 : c1 + 1] == pairs[hi][1]
            )
            lr0, lc0, lr1, lc1 = boxes[lo]
            lower_fill = (~dot_mask[lr0 : lr1 + 1, lc0 : lc1 + 1]) & (
                out[lr0 : lr1 + 1, lc0 : lc1 + 1] == pairs[lo][1]
            )
            if overlap_fill.any() and lower_fill.any():
                return True
    return False


def _choose_pairs(n_pairs: int) -> list[tuple[int, int]]:
    colors = list(COLORS)
    sources = random.sample(colors, n_pairs)
    non_sources = [c for c in colors if c not in sources]
    if random.random() < ROTATED_SOURCE_FILL_PROB:
        fills = sources[1:] + sources[:1]
        replace_count = random.randint(0, n_pairs)
        for idx, color in zip(random.sample(range(n_pairs), replace_count), random.sample(non_sources, replace_count)):
            fills[idx] = color
    else:
        fills = random.sample(non_sources, n_pairs)
    return list(zip(sources, fills))


def _candidate_origins(
    sizes: list[tuple[int, int]],
    origins: list[tuple[int, int]],
    legal: list[tuple[int, int]],
    mode: str,
    target: int | None = None,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for origin in legal:
        data = _scene_data(sizes[: len(origins) + 1], origins + [origin])
        if data is None:
            continue
        _remaining, boxes, dot_mask = data
        new_box = boxes[-1]
        if mode == MODE_OVERLAP:
            assert target is not None
            if not _intersection_has_background(new_box, boxes[target], dot_mask):
                continue
        elif mode == MODE_OVERLAP_ANY:
            if not any(_intersection_has_background(new_box, old_box, dot_mask) for old_box in boxes[:-1]):
                continue
        elif mode == MODE_SEPARATE:
            if not all(_separated_by_gap(new_box, old_box) for old_box in boxes[:-1]):
                continue
        candidates.append(origin)
    return candidates


def _choose_origins(n_pairs: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    legal = [_legal_origins(n_pairs, dot_rows, dot_cols) for dot_rows, dot_cols in sizes]
    if any(not options for options in legal):
        return None

    origins = [random.choice(legal[0])]
    forced_step = 1 if n_pairs == 2 else random.choice(THREE_PAIR_FORCED_OVERLAP_STEPS)
    for idx in range(1, n_pairs):
        if idx == forced_step:
            target = random.randrange(idx)
            candidates = _candidate_origins(sizes, origins, legal[idx], MODE_OVERLAP, target)
        elif idx < forced_step:
            candidates = _candidate_origins(sizes, origins, legal[idx], MODE_SEPARATE)
            if not candidates:
                candidates = _candidate_origins(sizes, origins, legal[idx], MODE_ANY)
        else:
            mode = random.choices(POST_FORCED_LAYOUT_MODES, weights=POST_FORCED_LAYOUT_WEIGHTS, k=1)[0]
            candidates = _candidate_origins(sizes, origins, legal[idx], mode)
            if not candidates:
                candidates = _candidate_origins(sizes, origins, legal[idx], MODE_ANY)
        if not candidates:
            return None
        origins.append(random.choice(candidates))
    return origins


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice(PAIR_COUNT_CHOICES)
    pairs = _choose_pairs(n_pairs)
    sizes = [
        (random.randint(DOT_ROWS_MIN, DOT_ROWS_MAX), random.randint(DOT_COLS_MIN, DOT_COLS_MAX))
        for _ in range(n_pairs)
    ]
    origins = _choose_origins(n_pairs, sizes)
    if origins is None:
        return None

    data = _scene_data(sizes, origins)
    if data is None:
        return None
    remaining, boxes, dot_mask = data
    out = _render_output(pairs, remaining, boxes, dot_mask)
    if not _visible_overlap(pairs, boxes, dot_mask, out):
        return None

    inp = _build_input(pairs, remaining)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 305b1341 input")
