from __future__ import annotations

from itertools import combinations

import numpy as np


BG_COLOR = 0
MARKER_COLOR = 2
FIGURE_COLORS = tuple(color for color in range(10) if color not in {BG_COLOR, MARKER_COLOR})
MAX_GENERATION_ATTEMPTS = 300
_RNG = np.random.default_rng()
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _component_areas(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    areas: list[int] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            area = 0
            while stack:
                rr, cc = stack.pop()
                area += 1
                for dr, dc in _DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            areas.append(area)
    return areas


def _identify_roles(grid: np.ndarray) -> tuple[int, int, int]:
    bg = _most_common_color(grid)
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert len(colors) == 2, "expected one background, one figure color, and one marker color"

    marker_candidates: list[int] = []
    for color in colors:
        mask = grid == color
        areas = _component_areas(mask)
        row_counts = mask.sum(axis=1)
        if areas and max(areas) == 1 and int(row_counts.max()) <= 3:
            marker_candidates.append(int(color))

    assert len(marker_candidates) == 1, "expected exactly one singleton-marker color"
    marker = int(marker_candidates[0])
    figure = next(int(color) for color in colors if int(color) != marker)
    return bg, figure, marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each row's marker-count parity to decide whether that row's hidden figure strip stays put or flips.

    Input structure:
    - The grid has a dominant black background, one large single-color central figure, and a second foreground color used only as sparse isolated marker pixels.
    - The figure occupies a contiguous horizontal span on each of its rows, while markers may sit either on top of the figure or on background cells.
    - Every row contains between 0 and 3 marker pixels, and marker pixels are not part of the final figure.

    Rule:
    - Identify the marker color by its isolated single-pixel components and count those markers independently in each row.
    - Recover each row's underlying figure span from the visible non-marker figure pixels.
    - If a row has an odd number of markers, mirror that recovered figure span horizontally across the row; if the marker count is even, keep the span in place.
    - Remove all marker pixels and output only the transformed figure on the background.

    Key insight:
    - The sparse marker pixels are control signals rather than object pixels, and only the parity of their row-wise count determines whether that row is mirrored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, figure, marker = _identify_roles(g)
    h, w = g.shape
    out = np.full((h, w), bg, dtype=int)

    for r in range(h):
        row = g[r]
        marker_count = int(np.count_nonzero(row == marker))
        visible = np.flatnonzero(row == figure)
        if visible.size == 0:
            continue
        left = int(visible.min())
        right = int(visible.max())
        assert left < right, "figure rows must keep visible endpoints after marker placement"
        if marker_count % 2 == 1:
            left, right = w - 1 - right, w - 1 - left
        out[r, left : right + 1] = figure

    return out


def _sample_shape(height: int, width: int) -> tuple[np.ndarray, dict[int, tuple[int, int]]] | None:
    min_rows = 6
    max_rows = min(height - 2, 14)
    if max_rows < min_rows:
        return None

    obj_rows = int(_RNG.integers(min_rows, max_rows + 1))
    top = int(_RNG.integers(1, height - obj_rows))

    min_span = max(4, width // 3)
    max_span = min(width - 4, max(min_span + 2, (2 * width) // 3))
    if max_span < min_span:
        return None

    span = int(_RNG.integers(min_span, max_span + 1))
    center_hint = int(_RNG.integers(max(2, width // 2 - 2), min(width - 3, width // 2 + 2) + 1))
    left = int(np.clip(center_hint - span // 2, 1, width - span - 1))
    intervals: list[tuple[int, int]] = [(left, span)]

    for _ in range(1, obj_rows):
        prev_left, prev_span = intervals[-1]
        prev_right = prev_left + prev_span - 1
        candidates: list[tuple[int, int]] = []
        for d_left in (-2, -1, 0, 1, 2):
            for d_span in (-2, -1, 0, 1, 2):
                next_span = int(np.clip(prev_span + d_span, min_span, max_span))
                next_left = int(np.clip(prev_left + d_left, 1, width - next_span - 1))
                next_right = next_left + next_span - 1
                if next_left <= prev_right and next_right >= prev_left:
                    candidates.append((next_left, next_span))
        if not candidates:
            return None
        uniq = sorted(set(candidates))
        intervals.append(uniq[int(_RNG.integers(0, len(uniq)))])

    mask = np.zeros((height, width), dtype=bool)
    row_bounds: dict[int, tuple[int, int]] = {}
    for offset, (row_left, row_span) in enumerate(intervals):
        r = top + offset
        row_right = row_left + row_span - 1
        mask[r, row_left : row_right + 1] = True
        row_bounds[r] = (row_left, row_right)

    if int(mask.sum()) >= mask.size // 2:
        return None
    if np.array_equal(mask, np.fliplr(mask)):
        return None
    if not any((left + right) != width - 1 for left, right in row_bounds.values()):
        return None

    return mask, row_bounds


def _plan_marker_rows(
    height: int,
    width: int,
    row_bounds: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    counts = np.zeros(height, dtype=int)
    inside_counts = np.zeros(height, dtype=int)

    object_rows = sorted(row_bounds)
    background_rows = [row for row in range(height) if row not in row_bounds]
    off_center_rows = [row for row in object_rows if sum(row_bounds[row]) != width - 1]
    if len(object_rows) < 2 or not background_rows or not off_center_rows:
        return None

    odd_anchor = int(_RNG.choice(off_center_rows))
    even_anchor_choices = [row for row in object_rows if row != odd_anchor]
    if not even_anchor_choices:
        return None
    even_anchor = int(_RNG.choice(even_anchor_choices))

    parity = {row: int(_RNG.integers(0, 2)) for row in object_rows}
    parity[odd_anchor] = 1
    parity[even_anchor] = 0

    for row in object_rows:
        left, right = row_bounds[row]
        max_inside = right - left - 1
        if parity[row] == 1:
            counts[row] = 3 if _RNG.random() < 0.35 else 1
        else:
            counts[row] = 2 if _RNG.random() < 0.55 else 0
        if counts[row] > 0:
            inside_counts[row] = 1 if max_inside == 1 else int(_RNG.integers(1, min(2, max_inside) + 1))

    counts[even_anchor] = 2
    left, right = row_bounds[even_anchor]
    max_inside = right - left - 1
    inside_counts[even_anchor] = 1 if max_inside == 1 else int(_RNG.integers(1, min(2, max_inside) + 1))

    n_bg_marker_rows = int(_RNG.integers(1, min(3, len(background_rows)) + 1))
    for row in _RNG.choice(background_rows, size=n_bg_marker_rows, replace=False):
        counts[int(row)] = int(_RNG.choice([1, 2]))
        inside_counts[int(row)] = 0

    if not any(counts[row] % 2 == 1 and row in row_bounds for row in range(height)):
        return None
    if not any(counts[row] % 2 == 0 and row in row_bounds for row in range(height)):
        return None
    if not any(counts[row] > 0 and row in row_bounds for row in range(height)):
        return None
    if not any(counts[row] > 0 and row not in row_bounds for row in range(height)):
        return None

    return counts, inside_counts


def _choose_marker_columns(
    width: int,
    row_bounds: tuple[int, int] | None,
    total: int,
    inside_total: int,
    prev_cols: tuple[int, ...],
) -> tuple[int, ...] | None:
    if total == 0:
        return ()

    if row_bounds is None:
        inside_candidates: list[int] = []
        outside_candidates = list(range(width))
    else:
        left, right = row_bounds
        inside_candidates = list(range(left + 1, right))
        outside_candidates = [col for col in range(width) if col < left or col > right]

    inside_set = set(inside_candidates)
    allowed = sorted(set(inside_candidates) | set(outside_candidates))
    prev_set = set(prev_cols)
    valid: list[tuple[int, ...]] = []
    for combo in combinations(allowed, total):
        if any(col in prev_set for col in combo):
            continue
        if any(b - a == 1 for a, b in zip(combo, combo[1:])):
            continue
        if sum(col in inside_set for col in combo) != inside_total:
            continue
        valid.append(combo)
    if not valid:
        return None
    return valid[int(_RNG.integers(0, len(valid)))]


def _place_markers(
    height: int,
    width: int,
    row_bounds: dict[int, tuple[int, int]],
    counts: np.ndarray,
    inside_counts: np.ndarray,
) -> dict[int, tuple[int, ...]] | None:
    for _ in range(120):
        placement: dict[int, tuple[int, ...]] = {}
        prev_cols: tuple[int, ...] = ()
        ok = True
        for row in range(height):
            cols = _choose_marker_columns(
                width=width,
                row_bounds=row_bounds.get(row),
                total=int(counts[row]),
                inside_total=int(inside_counts[row]),
                prev_cols=prev_cols,
            )
            if cols is None:
                ok = False
                break
            placement[row] = cols
            prev_cols = cols
        if ok:
            return placement
    return None


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(10, 25))
    width = int(_RNG.integers(12, 29))
    figure_color = int(_RNG.choice(FIGURE_COLORS))

    sampled = _sample_shape(height, width)
    if sampled is None:
        return None
    mask, row_bounds = sampled

    planned = _plan_marker_rows(height, width, row_bounds)
    if planned is None:
        return None
    counts, inside_counts = planned

    placement = _place_markers(height, width, row_bounds, counts, inside_counts)
    if placement is None:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[mask] = figure_color

    marker_on_figure = False
    marker_on_background = False
    for row, cols in placement.items():
        for col in cols:
            if bool(mask[row, col]):
                marker_on_figure = True
            else:
                marker_on_background = True
            grid[row, col] = MARKER_COLOR

    if not marker_on_figure or not marker_on_background:
        return None

    odd_changed = any(
        int(counts[row]) % 2 == 1 and row in row_bounds and sum(row_bounds[row]) != width - 1
        for row in range(height)
    )
    if not odd_changed:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def _fast_choose_cols(
    rng: np.random.Generator,
    width: int,
    candidates: list[int],
    count: int,
    prev_cols: set[int],
) -> tuple[int, ...] | None:
    if count == 0:
        return ()
    pool = [c for c in candidates if c not in prev_cols]
    rng.shuffle(pool)
    chosen: list[int] = []
    for col in pool:
        if all(abs(col - old) > 1 for old in chosen):
            chosen.append(int(col))
            if len(chosen) == count:
                return tuple(sorted(chosen))
    return None


def _fast_generate_input() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(12, 24))
    width = int(rng.integers(14, 29))
    figure_color = int(rng.choice(FIGURE_COLORS))
    grid = np.full((height, width), BG_COLOR, dtype=int)

    obj_rows = int(rng.integers(6, min(12, height - 2) + 1))
    top = int(rng.integers(1, height - obj_rows))
    span = int(rng.integers(max(5, width // 3), min(width - 4, (2 * width) // 3) + 1))
    left = int(rng.integers(1, width - span - 1))
    row_bounds: dict[int, tuple[int, int]] = {}
    for i in range(obj_rows):
        drift = int(np.clip((i % 5) - 2, -2, 2))
        row_left = int(np.clip(left + drift, 1, width - span - 1))
        row_right = row_left + span - 1
        r = top + i
        grid[r, row_left : row_right + 1] = figure_color
        row_bounds[r] = (row_left, row_right)

    rows = sorted(row_bounds)
    odd_row = rows[int(rng.integers(0, len(rows)))]
    even_row = rows[(rows.index(odd_row) + max(2, len(rows) // 2)) % len(rows)]
    bg_rows = [r for r in range(height) if r not in row_bounds]
    if not bg_rows:
        return None
    bg_marker_row = int(rng.choice(bg_rows))

    marker_plan: dict[int, tuple[int, int]] = {
        odd_row: (1, 1),
        even_row: (2, 1),
        bg_marker_row: (1, 0),
    }
    if len(rows) > 3:
        extra_row = rows[int(rng.integers(0, len(rows)))]
        marker_plan.setdefault(extra_row, (int(rng.choice([0, 2])), 0))

    prev_cols: set[int] = set()
    marker_on_figure = False
    marker_on_background = False
    for row in range(height):
        total, inside = marker_plan.get(row, (0, 0))
        if total == 0:
            prev_cols = set()
            continue
        if row in row_bounds:
            left_bound, right_bound = row_bounds[row]
            inside_candidates = list(range(left_bound + 1, right_bound))
            outside_candidates = [c for c in range(width) if c < left_bound or c > right_bound]
        else:
            inside_candidates = []
            outside_candidates = list(range(width))

        inside_cols = _fast_choose_cols(rng, width, inside_candidates, inside, prev_cols)
        if inside_cols is None:
            return None
        blocked = prev_cols | set(inside_cols)
        outside_cols = _fast_choose_cols(rng, width, outside_candidates, total - inside, blocked)
        if outside_cols is None:
            return None
        cols = tuple(sorted(inside_cols + outside_cols))
        for col in cols:
            if grid[row, col] == figure_color:
                marker_on_figure = True
            else:
                marker_on_background = True
            grid[row, col] = MARKER_COLOR
        prev_cols = set(cols)

    if not marker_on_figure or not marker_on_background:
        return None
    main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(50):
        grid = _fast_generate_input()
        if grid is not None:
            return grid
    for _ in range(40):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-row-flip input")
