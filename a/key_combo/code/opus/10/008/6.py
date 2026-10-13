from __future__ import annotations

from itertools import combinations, permutations

import numpy as np


BG = 0
BLANK = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _phase_index(row: int, col: int, period: int) -> int:
    if row <= col:
        return col
    return (col + 1) % period


def _find_table_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h == w >= 3 and len(coords) == h * w:
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected one solid square table component"
    return candidates[0]


def _left_frame_data(
    grid: np.ndarray, table_bbox: tuple[int, int, int, int], bg: int
) -> tuple[np.ndarray, list[int], dict[int, np.ndarray]]:
    left = np.array(grid, copy=True)
    r0, c0, r1, c1 = table_bbox
    left[r0 : r1 + 1, c0 : c1 + 1] = bg

    colors = sorted(int(v) for v in np.unique(left) if int(v) != bg)
    assert 3 <= len(colors) <= 5, "expected 3 to 5 frame colors"

    frames: dict[int, np.ndarray] = {}
    for color in colors:
        coords = np.argwhere(left == color)
        assert coords.size > 0, "each frame color must remain visible"
        bbox = _bbox(coords)
        mask = _frame_mask(grid.shape, bbox)
        assert np.all(mask[left == color]), "visible pixels must lie on the inferred perimeter"
        frames[color] = mask
    return left, colors, frames


def _crossing_edges(left: np.ndarray, colors: list[int], frames: dict[int, np.ndarray]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for a, b in combinations(colors, 2):
        overlap = frames[a] & frames[b]
        pts = np.argwhere(overlap)
        assert len(pts) > 0, "every frame pair must have an observable crossing"
        visible = {int(left[r, c]) for r, c in pts}
        assert visible <= {a, b}, "pairwise crossings must not be hidden by a third frame"
        assert len(visible) == 1, "each frame pair must encode a single direction"
        front = next(iter(visible))
        if front == a:
            edges.add((b, a))
        else:
            edges.add((a, b))
    return edges


def _unique_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    remaining = set(colors)
    order: list[int] = []
    while remaining:
        roots = [
            color
            for color in colors
            if color in remaining and not any((other, color) in edges for other in remaining)
        ]
        assert len(roots) == 1, "crossings must determine a unique total order"
        cur = roots[0]
        order.append(cur)
        remaining.remove(cur)
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            assert (a, b) in edges, "the inferred order must match every pairwise crossing"
    return order


def _fill_table(
    grid: np.ndarray, table_bbox: tuple[int, int, int, int], order: list[int]
) -> np.ndarray:
    r0, c0, r1, c1 = table_bbox
    table = grid[r0 : r1 + 1, c0 : c1 + 1]
    n = len(order)
    assert table.shape == (n, n), "table size must match the number of frame colors"

    table_colors = {int(v) for v in np.unique(table)}
    blank_candidates = table_colors - set(order)
    assert len(blank_candidates) == 1, "expected a unique blank color distinct from the frame colors"
    blank = next(iter(blank_candidates))

    rows, cols = np.indices((n, n))
    phase = np.where(rows <= cols, cols, (cols + 1) % n)
    pattern = np.asarray(order, dtype=int)[phase]
    visible = table != blank
    assert np.array_equal(table[visible], pattern[visible]), "visible table cells must follow the cyclic rule"

    out = np.array(grid, copy=True)
    filled = np.array(table, copy=True)
    filled[~visible] = pattern[~visible]
    out[r0 : r1 + 1, c0 : c1 + 1] = filled
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the table blanks using the color cycle encoded by the overlap order of the rectangular frames.

    Input structure:
    - The background is black, with overlapping one-cell-thick hollow rectangular frames on the left.
    - A solid square table on the right contains frame colors plus one blank color.
    - The table side length equals the number of distinct frame colors.

    Rule:
    - Infer each full frame from the bounding box of its visible pixels.
    - At every pairwise perimeter crossing, add a directed edge from the hidden frame color to the visible frame color.
    - Recover the unique bottom-to-top frame order from those edges.
    - Fill each blank table cell with `order[c]` when `r <= c`, otherwise with `order[(c + 1) mod N]`.

    Key insight:
    - The left-side frame overlaps do not merely identify colors; they encode the unique cycle order needed to complete the right-side diagonally shifted periodic table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    table_bbox = _find_table_bbox(g, bg)
    left, colors, frames = _left_frame_data(g, table_bbox, bg)
    edges = _crossing_edges(left, colors, frames)
    order = _unique_order(colors, edges)
    return _fill_table(g, table_bbox, order)


def _table_from_order(order: list[int]) -> np.ndarray:
    n = len(order)
    rows, cols = np.indices((n, n))
    phase = np.where(rows <= cols, cols, (cols + 1) % n)
    return np.asarray(order, dtype=int)[phase]


def _blank_mask(rng: np.random.Generator, phase: np.ndarray) -> np.ndarray:
    n = phase.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    visible_counts = {idx: int(np.sum(phase == idx)) for idx in range(n)}

    def add(cell: tuple[int, int]) -> None:
        r, c = cell
        idx = int(phase[r, c])
        mask[r, c] = True
        visible_counts[idx] -= 1

    add((0, 0))  # Phase 0 appears only once; hiding it forces the left-side order reasoning.

    upper = [(r, c) for r in range(n) for c in range(n) if r <= c and (r, c) != (0, 0)]
    lower = [(r, c) for r in range(n) for c in range(n) if r > c]
    rng.shuffle(upper)
    rng.shuffle(lower)

    for pool in (upper, lower):
        for cell in pool:
            idx = int(phase[cell])
            if visible_counts[idx] > 1:
                add(cell)
                break

    extras = upper + lower
    rng.shuffle(extras)
    target = int(rng.integers(0, n + 1))
    extra_count = 0
    for cell in extras:
        if extra_count >= target or mask[cell]:
            continue
        idx = int(phase[cell])
        if visible_counts[idx] <= 1:
            continue
        add(cell)
        extra_count += 1

    assert int(np.count_nonzero(mask & np.triu(np.ones_like(mask, dtype=bool), 0))) >= 2, (
        "need at least two blanks on/above the diagonal"
    )
    assert np.any(mask & np.tril(np.ones_like(mask, dtype=bool), -1)), "need blanks below the diagonal"
    assert all(visible_counts[idx] >= 1 for idx in range(1, n)), "all nonzero phases must stay visible"
    return mask


def _count_table_only_orders(table: np.ndarray) -> int:
    n = table.shape[0]
    visible = table != BLANK
    count = 0
    for order in permutations(FRAME_COLORS, n):
        ok = True
        for r in range(n):
            for c in range(n):
                if visible[r, c] and int(table[r, c]) != int(order[_phase_index(r, c, n)]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            count += 1
            if count > 1:
                return count
    return count


def _build_frame_scene(
    rng: np.random.Generator, order: list[int]
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]], tuple[int, int]]:
    n = len(order)
    dr = int(rng.integers(1, 3))
    dc = int(rng.integers(1, 3))
    min_h = (n - 1) * dr + 4
    min_w = (n - 1) * dc + 4
    max_h = min_h + 4
    max_w = min_w + 3
    h = int(rng.integers(min_h, max_h + 1))
    w = int(rng.integers(min_w, max_w + 1))

    frame_span_h = h + (n - 1) * dr
    frame_span_w = w + (n - 1) * dc
    gap = int(rng.integers(2, 5))
    left_margin = int(rng.integers(1, 3))
    right_margin = int(rng.integers(1, 4))
    width = left_margin + frame_span_w + gap + n + right_margin
    if width > 30:
        raise ValueError("width exceeds ARC limit")

    height_min = max(14, frame_span_h + 2, n + 2)
    if height_min > 30:
        raise ValueError("height exceeds ARC limit")
    height = int(rng.integers(height_min, 31))
    if height > 30:
        raise ValueError("height exceeds ARC limit")

    max_top = height - frame_span_h - 1
    if max_top < 1:
        raise ValueError("frames do not fit vertically")
    top = int(rng.integers(1, max_top + 1))
    left = left_margin

    grid = np.zeros((height, width), dtype=int)
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for idx, color in enumerate(order):
        bbox = (top + idx * dr, left + idx * dc, top + idx * dr + h - 1, left + idx * dc + w - 1)
        bboxes[color] = bbox
        grid[_frame_mask(grid.shape, bbox)] = color
    return grid, bboxes, (frame_span_w + left_margin, gap)


def _embed_table(
    rng: np.random.Generator, grid: np.ndarray, order: list[int], frame_extent: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    n = len(order)
    out = np.array(grid, copy=True)
    frame_right, gap = frame_extent
    table = _table_from_order(order)
    phase = np.where(np.indices((n, n))[0] <= np.indices((n, n))[1], np.indices((n, n))[1], (np.indices((n, n))[1] + 1) % n)
    mask = _blank_mask(rng, phase)
    table_in = np.array(table, copy=True)
    table_in[mask] = BLANK

    row = int(rng.integers(0, out.shape[0] - n + 1))
    col = frame_right + gap
    out[row : row + n, col : col + n] = table_in
    return out, (row, col, row + n - 1, col + n - 1)


def _validate_instance(grid: np.ndarray, order: list[int]) -> bool:
    try:
        solved = main(grid)
    except AssertionError:
        return False

    bg = _get_bg(grid)
    table_bbox = _find_table_bbox(grid, bg)
    r0, c0, r1, c1 = table_bbox
    table_in = grid[r0 : r1 + 1, c0 : c1 + 1]
    table_out = solved[r0 : r1 + 1, c0 : c1 + 1]
    expected = _table_from_order(order)
    if not np.array_equal(table_out, expected):
        return False
    if np.any(table_in == order[0]):
        return False
    if _count_table_only_orders(table_in) <= 1:
        return False
    blank_mask = table_in == BLANK
    if not np.any(blank_mask & np.tril(np.ones_like(blank_mask, dtype=bool), -1)):
        return False
    if not np.any(blank_mask & np.triu(np.ones_like(blank_mask, dtype=bool), 0)):
        return False
    left_in = np.array(grid, copy=True)
    left_in[r0 : r1 + 1, c0 : c1 + 1] = bg
    left_out = np.array(solved, copy=True)
    left_out[r0 : r1 + 1, c0 : c1 + 1] = bg
    return np.array_equal(left_in, left_out)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(3, 6))
    order = list(int(v) for v in rng.choice(np.asarray(FRAME_COLORS, dtype=int), size=n, replace=False))
    try:
        frame_grid, _bboxes, extent = _build_frame_scene(rng, order)
        grid, _table_bbox = _embed_table(rng, frame_grid, order, extent)
    except ValueError:
        return None
    if _validate_instance(grid, order):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
