import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

COUNT_TO_COLOR = {0: BLUE, 1: GREEN, 2: YELLOW, 3: ORANGE}
TARGET_COLORS = (RED, GREY)

GEN_RETRIES = 400
GRID_H_RANGE = (15, 22)
SEGMENT_COUNT_CHOICES = (3, 4, 4, 5)
MIDDLE_WIDTH_CHOICES = (2, 3)
EDGE_WIDTH = 3

SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_segments(target_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = target_mask.shape
    cols: list[tuple[int, int, int]] = []
    for c in range(w):
        rows = np.flatnonzero(target_mask[:, c])
        if rows.size == 0:
            continue
        r0, r1 = int(rows[0]), int(rows[-1])
        assert rows.size == r1 - r0 + 1
        assert np.all(target_mask[r0 : r1 + 1, c])
        cols.append((c, r0, r1))

    assert cols, "expected one non-black target shape"
    col_ids = [c for c, _, _ in cols]
    assert col_ids == list(range(col_ids[0], col_ids[-1] + 1)), "target columns must be contiguous"

    segments: list[tuple[int, int, int, int]] = []
    seg_c0, seg_r0, seg_r1 = cols[0]
    prev_c = seg_c0
    for c, r0, r1 in cols[1:]:
        if (r0, r1) == (seg_r0, seg_r1):
            prev_c = c
            continue
        segments.append((seg_c0, prev_c, seg_r0, seg_r1))
        seg_c0, seg_r0, seg_r1 = c, r0, r1
        prev_c = c
    segments.append((seg_c0, prev_c, seg_r0, seg_r1))
    assert len(segments) >= 2
    return segments


def _count_by_leftmost_segment(grid: np.ndarray, segments: list[tuple[int, int, int, int]]) -> tuple[list[int], list[list[tuple[int, int]]]]:
    red_components = _components(grid == RED)
    counts = [0] * len(segments)
    col_to_segment: dict[int, int] = {}
    for idx, (c0, c1, _r0, _r1) in enumerate(segments):
        for c in range(c0, c1 + 1):
            col_to_segment[c] = idx

    for comp in red_components:
        leftmost = min(c for _r, c in comp)
        counts[col_to_segment[leftmost]] += 1
    return counts, red_components


def _paint_output(shape: tuple[int, int], segments: list[tuple[int, int, int, int]], counts: list[int]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for (c0, c1, r0, r1), count in zip(segments, counts):
        assert count in COUNT_TO_COLOR
        out[r0 : r1 + 1, c0 : c1 + 1] = COUNT_TO_COLOR[count]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each target column-run by the number of embedded red components assigned to that run.

    Input structure:
    - Black cells are background.
    - A single target shape occupies a contiguous band of columns, and each occupied column contains one continuous vertical interval.
    - Target cells are grey, except for small red 4-connected components embedded inside the target.

    Rule:
    - Ignore the red/grey difference when recovering the target silhouette, then greedily merge adjacent columns whose occupied intervals are exactly identical into column-runs.
    - Count red 4-connected components, assigning each component to the column-run containing its leftmost red cell.
    - Erase all red cells and recolor every cell of each column-run by its assigned component count: 0->blue, 1->green, 2->yellow, 3->orange.
    - Leave the black background unchanged.

    Key insight:
    - The red pixels are temporary messengers: they disappear in the output, but their component grouping and leftmost-run ownership determine the final run color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert set(np.unique(g)).issubset({BLACK, RED, GREY}), "unexpected input colors"

    target_mask = np.isin(g, TARGET_COLORS)
    assert np.any(target_mask), "missing target"
    segments = _extract_segments(target_mask)

    h, w = g.shape
    for r, c in np.argwhere(g == RED):
        for nr, nc in _neighbors4(int(r), int(c)):
            assert 0 <= nr < h and 0 <= nc < w and target_mask[nr, nc], "red must stay fully enclosed"

    counts, _ = _count_by_leftmost_segment(g, segments)
    assert any(count != counts[0] for count in counts[1:]), "at least two runs should differ"
    return _paint_output(g.shape, segments, counts)


def _build_segments(h: int, w: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    n_segments = rng.choice(SEGMENT_COUNT_CHOICES)
    widths = [EDGE_WIDTH] + [rng.choice(MIDDLE_WIDTH_CHOICES) for _ in range(n_segments - 2)] + [EDGE_WIDTH]
    total_width = sum(widths)
    min_w = max(18, 2 * total_width + 2)
    if min_w > 30:
        return None
    if w < min_w:
        w = min_w

    top = rng.randint(2, h - 9)
    bottom = top + rng.randint(6, 9)
    bottom = min(bottom, h - 3)
    intervals = [(top, bottom)]
    for _ in range(1, n_segments):
        placed = False
        for _attempt in range(80):
            cand_top = max(1, min(h - 8, intervals[-1][0] + rng.randint(-2, 2)))
            cand_bottom = max(cand_top + 6, min(h - 2, intervals[-1][1] + rng.randint(-2, 2)))
            if (cand_top, cand_bottom) == intervals[-1]:
                continue
            overlap = min(cand_bottom, intervals[-1][1]) - max(cand_top, intervals[-1][0]) + 1
            if overlap >= 5:
                intervals.append((cand_top, cand_bottom))
                placed = True
                break
        if not placed:
            return None

    start_lo = w // 2
    start_hi = w - total_width - 1
    if start_lo > start_hi:
        return None
    c = rng.randint(start_lo, start_hi)
    segments: list[tuple[int, int, int, int]] = []
    for width_i, (r0, r1) in zip(widths, intervals):
        segments.append((c, c + width_i - 1, r0, r1))
        c += width_i
    return segments


def _target_mask_from_segments(shape: tuple[int, int], segments: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for c0, c1, r0, r1 in segments:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _interior_mask(target_mask: np.ndarray) -> np.ndarray:
    h, w = target_mask.shape
    interior = np.zeros_like(target_mask, dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not target_mask[r, c]:
                continue
            if all(target_mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                interior[r, c] = True
    return interior


def _cross_candidates(segments: list[tuple[int, int, int, int]]) -> list[tuple[int, list[int]]]:
    candidates: list[tuple[int, list[int]]] = []
    for i in range(len(segments) - 1):
        _c0, c1, r0a, r1a = segments[i]
        c2, _c3, r0b, r1b = segments[i + 1]
        rows = list(range(max(r0a + 1, r0b + 1), min(r1a - 1, r1b - 1) + 1))
        if rows and c2 == c1 + 1:
            candidates.append((i, rows))
    return candidates


def _shape_cells(anchor_r: int, anchor_c: int, shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((anchor_r + dr, anchor_c + dc) for dr, dc in shape)


def _can_place(cells: tuple[tuple[int, int], ...], interior_mask: np.ndarray, existing: set[tuple[int, int]]) -> bool:
    h, w = interior_mask.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if not interior_mask[r, c] or (r, c) in existing:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in existing and (nr, nc) not in cell_set:
                return False
    return True


def _place_crossing_component(
    boundary_idx: int,
    rows: list[int],
    segments: list[tuple[int, int, int, int]],
    interior_mask: np.ndarray,
    existing: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[tuple[int, int], ...] | None:
    _lc0, left_c1, _lr0, _lr1 = segments[boundary_idx]
    right_c0, right_c1, _rr0, _rr1 = segments[boundary_idx + 1]
    row_order = rows[:]
    rng.shuffle(row_order)
    for row in row_order:
        options = [
            ((row, left_c1), (row, right_c0)),
            ((row, left_c1 - 1), (row, left_c1), (row, right_c0)),
        ]
        if right_c1 - right_c0 + 1 >= 3:
            options.append(((row, left_c1), (row, right_c0), (row, right_c0 + 1)))
        rng.shuffle(options)
        for option in options:
            if _can_place(option, interior_mask, existing):
                return option
    return None


def _place_in_segment(
    segment: tuple[int, int, int, int],
    interior_mask: np.ndarray,
    existing: set[tuple[int, int]],
    rng: random.Random,
    require_multicell: bool,
) -> tuple[tuple[int, int], ...] | None:
    c0, c1, r0, r1 = segment
    shape_pool = [shape for shape in SHAPES if len(shape) > 1] if require_multicell else list(SHAPES)
    rng.shuffle(shape_pool)
    anchors = [(r, c) for r in range(r0 + 1, r1) for c in range(c0, c1 + 1)]
    rng.shuffle(anchors)
    for shape in shape_pool:
        for anchor_r, anchor_c in anchors:
            cells = _shape_cells(anchor_r, anchor_c, shape)
            if any(not (c0 <= cc <= c1) for _rr, cc in cells):
                continue
            if _can_place(cells, interior_mask, existing):
                return cells
    return None


def _choose_counts(n_segments: int, cross_left: int, rng: random.Random) -> list[int]:
    if n_segments == 3:
        counts = rng.sample([0, 1, 2, 3], 3)
    elif n_segments == 4:
        counts = rng.sample([0, 1, 2, 3], 4)
    else:
        counts = rng.sample([0, 1, 2, 3], 4) + [rng.randrange(4)]
        rng.shuffle(counts)

    if counts[cross_left] == 0:
        donor = rng.choice([i for i, val in enumerate(counts) if val > 0])
        counts[cross_left], counts[donor] = counts[donor], counts[cross_left]

    if rng.random() < 0.7:
        right = cross_left + 1
        zero_spots = [i for i, val in enumerate(counts) if val == 0 and i != cross_left]
        if zero_spots:
            donor = rng.choice(zero_spots)
            counts[right], counts[donor] = counts[donor], counts[right]
        else:
            counts[right] = 0
            if len(set(counts)) == 1:
                counts[(right + 1) % n_segments] = 1

    assert counts[cross_left] > 0
    assert len(set(counts)) >= 2
    return counts


def _visible_segment_counts(grid: np.ndarray, segments: list[tuple[int, int, int, int]]) -> list[int]:
    counts = []
    for c0, c1, r0, r1 in segments:
        sub = grid[r0 : r1 + 1, c0 : c1 + 1] == RED
        counts.append(len(_components(sub)))
    return counts


def _validate_semantics(grid: np.ndarray, expected_counts: list[int]) -> bool:
    target_mask = np.isin(grid, TARGET_COLORS)
    segments = _extract_segments(target_mask)
    if len(segments) != len(expected_counts):
        return False
    counts, red_components = _count_by_leftmost_segment(grid, segments)
    if counts != expected_counts:
        return False
    if not any(len(comp) > 1 for comp in red_components):
        return False
    local_counts = _visible_segment_counts(grid, segments)
    if local_counts == counts:
        return False
    visible_pixels = [int(np.count_nonzero(grid[r0 : r1 + 1, c0 : c1 + 1] == RED)) for c0, c1, r0, r1 in segments]
    if visible_pixels == counts:
        return False
    expected_output = _paint_output(grid.shape, segments, counts)
    return np.array_equal(main(grid), expected_output)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(20, 30)
    segments = _build_segments(h, w, rng)
    if segments is None:
        return None

    target_mask = _target_mask_from_segments((h, w), segments)
    interior = _interior_mask(target_mask)
    cross_options = _cross_candidates(segments)
    if not cross_options:
        return None

    cross_left, rows = rng.choice(cross_options)
    counts = _choose_counts(len(segments), cross_left, rng)
    remaining = counts[:]

    components: list[tuple[tuple[int, int], ...]] = []
    occupied: set[tuple[int, int]] = set()

    crossing = _place_crossing_component(cross_left, rows, segments, interior, occupied, rng)
    if crossing is None:
        return None
    components.append(crossing)
    occupied.update(crossing)
    remaining[cross_left] -= 1

    for idx, need in enumerate(remaining):
        for _ in range(need):
            cells = _place_in_segment(segments[idx], interior, occupied, rng, require_multicell=False)
            if cells is None:
                return None
            components.append(cells)
            occupied.update(cells)

    grid = np.full((h, w), BLACK, dtype=int)
    grid[target_mask] = GREY
    for comp in components:
        for r, c in comp:
            grid[r, c] = RED

    if not _validate_semantics(grid, counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
