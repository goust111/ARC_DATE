from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
BAR_COLORS = (3, 7)
GEN_ATTEMPTS = 400

SIDE_PATTERNS = (
    ("T", "R"),
    ("R", "B"),
    ("B", "L"),
    ("L", "T"),
    ("T", "L", "R"),
    ("R", "T", "B"),
    ("B", "L", "R"),
    ("L", "T", "B"),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _stroke_cells(
    box: tuple[int, int, int, int],
    sides: tuple[str, ...],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    if r1 < r0 or c1 < c0:
        return set()
    h, w = shape
    cells: set[tuple[int, int]] = set()
    if "T" in sides:
        for c in range(c0, c1 + 1):
            if 0 <= r0 < h and 0 <= c < w:
                cells.add((r0, c))
    if "B" in sides:
        for c in range(c0, c1 + 1):
            if 0 <= r1 < h and 0 <= c < w:
                cells.add((r1, c))
    if "L" in sides:
        for r in range(r0, r1 + 1):
            if 0 <= r < h and 0 <= c0 < w:
                cells.add((r, c0))
    if "R" in sides:
        for r in range(r0, r1 + 1):
            if 0 <= r < h and 0 <= c1 < w:
                cells.add((r, c1))
    return cells


def _extract_sides(comp: list[tuple[int, int]]) -> tuple[str, ...]:
    box = _bbox(comp)
    r0, c0, r1, c1 = box
    pts = set(comp)
    sides: list[str] = []
    if all((r0, c) in pts for c in range(c0, c1 + 1)):
        sides.append("T")
    if all((r1, c) in pts for c in range(c0, c1 + 1)):
        sides.append("B")
    if all((r, c0) in pts for r in range(r0, r1 + 1)):
        sides.append("L")
    if all((r, c1) in pts for r in range(r0, r1 + 1)):
        sides.append("R")
    side_cells = _stroke_cells(box, tuple(sides), (r1 + 1, c1 + 1))
    assert side_cells == pts, "blue component must be a union of full rectangle sides"
    assert len(sides) >= 2, "expected an L- or U-shaped rectangle stroke"
    return tuple(sides)


def _infer_delta(
    boxes: list[tuple[int, int, int, int]],
    indices: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    deltas: list[tuple[int, int, int, int]] = []
    arrs = [np.array(box, dtype=int) for box in boxes]
    for i in range(len(arrs) - 1):
        gap = indices[i + 1] - indices[i]
        diff = arrs[i + 1] - arrs[i]
        if np.any(diff % gap != 0):
            return None
        deltas.append(tuple(int(v) for v in (diff // gap)))
    if not deltas:
        return None
    if len(set(deltas)) != 1:
        return None
    return deltas[0]


def _candidate_patterns(n_obs: int) -> tuple[tuple[int, ...], ...]:
    if n_obs == 2:
        return ((0, 2), (0, 3))
    if n_obs == 3:
        return ((0, 1, 3), (0, 2, 3))
    raise AssertionError("expected 2 or 3 observed stroke components")


def _infer_sequence_pattern(
    boxes: list[tuple[int, int, int, int]]
) -> tuple[tuple[int, ...], tuple[int, int, int, int]]:
    for indices in _candidate_patterns(len(boxes)):
        delta = _infer_delta(boxes, indices)
        if delta is None:
            continue
        valid = True
        first = np.array(boxes[0], dtype=int)
        step = np.array(delta, dtype=int)
        for k in range(indices[-1] + 1):
            box = tuple(int(v) for v in (first + k * step))
            if box[2] < box[0] or box[3] < box[1]:
                valid = False
                break
        if valid:
            return indices, delta
    raise AssertionError("could not infer a unique arithmetic stroke pattern")


def _collect_full_sequence(
    first_box: tuple[int, int, int, int],
    delta: tuple[int, int, int, int],
    sides: tuple[str, ...],
    shape: tuple[int, int],
) -> dict[int, set[tuple[int, int]]]:
    base = np.array(first_box, dtype=int)
    step = np.array(delta, dtype=int)
    seq: dict[int, set[tuple[int, int]]] = {}

    idx = 0
    while True:
        box = tuple(int(v) for v in (base + idx * step))
        cells = _stroke_cells(box, sides, shape)
        if not cells:
            break
        seq[idx] = cells
        idx += 1

    idx = -1
    while True:
        box = tuple(int(v) for v in (base + idx * step))
        cells = _stroke_cells(box, sides, shape)
        if not cells:
            break
        seq[idx] = cells
        idx -= 1
    return seq


def _find_bars(grid: np.ndarray) -> list[dict[str, object]]:
    bars: list[dict[str, object]] = []
    for comp in _components((grid != BG) & (grid != BLUE)):
        if len(comp) != 2:
            continue
        (r0, c0), (r1, c1) = sorted(comp)
        if r0 != r1 or abs(c0 - c1) != 1:
            continue
        left_c = min(c0, c1)
        bars.append(
            {
                "row": r0,
                "left_col": left_c,
                "colors": (int(grid[r0, left_c]), int(grid[r0, left_c + 1])),
            }
        )
    assert len(bars) >= 2, "expected several horizontal two-color bars"
    bars.sort(key=lambda bar: (int(bar["row"]), int(bar["left_col"])))
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete an arithmetic sequence of rectangle-side strokes, align the two-color bars, and crop the strokes to the aligned bar rows.

    Input structure:
    - A black background contains 2-3 blue connected components, each being an L- or U-shaped subset of a rectangle border.
    - The rectangle stroke components form an increasing arithmetic sequence by bounding box area, with 1-2 missing middle members.
    - Several horizontal two-cell bars sit on distinct rows in the right region; the topmost bar is the reference, and later bars may use the same or reversed left-right color order.

    Rule:
    - Sort the blue stroke components by bounding-box area, infer the fixed row/column bounding-box delta, and insert the missing members; keep extending the same stroke pattern in both directions until the next member has no visible pixels.
    - Treat the topmost two-color bar as the reference. For every other bar, shift the whole two-cell bar horizontally so that its left pixel lands in the reference column assigned to that same color.
    - Collect the aligned bar rows and keep only blue pixels lying on those rows from the completed stroke sequence; erase the other blue pixels to black.
    - Preserve all bar colors at their aligned positions.

    Key insight:
    - The missing blue strokes are recovered from arithmetic bbox progression, but the final visible subset depends on a separate same-color bar alignment that acts as a row mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    blue_components = _components(g == BLUE)
    assert 2 <= len(blue_components) <= 3, "expected 2-3 blue stroke components"
    comps = sorted(
        (
            {
                "coords": comp,
                "bbox": _bbox(comp),
                "area": _bbox_area(_bbox(comp)),
                "sides": _extract_sides(comp),
            }
            for comp in blue_components
        ),
        key=lambda item: (int(item["area"]), item["bbox"]),
    )

    sides = comps[0]["sides"]
    assert all(item["sides"] == sides for item in comps), "all strokes must share the same side pattern"
    boxes = [item["bbox"] for item in comps]
    indices, delta = _infer_sequence_pattern(boxes)
    assert indices[0] == 0

    full_seq = _collect_full_sequence(boxes[0], delta, sides, g.shape)

    bars = _find_bars(g)
    ref = bars[0]
    ref_left = int(ref["left_col"])
    ref_colors = tuple(int(v) for v in ref["colors"])
    ref_cols = {ref_colors[0]: ref_left, ref_colors[1]: ref_left + 1}

    aligned_rows: set[int] = set()
    aligned_bars: list[tuple[int, int, tuple[int, int]]] = []
    for bar in bars:
        row = int(bar["row"])
        left_col = int(bar["left_col"])
        colors = tuple(int(v) for v in bar["colors"])
        shift = ref_cols[colors[0]] - left_col
        new_left = left_col + shift
        assert 0 <= new_left and new_left + 1 < g.shape[1]
        aligned_rows.add(row)
        aligned_bars.append((row, new_left, colors))

    out = np.full_like(g, BG)
    for cells in full_seq.values():
        for r, c in cells:
            if r in aligned_rows:
                out[r, c] = BLUE
    for row, left_col, colors in aligned_bars:
        out[row, left_col] = colors[0]
        out[row, left_col + 1] = colors[1]
    return out


def _term_box(
    grow_axis: str,
    top: int,
    left: int,
    idx: int,
    step_r: int,
    step_c: int,
    base_h: int,
    base_w: int,
    delta_size: int,
) -> tuple[int, int, int, int]:
    r0 = top + idx * step_r
    c0 = left + idx * step_c
    if grow_axis == "width":
        h = base_h
        w = base_w + idx * delta_size
    else:
        h = base_h + idx * delta_size
        w = base_w
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _bars_orders(ref_colors: tuple[int, int], count: int) -> list[tuple[int, int]]:
    same = ref_colors
    flipped = (ref_colors[1], ref_colors[0])
    if count == 1:
        return [same]
    if count == 2:
        items = [same, flipped]
        random.shuffle(items)
        return items
    items = [same, flipped]
    while len(items) < count:
        items.append(random.choice((same, flipped)))
    random.shuffle(items)
    return items


def _expected_from_construction(
    shape: tuple[int, int],
    seq_cells: dict[int, set[tuple[int, int]]],
    aligned_bars: list[tuple[int, int, tuple[int, int]]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    rows = {row for row, _left, _colors in aligned_bars}
    for cells in seq_cells.values():
        for r, c in cells:
            if r in rows:
                out[r, c] = BLUE
    for row, left, colors in aligned_bars:
        out[row, left] = colors[0]
        out[row, left + 1] = colors[1]
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    observed_count = random.choice((2, 3))
    if observed_count == 2:
        obs_indices = random.choice(((0, 2), (0, 3)))
    else:
        obs_indices = random.choice(((0, 1, 3), (0, 2, 3)))
    total_terms = obs_indices[-1] + 1

    grow_axis = random.choice(("width", "height"))
    sides = random.choice(SIDE_PATTERNS)

    if grow_axis == "width":
        base_h = random.randint(3, 5)
        base_w = random.randint(3, 4)
        delta_size = random.randint(1, 2)
        step_r = base_h + random.randint(1, 2)
        step_c = random.randint(0, 1)
    else:
        base_h = random.randint(3, 4)
        base_w = random.randint(3, 5)
        delta_size = random.randint(1, 2)
        max_h = base_h + (total_terms - 1) * delta_size
        step_r = max_h + random.randint(1, 2)
        step_c = random.randint(0, 1)

    top = random.randint(0, max(0, step_r - base_h))
    left = random.randint(1, 2)
    boxes = [
        _term_box(grow_axis, top, left, idx, step_r, step_c, base_h, base_w, delta_size)
        for idx in range(total_terms)
    ]

    inferred = _infer_sequence_pattern([boxes[idx] for idx in obs_indices])
    if inferred[0] != obs_indices:
        return None

    last_r1 = boxes[-1][2]
    next_r0 = top + total_terms * step_r
    if last_r1 + 1 > next_r0:
        return None
    h_upper = min(next_r0, 18)
    h_lower = max(last_r1 + 1, 13)
    if h_lower > h_upper:
        return None
    height = random.randint(h_lower, h_upper)

    # Reserve a right-side bar region that never overlaps the completed strokes.
    max_c1 = max(box[3] for box in boxes)
    ref_left = max_c1 + random.randint(3, 4)
    width = ref_left + 4
    if width > 20:
        return None

    seq_cells = {
        idx: _stroke_cells(box, sides, (height, width))
        for idx, box in enumerate(boxes)
    }
    if not all(seq_cells.values()):
        return None
    used_rows = sorted({r for cells in seq_cells.values() for r, _ in cells})
    if len(used_rows) < 3:
        return None

    bar_count = random.randint(3, 4)
    if len(used_rows) < bar_count:
        return None
    bar_rows = sorted(random.sample(used_rows, bar_count))
    ref_colors = random.choice((BAR_COLORS, BAR_COLORS[::-1]))
    row_orders = [ref_colors] + _bars_orders(ref_colors, bar_count - 1)

    color_to_ref_col = {ref_colors[0]: ref_left, ref_colors[1]: ref_left + 1}
    input_bars: list[tuple[int, int, tuple[int, int]]] = []
    aligned_bars: list[tuple[int, int, tuple[int, int]]] = []
    moved = False
    for row, colors in zip(bar_rows, row_orders):
        final_left = color_to_ref_col[colors[0]]
        offsets = [-1, 0, 1]
        random.shuffle(offsets)
        current_left = None
        for offset in offsets:
            cand = final_left + offset
            if cand <= max_c1 or cand + 1 >= width:
                continue
            current_left = cand
            moved = moved or offset != 0
            break
        if current_left is None:
            return None
        input_bars.append((row, current_left, colors))
        aligned_bars.append((row, final_left, colors))
    if not moved:
        return None

    inp = np.full((height, width), BG, dtype=int)
    for idx in obs_indices:
        for r, c in seq_cells[idx]:
            inp[r, c] = BLUE
    for row, left_col, colors in input_bars:
        if np.any(inp[row, left_col : left_col + 2] != BG):
            return None
        inp[row, left_col] = colors[0]
        inp[row, left_col + 1] = colors[1]

    expected = _expected_from_construction((height, width), seq_cells, aligned_bars)
    if np.array_equal(inp, expected):
        return None
    if not np.any(expected == BLUE):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        inp, expected = case
        try:
            out = main(inp)
        except AssertionError:
            continue
        if np.array_equal(out, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 8")
