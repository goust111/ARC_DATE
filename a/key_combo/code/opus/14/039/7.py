from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
ARC_MAX_SIDE = 30
MAX_GENERATION_ATTEMPTS = 500

GRID_HEIGHT_RANGE = (13, 19)
GRID_WIDTH_MAX = 26
INPUT_RECT_HEIGHT_RANGE = (2, 6)
INPUT_RECT_WIDTH_RANGE = (1, 4)
INPUT_COLOR_POOL = tuple(range(1, 10))

COUNT_PATTERNS = (
    (3, 2, 1),
    (4, 2, 1),
    (4, 3, 1),
    (5, 2, 1),
    (5, 3, 1),
    (4, 3, 2),
    (5, 3, 2),
    (6, 2, 1),
    (6, 3, 1),
    (4, 3, 2, 1),
)

ALL_DIMS = tuple(
    (h, w)
    for h in range(INPUT_RECT_HEIGHT_RANGE[0], INPUT_RECT_HEIGHT_RANGE[1] + 1)
    for w in range(INPUT_RECT_WIDTH_RANGE[0], INPUT_RECT_WIDTH_RANGE[1] + 1)
)
TALL_DIMS = tuple((h, w) for h, w in ALL_DIMS if h > w)
WIDE_DIMS = tuple((h, w) for h, w in ALL_DIMS if w > h)
SMALL_DIMS = tuple((h, w) for h, w in ALL_DIMS if h <= 3 and w <= 2)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_color_components(grid: np.ndarray, bg: int = BACKGROUND) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(bg) or seen[r, c]:
                continue

            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return objects


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for obj in _find_color_components(grid, bg=BACKGROUND):
        r0, c0, r1, c1 = obj["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert obj["area"] == h * w, "foreground objects must be solid rectangles"
        assert np.all(patch == int(obj["color"])), "rectangle bbox must be fully filled"
        rects.append(
            {
                "color": int(obj["color"]),
                "bbox": obj["bbox"],
                "area": int(obj["area"]),
                "h": h,
                "w": w,
                "x": int(c0),
                "y": int(r0),
            }
        )
    assert rects, "expected at least one rectangle"
    return rects


def _build_output(rectangles: list[dict], *, swap: bool, order_key) -> np.ndarray:
    ordered = sorted(rectangles, key=order_key)
    assert ordered, "need at least one rectangle"

    transformed = []
    for rect in ordered:
        out_h = rect["w"] if swap else rect["h"]
        out_w = rect["h"] if swap else rect["w"]
        transformed.append((out_h, out_w, rect["color"]))

    canvas_h = max(h for h, _w, _color in transformed)
    canvas_w = sum(w for _h, w, _color in transformed) + len(transformed) - 1
    out = np.zeros((canvas_h, canvas_w), dtype=int)

    cursor = 0
    for h, w, color in transformed:
        out[canvas_h - h : canvas_h, cursor : cursor + w] = int(color)
        cursor += w + 1
    return out


def _solve_from_rectangles(rectangles: list[dict]) -> np.ndarray:
    counts = Counter(rect["color"] for rect in rectangles)
    ranked = counts.most_common()
    assert len(ranked) >= 2, "need at least one distractor color and one kept color"
    skip_color, skip_count = ranked[0]
    assert skip_count > ranked[1][1], "highest-frequency color must be unique"
    kept = [rect for rect in rectangles if rect["color"] != int(skip_color)]
    assert kept, "filtering must leave at least one rectangle"
    xs = [rect["x"] for rect in kept]
    assert len(xs) == len(set(xs)), "kept rectangles need unique left columns"
    return _build_output(kept, swap=True, order_key=lambda rect: rect["x"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count rectangle colors, discard the most frequent color, then swap width and height before packing.

    Input structure:
    - The grid has black background and several separated solid monochromatic rectangles.
    - Rectangles use 3-4 non-black colors; one color appears on more rectangles than any other color.
    - Kept rectangles have distinct original left-column coordinates, so their horizontal order is unambiguous.

    Rule:
    - Detect every non-black rectangle and record its color, width, height, and original left column.
    - Count rectangles per color and remove every rectangle whose color is the unique highest-frequency color.
    - For each remaining rectangle, swap width and height, turning `(w, h)` into `(h, w)` while keeping the color.
    - Sort the swapped rectangles by the original left-column coordinate and place them left-to-right in one row band, bottom-aligned, with one black column between neighbors.

    Key insight:
    - The distractor color is identifiable only by cross-object frequency, and the surviving rectangles then undergo a separate geometric transform, so counting and width-height swapping must both be applied in sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rectangles = _extract_rectangles(g)
    return _solve_from_rectangles(rectangles)


def _sample_retained_dims(kept_count: int) -> list[tuple[int, int]] | None:
    budget = ARC_MAX_SIDE - (kept_count - 1)
    roles = [WIDE_DIMS, TALL_DIMS] + [ALL_DIMS] * (kept_count - 2)
    dims: list[tuple[int, int]] = []
    used_height = 0

    for i, pool in enumerate(roles):
        slots_left = kept_count - i - 1
        max_h = budget - used_height - 2 * slots_left
        candidates = [dim for dim in pool if dim[0] <= max_h]
        if not candidates:
            return None
        weights = [5 if dim[0] <= 4 else 2 for dim in candidates]
        dims.append(random.choices(candidates, weights=weights, k=1)[0])
        used_height += dims[-1][0]

    if len(set(dims)) < 2:
        return None
    if not any(h > w for h, w in dims):
        return None
    if not any(w > h for h, w in dims):
        return None
    return dims


def _assign_dims_to_color_groups(
    pattern: tuple[int, ...],
) -> tuple[list[dict], list[dict]] | None:
    distractor_count = int(pattern[0])
    survivor_counts = [int(v) for v in pattern[1:]]
    kept_count = sum(survivor_counts)
    kept_dims = _sample_retained_dims(kept_count)
    if kept_dims is None:
        return None

    dims_by_area = sorted(kept_dims, key=lambda dim: (dim[0] * dim[1], dim[0], dim[1]), reverse=True)
    distractor_dims = [random.choice(SMALL_DIMS) for _ in range(distractor_count)]
    distractor_area = sum(h * w for h, w in distractor_dims)

    groups: list[list[tuple[int, int]]] = []
    idx = 0
    for count in survivor_counts:
        groups.append(dims_by_area[idx : idx + count])
        idx += count

    if sum(h * w for h, w in groups[0]) <= distractor_area:
        return None

    survivor_rects: list[dict] = []
    for group_index, dims in enumerate(groups):
        for dim in dims:
            survivor_rects.append(
                {
                    "role": f"survivor_{group_index}",
                    "h": int(dim[0]),
                    "w": int(dim[1]),
                }
            )

    distractor_rects = [{"role": "distractor", "h": int(h), "w": int(w)} for h, w in distractor_dims]
    return distractor_rects, survivor_rects


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    rows, cols = occupied.shape
    if top < 0 or left < 0 or top + h > rows or left + w > cols:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(rows, top + h + 1)
    c1 = min(cols, left + w + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_rectangles(rectangles: list[dict], rows: int, cols: int) -> bool:
    occupied = np.zeros((rows, cols), dtype=bool)
    used_lefts: set[int] = set()
    order = sorted(
        range(len(rectangles)),
        key=lambda idx: (rectangles[idx]["h"] * rectangles[idx]["w"], rectangles[idx]["h"], rectangles[idx]["w"]),
        reverse=True,
    )

    for idx in order:
        rect = rectangles[idx]
        h = int(rect["h"])
        w = int(rect["w"])
        candidates: list[tuple[int, int]] = []
        lefts = list(range(cols - w + 1))
        random.shuffle(lefts)
        for left in lefts:
            if left in used_lefts:
                continue
            tops = list(range(rows - h + 1))
            random.shuffle(tops)
            for top in tops:
                if _can_place(occupied, top, left, h, w):
                    candidates.append((top, left))
        if not candidates:
            return False
        top, left = random.choice(candidates)
        rect["bbox"] = (top, left, top + h - 1, left + w - 1)
        rect["x"] = int(left)
        rect["y"] = int(top)
        used_lefts.add(int(left))
        occupied[top : top + h, left : left + w] = True
    return True


def _paint_input(rectangles: list[dict], rows: int, cols: int) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    return grid


def _row_major_order(rects: list[dict]) -> list[dict]:
    return sorted(rects, key=lambda rect: (rect["y"], rect["x"]))


def _semantic_checks(grid: np.ndarray) -> bool:
    rects = _extract_rectangles(grid)
    counts = Counter(rect["color"] for rect in rects)
    count_values = list(counts.values())
    if len(count_values) < 3 or len(count_values) > 4:
        return False
    if len(count_values) != len(set(count_values)):
        return False

    skip_color, skip_count = counts.most_common(1)[0]
    if list(counts.values()).count(skip_count) != 1:
        return False

    kept = [rect for rect in rects if rect["color"] != int(skip_color)]
    if not kept:
        return False
    if len({rect["x"] for rect in kept}) != len(kept):
        return False
    if not any(rect["h"] > rect["w"] for rect in kept):
        return False
    if not any(rect["w"] > rect["h"] for rect in kept):
        return False

    expected = _build_output(kept, swap=True, order_key=lambda rect: rect["x"])
    if expected.shape[0] > ARC_MAX_SIDE or expected.shape[1] > ARC_MAX_SIDE:
        return False
    if not np.array_equal(main(grid), expected):
        return False

    if np.array_equal(_build_output(rects, swap=True, order_key=lambda rect: rect["x"]), expected):
        return False
    if np.array_equal(_build_output(kept, swap=False, order_key=lambda rect: rect["x"]), expected):
        return False
    if np.array_equal(_build_output(_row_major_order(kept), swap=True, order_key=lambda rect: (rect["y"], rect["x"])), expected):
        return False

    pixel_counts = Counter()
    for rect in rects:
        pixel_counts[rect["color"]] += rect["area"]
    pixel_skip = pixel_counts.most_common(1)[0][0]
    if pixel_skip == skip_color:
        return False
    wrong_kept = [rect for rect in rects if rect["color"] != int(pixel_skip)]
    if np.array_equal(_build_output(wrong_kept, swap=True, order_key=lambda rect: rect["x"]), expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    pattern = random.choice(COUNT_PATTERNS)
    assigned = _assign_dims_to_color_groups(pattern)
    if assigned is None:
        return None
    distractors, survivors = assigned

    n_colors = len(pattern)
    colors = random.sample(INPUT_COLOR_POOL, n_colors)
    rectangles: list[dict] = []

    for rect in distractors:
        rectangles.append({**rect, "color": colors[0]})

    survivor_colors = colors[1:]
    offset = 0
    survivor_counts = list(pattern[1:])
    for color, count in zip(survivor_colors, survivor_counts):
        group = survivors[offset : offset + count]
        offset += count
        for rect in group:
            rectangles.append({**rect, "color": color})

    rows_min = max(GRID_HEIGHT_RANGE[0], max(rect["h"] for rect in rectangles) + 2)
    rows_max = GRID_HEIGHT_RANGE[1]
    if rows_min > rows_max:
        return None
    rows = random.randint(rows_min, rows_max)

    cols_min = max(15, len(rectangles) + 10, max(rect["w"] for rect in rectangles) + 2)
    cols_max = GRID_WIDTH_MAX
    if cols_min > cols_max:
        return None
    cols = random.randint(cols_min, cols_max)

    if not _place_rectangles(rectangles, rows, cols):
        return None

    grid = _paint_input(rectangles, rows, cols)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
