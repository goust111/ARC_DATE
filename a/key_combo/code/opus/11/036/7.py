from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400

MIN_LAYERS = 3
MAX_LAYERS = 4
INNER_FRAME_MIN = 5
INNER_FRAME_MAX = 8
GAP_MIN = 3
GAP_MAX = 5
OUTER_MARGIN_MIN = 1
OUTER_MARGIN_MAX = 3


BBox = tuple[int, int, int, int]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _frame_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_nested_frames(grid: np.ndarray) -> list[BBox]:
    frame_mask = grid == FRAME_COLOR
    comps = _components(frame_mask)
    assert comps, "expected at least one grey frame"

    boxes: list[BBox] = []
    used_sizes: set[tuple[int, int]] = set()
    for comp in comps:
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have nonempty interior"
        comp_mask = np.zeros_like(frame_mask, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        assert np.array_equal(comp_mask, _frame_mask(grid.shape, box)), "grey component must be a 1-cell rectangle"
        size = (int(r1 - r0 + 1), int(c1 - c0 + 1))
        assert size not in used_sizes, "frame sizes must be unique"
        used_sizes.add(size)
        boxes.append(box)

    boxes.sort(key=_bbox_area, reverse=True)
    for outer, inner in zip(boxes, boxes[1:]):
        or0, oc0, or1, oc1 = outer
        ir0, ic0, ir1, ic1 = inner
        assert ir0 >= or0 + GAP_MIN and ic0 >= oc0 + GAP_MIN
        assert ir1 <= or1 - GAP_MIN and ic1 <= oc1 - GAP_MIN
    return boxes


def _layer_mask(shape: tuple[int, int], boxes: list[BBox], idx: int) -> np.ndarray:
    r0, c0, r1, c1 = boxes[idx]
    mask = np.zeros(shape, dtype=bool)
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    if idx + 1 < len(boxes):
        ir0, ic0, ir1, ic1 = boxes[idx + 1]
        mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    return mask


def _minority_after_swap(grid: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    values = list(map(int, grid[mask]))
    colors = sorted(set(values))
    assert len(colors) == 2, "each layer region must contain exactly two colors"
    assert BACKGROUND not in colors and FRAME_COLOR not in colors
    counts = Counter(values)
    a, b = colors
    assert counts[a] != counts[b], "two colors must have unequal counts"
    swapped_counts = {a: counts[b], b: counts[a]}
    if swapped_counts[a] > swapped_counts[b]:
        return int(a), int(b)
    return int(b), int(a)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the two fill colors inside each nested rectangular layer and compress the layers into a depth-ordered color row.

    Input structure:
    - The grid has a black background and 3-4 disconnected one-cell-thick grey rectangular frames.
    - The grey frames are strictly nested, and each frame encloses exactly one ring-like layer region.
    - Every layer region contains exactly two non-black, non-grey colors with unequal counts.

    Rule:
    - Detect the grey frames and sort them by bounding-box area from largest to smallest.
    - For each frame, take its strict interior and remove the next inner frame together with everything inside that inner frame; for the innermost frame, use its full strict interior.
    - Swap the two colors inside that layer region, then keep only the post-swap majority color as the representative for that depth.
    - Output a 1xN row listing the representative colors from outermost layer to innermost layer.

    Key insight:
    - The swap reverses which color is the local majority, so the correct representative for each layer is the original minority color, read in nesting order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    boxes = _find_nested_frames(g)
    out = np.zeros((1, len(boxes)), dtype=int)
    for idx in range(len(boxes)):
        mask = _layer_mask(g.shape, boxes, idx)
        assert np.any(mask), "layer region must be nonempty"
        representative, _other = _minority_after_swap(g, mask)
        out[0, idx] = representative
    return out


def _sample_boxes(n_layers: int) -> tuple[int, int, list[BBox]] | None:
    if n_layers == 4:
        inner_h = random.randint(INNER_FRAME_MIN, INNER_FRAME_MIN + 1)
        inner_w = random.randint(INNER_FRAME_MIN, INNER_FRAME_MIN + 1)
        grow_hi = GAP_MIN + 1
        margin_hi = OUTER_MARGIN_MIN + 1
    else:
        inner_h = random.randint(INNER_FRAME_MIN, INNER_FRAME_MAX)
        inner_w = random.randint(INNER_FRAME_MIN, INNER_FRAME_MAX)
        grow_hi = GAP_MAX
        margin_hi = OUTER_MARGIN_MAX
    boxes_inner_to_outer: list[BBox] = [(0, 0, inner_h - 1, inner_w - 1)]

    for _ in range(n_layers - 1):
        r0, c0, r1, c1 = boxes_inner_to_outer[-1]
        grow_top = random.randint(GAP_MIN, grow_hi)
        grow_bottom = random.randint(GAP_MIN, grow_hi)
        grow_left = random.randint(GAP_MIN, grow_hi)
        grow_right = random.randint(GAP_MIN, grow_hi)
        boxes_inner_to_outer.append(
            (r0 - grow_top, c0 - grow_left, r1 + grow_bottom, c1 + grow_right)
        )

    outer = boxes_inner_to_outer[-1]
    top_margin = random.randint(OUTER_MARGIN_MIN, margin_hi)
    left_margin = random.randint(OUTER_MARGIN_MIN, margin_hi)
    bottom_margin = random.randint(OUTER_MARGIN_MIN, margin_hi)
    right_margin = random.randint(OUTER_MARGIN_MIN, margin_hi)

    shift_r = -outer[0] + top_margin
    shift_c = -outer[1] + left_margin
    boxes_shifted = [
        (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)
        for r0, c0, r1, c1 in boxes_inner_to_outer
    ]
    outer_shifted = boxes_shifted[-1]
    height = outer_shifted[2] + 1 + bottom_margin
    width = outer_shifted[3] + 1 + right_margin
    if max(height, width) > 30:
        return None

    return int(height), int(width), list(reversed(boxes_shifted))


def _minority_count(area: int) -> int:
    lo = max(1, int(np.ceil(area * 0.28)))
    hi = min(area - 1, int(np.floor(area * 0.42)))
    if hi < lo:
        lo = max(1, area // 3)
        hi = max(lo, area // 2 - 1)
    return random.randint(lo, hi)


def _pick_subset(coords: list[tuple[int, int]], count: int, style: str) -> set[tuple[int, int]]:
    assert 0 < count < len(coords)
    if style == "checker":
        phase = random.randint(0, 1)
        pivot = random.choice(coords)
        ordered = sorted(
            coords,
            key=lambda rc: (
                ((rc[0] + rc[1]) & 1) != phase,
                abs(rc[0] - pivot[0]) + abs(rc[1] - pivot[1]),
                random.random(),
            ),
        )
        return set(ordered[:count])

    if style == "diagonal":
        a, b = random.choice([(1, 1), (1, -1), (2, 1), (1, 2)])
        sign = random.choice([-1, 1])
        ordered = sorted(
            coords,
            key=lambda rc: (
                sign * (a * rc[0] + b * rc[1]),
                ((rc[0] + rc[1]) & 1),
                random.random(),
            ),
        )
        return set(ordered[:count])

    seeds = random.sample(coords, k=min(2, len(coords)))
    counter = random.choice(coords)
    ordered = sorted(
        coords,
        key=lambda rc: (
            min(abs(rc[0] - sr) + abs(rc[1] - sc) for sr, sc in seeds)
            - abs(rc[0] - counter[0])
            - abs(rc[1] - counter[1]),
            ((rc[0] ^ rc[1]) & 1),
            random.random(),
        ),
    )
    return set(ordered[:count])


def _paint_layer(
    grid: np.ndarray,
    mask: np.ndarray,
    representative: int,
    companion: int,
    style: str,
) -> tuple[int, int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    area = len(coords)
    assert area >= 4
    minority = representative
    majority = companion
    minority_cells = _pick_subset(coords, _minority_count(area), style)
    for r, c in coords:
        grid[r, c] = minority if (r, c) in minority_cells else majority
    counts = Counter(map(int, grid[mask]))
    return int(counts[minority]), int(counts[majority])


def _expected_from_boxes(grid: np.ndarray, boxes: list[BBox]) -> np.ndarray:
    out = np.zeros((1, len(boxes)), dtype=int)
    for idx in range(len(boxes)):
        rep, _ = _minority_after_swap(grid, _layer_mask(grid.shape, boxes, idx))
        out[0, idx] = rep
    return out


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30

    boxes = _find_nested_frames(grid)
    assert MIN_LAYERS <= len(boxes) <= MAX_LAYERS

    fill_mask = np.zeros_like(grid, dtype=bool)
    original_majorities: list[int] = []
    original_minorities: list[int] = []

    for idx, _box in enumerate(boxes):
        layer = _layer_mask(grid.shape, boxes, idx)
        fill_mask |= layer
        values = list(map(int, grid[layer]))
        colors = sorted(set(values))
        assert len(colors) == 2
        assert BACKGROUND not in colors and FRAME_COLOR not in colors
        counts = Counter(values)
        a, b = colors
        assert counts[a] != counts[b]
        majority = int(a if counts[a] > counts[b] else b)
        minority = int(b if majority == a else a)
        frac = max(counts[a], counts[b]) / len(values)
        assert 0.56 <= frac <= 0.72, "layer majority should stay visibly unequal"
        original_majorities.append(majority)
        original_minorities.append(minority)

    assert np.array_equal(fill_mask, (grid != BACKGROUND) & (grid != FRAME_COLOR))
    assert len(set(original_minorities)) == len(original_minorities), "output colors should stay distinct"

    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual[0], np.array(original_minorities, dtype=int))
    assert all(int(actual[0, i]) != int(original_majorities[i]) for i in range(len(boxes)))
    assert list(actual[0]) != list(reversed(actual[0])), "depth order must matter"
    assert np.array_equal(actual, _expected_from_boxes(grid, boxes))


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    n_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    sampled = _sample_boxes(n_layers)
    if sampled is None:
        return None
    height, width, boxes = sampled

    representatives = random.sample(FILL_COLORS, n_layers)
    companions = random.sample([c for c in FILL_COLORS if c not in representatives], n_layers)
    styles = [random.choice(["checker", "diagonal", "cluster"]) for _ in range(n_layers)]
    if len(set(styles)) == 1 and n_layers > 1:
        styles[-1] = random.choice([s for s in ("checker", "diagonal", "cluster") if s != styles[0]])

    grid = np.zeros((height, width), dtype=int)
    for idx in range(n_layers):
        layer = _layer_mask(grid.shape, boxes, idx)
        minority_count, majority_count = _paint_layer(
            grid,
            layer,
            representatives[idx],
            companions[idx],
            styles[idx],
        )
        assert minority_count < majority_count

    for box in boxes:
        grid[_frame_mask(grid.shape, box)] = FRAME_COLOR

    expected = np.array([representatives], dtype=int)
    try:
        _validate_generated(grid, expected)
    except AssertionError:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
