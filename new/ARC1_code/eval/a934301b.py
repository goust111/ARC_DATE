from __future__ import annotations

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste, random_free_location
import random


def _sparse_marker_color(g: np.ndarray, bg: int) -> int:
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    marker_count = min(counts.values())
    markers = [color for color, count in counts.items() if count == marker_count]
    assert len(markers) == 1, "sparse marker color must be unique"
    assert any(count > marker_count for count in counts.values()), "expected one color to be denser than the marker"
    return markers[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each foreground object that contains more than one sparse marker-color pixel.

    Input structure
    - A dominant background surrounds separated 4-connected foreground objects.
    - The foreground uses one dense object color and one sparse marker color.
    Rule
    - Infer the marker as the unique less frequent foreground color.
    - Keep objects with zero or one marker pixel.
    - Recolor every pixel of an object to background when it contains two or more marker pixels.
    Key insight
    - The marker count inside each whole object, not the object's size or position, decides whether that object remains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    marker = _sparse_marker_color(g, bg)

    objects = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert objects, "expected foreground objects"

    out = g.copy()
    kept_any = False
    removed_any = False
    for obj in objects:
        coords = obj["coords"]
        marker_count = int(np.count_nonzero(g[coords[:, 0], coords[:, 1]] == marker))
        if marker_count >= 2:
            out[coords[:, 0], coords[:, 1]] = bg
            removed_any = True
        else:
            kept_any = True

    assert kept_any and removed_any, "expected both kept and erased objects"
    return out

# --------------- generator (o5x) ---------------

# Canvas and object geometry.
ROW_RANGE, COL_RANGE = (13, 14), (12, 15)
OBJECT_RANGE = (6, 7)
RECT_H_RANGE, RECT_W_RANGE = (2, 4), (2, 6)
RECT_DIMS = tuple((h, w) for h in range(RECT_H_RANGE[0], RECT_H_RANGE[1] + 1) for w in range(RECT_W_RANGE[0], RECT_W_RANGE[1] + 1))
RECT_MIN_AREA = RECT_H_RANGE[0] * RECT_W_RANGE[0]
RECT_MAX_AREA = RECT_H_RANGE[1] * RECT_W_RANGE[1]

# Marker assignment and foreground density.
KEPT_OBJECT_COUNT = 3
MARKER_TOTAL_RANGE = (9, 14)
KEPT_MARKERS = (0, 1)
REMOVED_MARKERS = (2, 4)
FG_AREA_FACTORS = (5, 9)
FG_AREA_DIVISOR = 2

# Rendering, placement, and retries.
BG_COLOR, DENSE_COLOR, MARKER_COLOR = 0, 1, 8
PLACE_PAD, PLACE_CONN = 1, 8
PLACE_KWARGS = dict(transparent=BG_COLOR, padding=PLACE_PAD, padding_connectivity=PLACE_CONN)
MAX_GENERATION_ATTEMPTS = 300


def _gen_bounded_parts(n: int, total: int, lo: int, hi: int) -> list[int] | None:
    parts: list[int] = []
    remaining = int(total)
    for i in range(int(n)):
        rem = int(n) - i - 1
        low = max(int(lo), remaining - int(hi) * rem)
        high = min(int(hi), remaining - int(lo) * rem)
        if low > high:
            return None
        value = random.randint(low, high)
        parts.append(value)
        remaining -= value
    return parts if remaining == 0 else None


def _gen_marker_counts(n_objects: int, total_markers: int) -> list[tuple[int, bool]] | None:
    kept_count = KEPT_OBJECT_COUNT
    removed_count = int(n_objects) - kept_count
    kept_totals = [
        k
        for k in range(kept_count + 1)
        if REMOVED_MARKERS[0] * removed_count <= int(total_markers) - k <= REMOVED_MARKERS[1] * removed_count
    ]
    if not kept_totals:
        return None

    kept_total = random.choice(kept_totals)
    kept_markers = [KEPT_MARKERS[1]] * kept_total + [KEPT_MARKERS[0]] * (kept_count - kept_total)
    random.shuffle(kept_markers)

    removed_markers = _gen_bounded_parts(removed_count, int(total_markers) - kept_total, *REMOVED_MARKERS)
    if removed_markers is None:
        return None

    objects = [(m, True) for m in kept_markers] + [(m, False) for m in removed_markers]
    random.shuffle(objects)
    return objects


def _gen_rect_dims(n_objects: int, min_area: int, max_area: int) -> list[tuple[int, int]] | None:
    dims: list[tuple[int, int]] = []
    area = 0
    for i in range(int(n_objects)):
        rem = int(n_objects) - i - 1
        feasible = []
        for h, w in RECT_DIMS:
            new_area = area + h * w
            if new_area + rem * RECT_MIN_AREA <= int(max_area) and new_area + rem * RECT_MAX_AREA >= int(min_area):
                feasible.append((h, w))
        if not feasible:
            return None
        h, w = random.choice(feasible)
        dims.append((h, w))
        area += h * w

    if not (int(min_area) <= area <= int(max_area)):
        return None
    random.shuffle(dims)
    return dims


def _gen_pair_dims_counts(dims: list[tuple[int, int]], counts: list[tuple[int, bool]]):
    remaining = list(dims)
    paired: list[tuple[int, int, int, bool]] = []
    for marker_count, keep in sorted(counts, reverse=True):
        compatible = [i for i, (h, w) in enumerate(remaining) if h * w > marker_count]
        if not compatible:
            return None
        idx = random.choice(compatible)
        h, w = remaining.pop(idx)
        paired.append((h, w, marker_count, keep))
    random.shuffle(paired)
    return paired


def _gen_marked_rect(h: int, w: int, marker_count: int) -> np.ndarray:
    rect = np.full((h, w), DENSE_COLOR, dtype=int)
    rect.flat[random.sample(range(h * w), marker_count)] = MARKER_COLOR
    return rect


def _render_specs(rows: int, cols: int, specs: list[tuple[int, int, int, bool]]):
    """Return generated input and expected output, or None if placement fails."""
    inp = np.full((rows, cols), BG_COLOR, dtype=int)
    expected = np.full_like(inp, BG_COLOR)
    occupancy = np.full_like(inp, BG_COLOR)
    order = list(specs)
    random.shuffle(order)
    order.sort(key=lambda item: item[0] * item[1], reverse=True)

    for h, w, marker_count, keep in order:
        patch = np.full((h, w), DENSE_COLOR, dtype=int)
        try:
            r0, c0 = random_free_location(occupancy, patch, **PLACE_KWARGS)
        except ValueError:
            return None

        rect = _gen_marked_rect(h, w, marker_count)
        paste(inp, rect, r0, c0)
        paste(occupancy, patch, r0, c0)
        if keep:
            paste(expected, rect, r0, c0)
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*ROW_RANGE)
    cols = random.randint(*COL_RANGE)
    n_objects = random.randint(*OBJECT_RANGE)
    total_markers = random.randint(*MARKER_TOTAL_RANGE)

    grid_area = rows * cols
    min_area = FG_AREA_FACTORS[0] * total_markers
    max_area = min(FG_AREA_FACTORS[1] * total_markers, (grid_area + total_markers - 1) // FG_AREA_DIVISOR)
    if min_area > max_area:
        return None

    dims = _gen_rect_dims(n_objects, min_area, max_area)
    counts = _gen_marker_counts(n_objects, total_markers)
    if dims is None or counts is None:
        return None
    specs = _gen_pair_dims_counts(dims, counts)
    if specs is None:
        return None

    rendered = _render_specs(rows, cols, specs)
    if rendered is None:
        return None
    inp, expected = rendered

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
