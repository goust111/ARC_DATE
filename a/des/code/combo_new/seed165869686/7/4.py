from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
RING_COLOR = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400
LOCAL_SHAPE_ATTEMPTS = 150


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: dict[int, list[tuple[int, int]]] = {}
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            q = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            comps[next_id] = coords
    return labels, comps


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _boundary_mask(shape: np.ndarray) -> np.ndarray:
    h, w = shape.shape
    ring = np.zeros_like(shape, dtype=bool)
    for r, c in zip(*np.where(shape)):
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w and shape[nr, nc]):
                ring[r, c] = True
                break
    return ring


def _ring_and_cavity_from_shape(shape: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    ring = _boundary_mask(shape)
    cavity = shape & ~ring
    if cavity.sum() == 0:
        return None

    _, ring_comps = _label_components(ring)
    _, cavity_comps = _label_components(cavity)
    if len(ring_comps) != 1 or len(cavity_comps) != 1:
        return None

    _, bg_comps = _label_components(~ring)
    enclosed = [coords for coords in bg_comps.values() if not _touches_edge(coords, *ring.shape)]
    if len(enclosed) != 1:
        return None

    if set(enclosed[0]) != set(next(iter(cavity_comps.values()))):
        return None
    return ring, cavity


def _rectangle_ring(target: int) -> tuple[np.ndarray, np.ndarray] | None:
    options: list[tuple[np.ndarray, np.ndarray]] = []
    for h in range(3, 7):
        for w in range(3, 7):
            if 2 * (h + w) - 4 != target:
                continue
            shape = np.ones((h, w), dtype=bool)
            result = _ring_and_cavity_from_shape(shape)
            if result is not None:
                options.append(result)
    if not options:
        return None
    ring, cavity = random.choice(options)
    return ring.copy(), cavity.copy()


def _random_interval_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 6)
    canvas_w = rng.randint(3, 6)
    left = rng.randint(0, canvas_w - 3)
    right = rng.randint(left + 2, canvas_w - 1)
    intervals = [(left, right)]

    for _ in range(1, h):
        prev_left, prev_right = intervals[-1]
        candidates = []
        for new_left in range(canvas_w):
            for new_right in range(new_left + 2, canvas_w):
                if abs(new_left - prev_left) > 1 or abs(new_right - prev_right) > 1:
                    continue
                if new_right < prev_left or new_left > prev_right:
                    continue
                candidates.append((new_left, new_right))
        if not candidates:
            return np.zeros((0, 0), dtype=bool)
        intervals.append(rng.choice(candidates))

    shape = np.zeros((h, canvas_w), dtype=bool)
    for r, (c0, c1) in enumerate(intervals):
        shape[r, c0 : c1 + 1] = True
    return _trim_mask(shape)


def _blob_ring(target: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(LOCAL_SHAPE_ATTEMPTS):
        shape = _random_interval_shape(rng)
        if shape.size == 0:
            continue
        result = _ring_and_cavity_from_shape(shape)
        if result is None:
            continue
        ring, cavity = result
        if int(ring.sum()) == target:
            return ring.copy(), cavity.copy()
    return None


def _sample_local_ring(target: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    families = ["blob"]
    if target in (8, 10):
        families.append("rect")
    rng.shuffle(families)
    for family in families:
        if family == "rect":
            result = _rectangle_ring(target)
        else:
            result = _blob_ring(target, rng)
        if result is not None:
            ring, cavity = result
            if rng.random() < 0.5:
                ring = np.fliplr(ring)
                cavity = np.fliplr(cavity)
            k = rng.randrange(4)
            ring = np.rot90(ring, k).copy()
            cavity = np.rot90(cavity, k).copy()
            return ring, cavity
    return None


def _background_from_corners(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return Counter(corners).most_common(1)[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey ring's interior using the bottom legend color indexed by ring perimeter.

    Input structure:
    - The background is black, and the bottom row is a left-to-right legend strip of distinct colors.
    - Above the legend are several separate grey 4-connected closed rings with black interiors.
    - Rings do not touch one another, and each enclosed interior is a background component.

    Rule:
    - Read the bottom row as a 1-based color legend, where position k stores the color for index k.
    - Count the number of grey pixels in each ring to get its perimeter value k.
    - Find enclosed black regions whose boundary touches exactly one grey ring and fill that region with the legend's kth color, leaving the grey ring itself unchanged.

    Key insight:
    The bottom strip is an index legend, and the lookup key is not a color or shape type but the ring's visible grey-pixel perimeter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    bg = _background_from_corners(g)
    legend = np.array(g[-1], dtype=int)
    assert len(set(int(x) for x in legend)) == legend.size

    field = g[:-1]
    ring_colors = [int(color) for color in np.unique(field) if int(color) != bg]
    assert len(ring_colors) == 1, "expected exactly one non-background ring color"
    ring_color = ring_colors[0]

    ring_labels, ring_comps = _label_components(field == ring_color)
    bg_labels, bg_comps = _label_components(field == bg)

    out = g.copy()
    field_h, field_w = field.shape
    ring_sizes = {comp_id: len(coords) for comp_id, coords in ring_comps.items()}

    for coords in bg_comps.values():
        if _touches_edge(coords, field_h, field_w):
            continue
        adjacent_rings = set()
        for r, c in coords:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < field_h and 0 <= nc < field_w and field[nr, nc] == ring_color:
                    adjacent_rings.add(int(ring_labels[nr, nc]))
        if len(adjacent_rings) != 1:
            continue
        ring_id = next(iter(adjacent_rings))
        perimeter = ring_sizes[ring_id]
        assert 1 <= perimeter <= legend.size, "ring perimeter exceeds legend length"
        out_coords = np.array(coords, dtype=int)
        out[out_coords[:, 0], out_coords[:, 1]] = int(legend[perimeter - 1])

    return out


def _sample_legend(width: int, rng: random.Random) -> np.ndarray:
    colors = list(range(1, 10)) if width == 9 else list(range(10))
    active_positions = list(range(7, width))
    while True:
        rng.shuffle(colors)
        if all(colors[pos] not in (BG_COLOR, RING_COLOR) for pos in active_positions):
            return np.array(colors, dtype=int)


def _sample_perimeters(width: int, count: int, rng: random.Random) -> list[int]:
    allowed = list(range(8, width + 1))
    want_distinct = min(count, len(allowed), 2)
    for _ in range(50):
        perimeters = [rng.choice(allowed) for _ in range(count)]
        if len(set(perimeters)) >= want_distinct:
            return perimeters
    return [rng.choice(allowed) for _ in range(count)]


def _place_rings(
    grid_h: int,
    grid_w: int,
    rings: list[tuple[np.ndarray, np.ndarray]],
    legend: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    field_h = grid_h - 1
    inp = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    out = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    inp[-1] = legend
    out[-1] = legend

    reserved = np.zeros((field_h, grid_w), dtype=bool)
    for ring, cavity in rings:
        patch_h, patch_w = ring.shape
        spots = []
        for r0 in range(1, field_h - patch_h):
            for c0 in range(1, grid_w - patch_w):
                if reserved[r0 - 1 : r0 + patch_h + 1, c0 - 1 : c0 + patch_w + 1].any():
                    continue
                spots.append((r0, c0))
        if not spots:
            return None

        r0, c0 = rng.choice(spots)
        inp_patch = inp[r0 : r0 + patch_h, c0 : c0 + patch_w]
        out_patch = out[r0 : r0 + patch_h, c0 : c0 + patch_w]
        inp_patch[ring] = RING_COLOR
        out_patch[ring] = RING_COLOR
        out_patch[cavity] = int(legend[int(ring.sum()) - 1])
        reserved[r0 - 1 : r0 + patch_h + 1, c0 - 1 : c0 + patch_w + 1] = True

    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_w = rng.choice([9, 10])
    grid_h = rng.randint(8, 15)
    legend = _sample_legend(grid_w, rng)

    count_choices = [1, 2]
    if grid_h >= 11:
        count_choices.append(3)
    ring_count = rng.choice(count_choices)
    perimeters = _sample_perimeters(grid_w, ring_count, rng)

    rings: list[tuple[np.ndarray, np.ndarray]] = []
    for perimeter in perimeters:
        result = _sample_local_ring(perimeter, rng)
        if result is None:
            return None
        rings.append(result)

    placed = _place_rings(grid_h, grid_w, rings, legend, rng)
    if placed is None:
        return None
    inp, expected = placed

    actual = main(inp)
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
