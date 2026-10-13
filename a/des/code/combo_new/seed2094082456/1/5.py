import random

import numpy as np


ARC_GRID_MAX = 30
BG = 4
NON_BG_COLORS = tuple(color for color in range(10) if color != BG)
RING_COUNT_VALUES = (1, 2, 3, 4)
OBJECT_COUNT_VALUES = (2, 3, 4)
CANVAS_SIZE = ARC_GRID_MAX
MAX_CROP_MARGIN = 6
MAX_GENERATION_ATTEMPTS = 400
MAX_PADDED_AREA = 700
INNER_CAPS_BY_COUNT = {
    2: {1: 8, 2: 7, 3: 6, 4: 5},
    3: {1: 7, 2: 6, 3: 5, 4: 4},
    4: {1: 6, 2: 5, 3: 4, 4: 3},
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

    return count


def _extract_objects(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    objects = []
    for color in (int(c) for c in np.unique(grid) if int(c) != bg):
        mask = grid == color
        if not np.any(mask):
            continue
        rows, cols = np.where(mask)
        ring_count = _count_components(mask)
        objects.append((ring_count, int(cols.min()), int(rows.min()), color))
    return objects


def _render_output(objects: list[tuple[int, int, int, int]], bg: int) -> np.ndarray:
    if not objects:
        return np.full((1, 1), bg, dtype=int)

    ordered = sorted(objects, key=lambda item: (item[0], item[1], item[2]))
    out_h = max(ring_count for ring_count, _, _, _ in ordered)
    out_w = sum(ring_count for ring_count, _, _, _ in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for ring_count, _, _, color in ordered:
        out[out_h - ring_count :, col : col + ring_count] = color
        col += ring_count + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered concentric ring objects by layer count and replace each with a solid square of that size.

    Input structure:
    - The background is yellow.
    - Each foreground object uses one non-yellow color and consists of 1-4 nested hollow rectangular rings.
    - Rings of the same object are separated by a one-cell yellow gap, so a multi-ring object appears as several disconnected same-color rectangles.

    Rule:
    - For each color, count how many disconnected rectangular rings it contains; that count is the object's layer count K.
    - Sort the objects by increasing K, breaking ties by their left-to-right order in the input.
    - Build a yellow output grid whose height is the largest K, and replace each object by a solid K x K square of the same color.
    - Arrange the squares in one row with one yellow column between neighbors and align all square bottoms.

    Key insight:
    - The number of disconnected same-color rectangular rings is the hidden size code: it simultaneously determines the sort key and the side length of the simplified output square.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else _most_common_color(g)
    objects = _extract_objects(g, bg)
    return _render_output(objects, bg)


def _make_ring_object(ring_count: int, inner_h: int, inner_w: int, color: int) -> np.ndarray:
    h = inner_h + 4 * (ring_count - 1)
    w = inner_w + 4 * (ring_count - 1)
    obj = np.full((h, w), BG, dtype=int)

    for layer in range(ring_count):
        r0 = 2 * layer
        c0 = 2 * layer
        r1 = h - 1 - 2 * layer
        c1 = w - 1 - 2 * layer
        obj[r0, c0 : c1 + 1] = color
        obj[r1, c0 : c1 + 1] = color
        obj[r0 : r1 + 1, c0] = color
        obj[r0 : r1 + 1, c1] = color

    return obj


def _sample_ring_counts(n_objects: int, rng: random.Random) -> list[int]:
    if rng.random() < 0.5:
        while True:
            ring_counts = [rng.choice(RING_COUNT_VALUES) for _ in range(n_objects)]
            if len(set(ring_counts)) < len(ring_counts):
                return ring_counts
    return rng.sample(RING_COUNT_VALUES, n_objects)


def _sample_specs(rng: random.Random) -> list[dict[str, object]] | None:
    n_objects = rng.choice(OBJECT_COUNT_VALUES)
    ring_counts = _sample_ring_counts(n_objects, rng)
    colors = rng.sample(NON_BG_COLORS, n_objects)
    caps = INNER_CAPS_BY_COUNT[n_objects]

    specs = []
    for color, ring_count in zip(colors, ring_counts):
        cap = caps[ring_count]
        inner_h = rng.randint(3, cap)
        inner_w = rng.randint(3, cap)
        if cap > 3 and rng.random() < 0.35:
            inner_h = 3
        if cap > 3 and rng.random() < 0.35:
            inner_w = 3
        obj = _make_ring_object(ring_count, inner_h, inner_w, color)
        h, w = obj.shape
        specs.append(
            {
                "color": color,
                "ring_count": ring_count,
                "array": obj,
                "h": h,
                "w": w,
            }
        )

    padded_area = sum((int(spec["h"]) + 2) * (int(spec["w"]) + 2) for spec in specs)
    if padded_area > MAX_PADDED_AREA:
        return None

    return specs


def _place_specs(specs: list[dict[str, object]], rng: random.Random) -> np.ndarray | None:
    grid = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    reserved = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    used_lefts = set()

    order = sorted(
        range(len(specs)),
        key=lambda idx: int(specs[idx]["h"]) * int(specs[idx]["w"]),
        reverse=True,
    )

    for idx in order:
        spec = specs[idx]
        obj = spec["array"]
        h = int(spec["h"])
        w = int(spec["w"])
        positions = []
        for r0 in range(1, CANVAS_SIZE - h):
            for c0 in range(1, CANVAS_SIZE - w):
                if c0 in used_lefts:
                    continue
                if reserved[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                    continue
                positions.append((r0, c0))

        if not positions:
            return None

        r0, c0 = rng.choice(positions)
        grid[r0 : r0 + h, c0 : c0 + w] = obj
        reserved[r0 : r0 + h, c0 : c0 + w] = True
        used_lefts.add(c0)
        spec["r0"] = r0
        spec["c0"] = c0

    return grid


def _crop_canvas(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    rows, cols = np.where(grid != BG)
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())

    top_margin = rng.randint(1, min(MAX_CROP_MARGIN, rmin))
    bottom_margin = rng.randint(1, min(MAX_CROP_MARGIN, CANVAS_SIZE - 1 - rmax))
    left_margin = rng.randint(1, min(MAX_CROP_MARGIN, cmin))
    right_margin = rng.randint(1, min(MAX_CROP_MARGIN, CANVAS_SIZE - 1 - cmax))

    return grid[
        rmin - top_margin : rmax + bottom_margin + 1,
        cmin - left_margin : cmax + right_margin + 1,
    ]


def _expected_output(specs: list[dict[str, object]]) -> np.ndarray:
    objects = [
        (
            int(spec["ring_count"]),
            int(spec["c0"]),
            int(spec["r0"]),
            int(spec["color"]),
        )
        for spec in specs
    ]
    return _render_output(objects, BG)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    specs = _sample_specs(rng)
    if specs is None:
        return None

    canvas = _place_specs(specs, rng)
    if canvas is None:
        return None

    grid = _crop_canvas(canvas, rng)
    expected = _expected_output(specs)

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
