import numpy as np

from arc_utils import find_components, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cycle each solid odd-sided square to the next smaller centered square, wrapping a single pixel to a 9x9 square.

    Input structure:
    - The grid has one dominant background color.
    - Every foreground object is a separated 4-connected monochrome solid square.
    - Square side lengths are odd and belong to the cycle 1, 3, 5, 7, 9.

    Rule:
    - Clear the canvas to the background.
    - Keep each square's color and center.
    - Replace side length n by n - 2 when n > 1; replace side length 1 by side length 9.

    Key insight:
    - The visible operation is on square size, not color: side lengths cycle 1 -> 9 -> 7 -> 5 -> 3 -> 1.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert components, "expected at least one foreground square"

    out = np.full_like(g, bg)
    painted = np.zeros(g.shape, dtype=bool)
    H, W = g.shape

    for obj in components:
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        side = int(r1 - r0 + 1)
        assert side == int(c1 - c0 + 1), "foreground object must have a square bbox"
        assert side in {1, 3, 5, 7, 9}, "square side length must be in the odd-size cycle"
        assert int(obj["area"]) == side * side, "foreground object must fill its bbox"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "foreground object must be solid"

        new_side = 9 if side == 1 else side - 2
        center_r = int((r0 + r1) // 2)
        center_c = int((c0 + c1) // 2)
        nr0 = center_r - new_side // 2
        nc0 = center_c - new_side // 2
        nr1 = nr0 + new_side
        nc1 = nc0 + new_side
        assert 0 <= nr0 < nr1 <= H and 0 <= nc0 < nc1 <= W, "transformed square must fit in the grid"

        region = (slice(nr0, nr1), slice(nc0, nc1))
        assert not painted[region].any(), "transformed squares must not overlap"
        out[region] = color
        painted[region] = True

    return out

# --------------- generator (att2, o5x) ---------------

# Canvas and palette.
GRID_SIZE = 16
BACKGROUND_COLOR = 7
FOREGROUND_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]
INPUT_SIDE_FROM_OUTPUT = {1: 3, 3: 5, 5: 7, 7: 9, 9: 1}
# Object and side sampling.
SQUARE_COUNT_RANGE = (3, 5)
BASE_OUTPUT_SIDE_CHOICES = [1, 3, 5]
OUTPUT_SIDE_7_PROB = 0.45
OUTPUT_SIDE_9_PROB = 0.35
# Placement and retry policy.
OUTPUT_CLEARANCE_GAP = 1
INPUT_CLEARANCE_GAP = 0
MAX_GENERATION_ATTEMPTS = 300


def _bbox(center_r, center_c, side):
    d = side // 2
    return center_r - d, center_c - d, center_r + d, center_c + d


def _separated(a, b, gap=0):
    return (
        a[2] + gap + 1 <= b[0]
        or b[2] + gap + 1 <= a[0]
        or a[3] + gap + 1 <= b[1]
        or b[3] + gap + 1 <= a[1]
    )


def _clear(rect, rects, gap=0):
    return all(_separated(rect, other, gap) for other in rects)


def _draw_square(grid, color, box):
    r0, c0, r1, c1 = box
    shape = (r1 - r0 + 1, c1 - c0 + 1)
    paste(grid, np.full(shape, int(color), dtype=int), r0, c0)


def _choose_output_sides(count):
    sides = [random.choice(BASE_OUTPUT_SIDE_CHOICES) for _ in range(count)]
    slots = list(range(count))
    random.shuffle(slots)

    if random.random() < OUTPUT_SIDE_7_PROB:
        sides[slots.pop()] = 7
    if slots and random.random() < OUTPUT_SIDE_9_PROB:
        sides[slots.pop()] = 9

    random.shuffle(sides)
    return sides


def _sample_objects():
    """Sample color and side specs, returning larger required footprints first."""
    count = random.randint(*SQUARE_COUNT_RANGE)
    colors = random.sample(FOREGROUND_COLORS, count)
    out_sides = _choose_output_sides(count)
    in_sides = [INPUT_SIDE_FROM_OUTPUT[s] for s in out_sides]

    # Keep the configured background as the solver's dominant color.
    areas = [s * s for s in in_sides]
    if GRID_SIZE * GRID_SIZE - sum(areas) <= max(areas):
        return None

    objects = [
        {"color": color, "out_side": out_side, "in_side": in_side}
        for color, out_side, in_side in zip(colors, out_sides, in_sides)
    ]
    random.shuffle(objects)
    objects.sort(key=lambda obj: (max(obj["out_side"], obj["in_side"]), obj["out_side"]), reverse=True)
    return objects


def _place_objects(objects):
    """Place each object at a center valid for both input and output squares."""
    placed = []
    output_boxes = []
    input_boxes = []

    for obj in objects:
        out_side = obj["out_side"]
        in_side = obj["in_side"]
        margin = max(out_side // 2, in_side // 2)
        rows = range(margin, GRID_SIZE - margin)
        centers = [(r, c) for r in rows for c in rows]
        random.shuffle(centers)

        for r, c in centers:
            out_box = _bbox(r, c, out_side)
            in_box = _bbox(r, c, in_side)
            if _clear(out_box, output_boxes, OUTPUT_CLEARANCE_GAP) and _clear(in_box, input_boxes, INPUT_CLEARANCE_GAP):
                placed.append((obj["color"], out_box, in_box))
                output_boxes.append(out_box)
                input_boxes.append(in_box)
                break
        else:
            return None
    return placed


def _render_grid(placed, box_index):
    """Render one phase from placed tuples: 1 for output boxes, 2 for input boxes."""
    grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    for item in placed:
        _draw_square(grid, item[0], item[box_index])
    return grid


def _gen_attempt():
    objects = _sample_objects()
    if objects is None:
        return None

    placed = _place_objects(objects)
    if placed is None:
        return None

    out = _render_grid(placed, 1)
    inp = _render_grid(placed, 2)
    return inp if np.array_equal(main(inp), out) else None


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 522fdd07 input")
