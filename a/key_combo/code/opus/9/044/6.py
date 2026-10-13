from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BACKGROUND = 0
DEFECT = 7
RECT_COLORS = (1, 3, 4, 5)
DIR_BY_MOD = {
    0: (-1, 0),   # up
    1: (0, 1),    # right
    2: (1, 0),    # down
    3: (0, -1),   # left
}
THREE_RECT_SUBSETS = (
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 4),
    (2, 3, 4),
)
GEN_ATTEMPTS = 500

_GEN_COUNTER = 0
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _shift_box(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    return box[0] + dr, box[1] + dc, box[2] + dr, box[3] + dc


def _in_bounds_box(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    return 0 <= box[0] <= box[2] < h and 0 <= box[1] <= box[3] < w


def _eligible_defect_cells(height: int, width: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            same_neighbors = 0
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width:
                    same_neighbors += 1
            if same_neighbors >= 3:
                cells.append((r, c))
    return cells


def _non_adjacent(cells: tuple[tuple[int, int], ...]) -> bool:
    for i, (r0, c0) in enumerate(cells):
        for r1, c1 in cells[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) == 1:
                return False
    return True


def _build_defect_patterns() -> dict[tuple[int, int, int], list[tuple[tuple[int, int], ...]]]:
    patterns: dict[tuple[int, int, int], list[tuple[tuple[int, int], ...]]] = {}
    for height in range(2, 5):
        for width in range(3, 6):
            candidates = _eligible_defect_cells(height, width)
            for count in range(1, 5):
                valid = [
                    combo
                    for combo in combinations(candidates, count)
                    if _non_adjacent(combo)
                ]
                if valid:
                    patterns[(height, width, count)] = valid
    return patterns


DEFECT_PATTERNS = _build_defect_patterns()
SIZE_OPTIONS_BY_COUNT = {
    count: [dims for dims in [(h, w) for h in range(2, 5) for w in range(3, 6)] if (dims[0], dims[1], count) in DEFECT_PATTERNS]
    for count in range(1, 5)
}


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in {BACKGROUND, DEFECT})
    assert 3 <= len(colors) <= 4, "expected 3 or 4 main rectangles"

    covered_defects = np.zeros(g.shape, dtype=int)
    rects: list[dict] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape[0] in {2, 3, 4}, "rectangle height must be 2..4"
        assert patch.shape[1] in {3, 4, 5}, "rectangle width must be 3..5"
        assert np.all((patch == color) | (patch == DEFECT)), "bbox must be solid except orange defects"

        defect_local = np.argwhere(patch == DEFECT)
        defect_count = int(defect_local.shape[0])
        assert 1 <= defect_count <= 4, "each rectangle needs 1..4 defects"
        assert (patch.shape[0], patch.shape[1], defect_count) in DEFECT_PATTERNS

        for lr, lc in defect_local:
            rr = r0 + int(lr)
            cc = c0 + int(lc)
            same_neighbors = 0
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == color:
                    same_neighbors += 1
            assert same_neighbors >= 3, "defects must be embedded in the rectangle"
            covered_defects[rr, cc] += 1

        rects.append(
            {
                "color": color,
                "bbox": bbox,
                "height": int(patch.shape[0]),
                "width": int(patch.shape[1]),
                "defect_count": defect_count,
            }
        )

    defect_mask = g == DEFECT
    assert np.array_equal(defect_mask, covered_defects == 1), "every orange pixel must belong to exactly one rectangle"
    mods = [rect["defect_count"] % 4 for rect in rects]
    assert len(set(mods)) == len(mods), "defect-count directions must be unique"
    rects.sort(key=lambda rect: (rect["bbox"][1], rect["bbox"][0]))
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair embedded orange defects and move each rectangle by its own height in the defect-count direction.

    Input structure:
    - A black background contains 3 or 4 separated solid rectangles in distinct colors.
    - Each rectangle is axis-aligned and monochrome except for 1 to 4 embedded orange defect pixels.
    - Every orange pixel lies inside one rectangle and has at least three 4-neighbors of that rectangle's main color.

    Rule:
    - For each rectangle, count its orange defects and repaint those defect cells to the rectangle's main color.
    - Map the defect count modulo 4 to a direction: 0=up, 1=right, 2=down, 3=left.
    - Move the repaired whole rectangle by exactly its own height in that direction.
    - Clear the original position to black and keep only the translated repaired rectangles in the output.

    Key insight:
    The same orange pixels play two roles at once: they are visual defects that must be repaired, and their count encodes the rectangle's translation direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects = _extract_rectangles(g)
    out = np.full_like(g, BACKGROUND)
    occupied = np.zeros(g.shape, dtype=bool)

    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        height = rect["height"]
        width = rect["width"]
        color = rect["color"]
        dr, dc = DIR_BY_MOD[rect["defect_count"] % 4]

        dst_r0 = r0 + dr * height
        dst_c0 = c0 + dc * height
        dst_box = (dst_r0, dst_c0, dst_r0 + height - 1, dst_c0 + width - 1)
        assert _in_bounds_box(dst_box, g.shape), "translated rectangle must stay on-grid"
        rr0, cc0, rr1, cc1 = dst_box
        assert not np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]), "translated rectangles must not overlap"

        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = color
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    return out


def _choose_counts() -> list[int]:
    global _GEN_COUNTER
    idx = _GEN_COUNTER
    _GEN_COUNTER += 1
    if idx % 2 == 0:
        counts = [1, 2, 3, 4]
    else:
        counts = list(THREE_RECT_SUBSETS[(idx // 2) % len(THREE_RECT_SUBSETS)])
    random.shuffle(counts)
    return counts


def _choose_specs(counts: list[int]) -> list[dict] | None:
    for _ in range(120):
        specs = []
        heights = []
        for count in counts:
            height, width = random.choice(SIZE_OPTIONS_BY_COUNT[count])
            defects = random.choice(DEFECT_PATTERNS[(height, width, count)])
            specs.append(
                {
                    "count": count,
                    "height": height,
                    "width": width,
                    "defects": defects,
                }
            )
            heights.append(height)
        if len(set(heights)) >= 2:
            return specs
    return None


def _fits_with_clearance(
    box: tuple[int, int, int, int],
    dst: tuple[int, int, int, int],
    height: int,
    placed: list[dict],
    shape: tuple[int, int],
) -> bool:
    if not _in_bounds_box(box, shape) or not _in_bounds_box(dst, shape):
        return False

    expanded = _expand_box(box, height)
    for other in placed:
        other_box = other["bbox"]
        other_dst = other["dst_bbox"]
        other_height = other["height"]
        other_expanded = _expand_box(other_box, other_height)

        if _boxes_overlap(box, other_box):
            return False
        if _boxes_overlap(expanded, other_box) or _boxes_overlap(box, other_expanded):
            return False
        if _boxes_overlap(dst, other_box) or _boxes_overlap(box, other_dst):
            return False
        if _boxes_overlap(dst, other_dst):
            return False

    return True


def _place_rectangles(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    h_grid, w_grid = shape
    order = list(range(len(specs)))
    random.shuffle(order)
    placed: list[dict | None] = [None] * len(specs)
    active: list[dict] = []

    for idx in order:
        spec = specs[idx]
        height = spec["height"]
        width = spec["width"]
        dr, dc = DIR_BY_MOD[spec["count"] % 4]

        r_low = height
        r_high = h_grid - 2 * height
        c_low = height
        c_high = w_grid - width - height
        if r_low > r_high or c_low > c_high:
            return None

        placed_here = None
        for _ in range(300):
            r0 = random.randint(r_low, r_high)
            c0 = random.randint(c_low, c_high)
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            dst = _shift_box(box, dr * height, dc * height)
            if _fits_with_clearance(box, dst, height, active, shape):
                placed_here = {
                    **spec,
                    "bbox": box,
                    "dst_bbox": dst,
                }
                break
        if placed_here is None:
            return None

        placed[idx] = placed_here
        active.append(placed_here)

    return [spec for spec in placed if spec is not None]


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    colors = list(RECT_COLORS)
    random.shuffle(colors)
    chosen_colors = colors[: len(specs)]
    random.shuffle(chosen_colors)

    for spec, color in zip(specs, chosen_colors):
        spec["color"] = color
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        for dr, dc in spec["defects"]:
            grid[r0 + dr, c0 + dc] = DEFECT
    return grid


def _expected_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["dst_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = spec["color"]
    return out


def _semantic_checks(inp: np.ndarray, specs: list[dict], expected: np.ndarray) -> bool:
    non_bg_colors = [int(color) for color in np.unique(inp) if int(color) != BACKGROUND]
    if int((inp == BACKGROUND).sum()) <= max(int((inp == color).sum()) for color in non_bg_colors):
        return False

    if len({spec["color"] for spec in specs}) != len(specs):
        return False
    if len({spec["count"] % 4 for spec in specs}) != len(specs):
        return False
    if len({spec["height"] for spec in specs}) < 2:
        return False

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        color = spec["color"]
        patch = inp[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all((patch == color) | (patch == DEFECT)):
            return False
        if int(np.count_nonzero(patch == DEFECT)) != spec["count"]:
            return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    out_sig = (tuple(int(x) for x in expected.shape), expected.tobytes())
    if out_sig in _SEEN_OUTPUTS:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_shape = (
        random.randint(22, 28),
        random.randint(22, 28),
    )
    counts = _choose_counts()
    specs = _choose_specs(counts)
    if specs is None:
        return None

    specs = _place_rectangles(grid_shape, specs)
    if specs is None:
        return None

    grid = _render_input(grid_shape, specs)
    expected = _expected_output(grid_shape, specs)
    if not _semantic_checks(grid, specs, expected):
        return None

    _SEEN_OUTPUTS.add((tuple(int(x) for x in expected.shape), expected.tobytes()))
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
