from __future__ import annotations

import random
from collections import Counter

import numpy as np


# Coordinates use the shared ARC convention (row, column), and boxes are
# inclusive (r0, c0, r1, c1), matching arc_utils.py.  The few helpers needed
# here are kept local so SourceFileLoader can load this file without sys.path
# setup for the shared toolbox.

_DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _border(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _side_midpoints(box: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0, (
        "ring height and width must be odd so a side-centered gap has a cardinal direction"
    )
    rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
    return {
        "up": (r0, cm),
        "down": (r1, cm),
        "left": (rm, c0),
        "right": (rm, c1),
    }


def _gap_direction(
    box: tuple[int, int, int, int], gap: tuple[int, int]
) -> tuple[int, int]:
    matches = [name for name, cell in _side_midpoints(box).items() if cell == gap]
    assert len(matches) == 1, "the unique gap must be centered on exactly one side"
    return _DIRS[matches[0]]


def _ring_candidates(g: np.ndarray, bg: int, color: int) -> list[dict]:
    """Enumerate every complete gapped-ring template consistent with one color."""
    h, w = g.shape
    coords = np.argwhere(g == int(color))
    assert coords.size, "every ring color must be visible"
    visible = {tuple(map(int, p)) for p in coords}

    def candidate_for(box: tuple[int, int, int, int]) -> dict | None:
        border = _border(box)
        if not visible <= border:
            return None
        bg_cells = [p for p in border if int(g[p]) == int(bg)]
        if len(bg_cells) != 1:
            return None
        gap = bg_cells[0]
        names = [name for name, p in _side_midpoints(box).items() if p == gap]
        if len(names) != 1:
            return None
        return {
            "color": int(color),
            "box": box,
            "border": border,
            "gap": gap,
            "direction": _DIRS[names[0]],
        }

    # Four visible extrema corners pin all four border coordinates directly.
    # This is the common fast path, but the exhaustive path below also handles
    # valid inputs in which one or more corners were occluded.
    vr0, vc0 = coords.min(axis=0)
    vr1, vc1 = coords.max(axis=0)
    visible_box = (int(vr0), int(vc0), int(vr1), int(vc1))
    visible_corners = {
        (int(vr0), int(vc0)), (int(vr0), int(vc1)),
        (int(vr1), int(vc0)), (int(vr1), int(vc1)),
    }
    if all(int(g[p]) == int(color) for p in visible_corners):
        height, width = int(vr1 - vr0 + 1), int(vc1 - vc0 + 1)
        if height >= 5 and width >= 5 and height % 2 == 1 and width % 2 == 1:
            candidate = candidate_for(visible_box)
            if candidate is not None:
                return [candidate]

    row_pairs = [
        (r0, r1)
        for r0 in range(h)
        for r1 in range(r0 + 4, h)
        if (r1 - r0) % 2 == 0
    ]
    col_pairs = [
        (c0, c1)
        for c0 in range(w)
        for c1 in range(c0 + 4, w)
        if (c1 - c0) % 2 == 0
    ]

    candidates: list[dict] = []
    rr, cc = coords[:, 0], coords[:, 1]
    for r0, r1 in row_pairs:
        row_side = (rr == r0) | (rr == r1)
        for c0, c1 in col_pairs:
            # Every visible pixel of this color must belong to this border.
            if not bool(np.all(row_side | (cc == c0) | (cc == c1))):
                continue
            box = (r0, c0, r1, c1)
            candidate = candidate_for(box)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _analyze(grid: np.ndarray) -> tuple[int, list[dict], list[int]]:
    """Recover rings and the unique top-to-bottom order from visible ownership."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2-D grid"
    bg = _background(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert 2 <= len(colors) <= 9, "expected several differently colored rings"

    objects: list[dict] = []
    by_color: dict[int, dict] = {}
    for color in colors:
        candidates = _ring_candidates(g, bg, color)
        assert len(candidates) == 1, "visible fragments must determine one unique ring template"
        obj = candidates[0]
        objects.append(obj)
        by_color[color] = obj

    # edge a -> b means a is above b.  A foreign color on b's complete
    # template is direct, visible evidence that a occluded b there.
    above: dict[int, set[int]] = {color: set() for color in colors}
    for obj in objects:
        color = int(obj["color"])
        for p in obj["border"]:
            if p == obj["gap"]:
                continue
            owner = int(g[p])
            assert owner != bg, "only the designated gap may be background on a ring template"
            if owner != color:
                assert owner in by_color, "an occluder must be another recovered ring"
                above[owner].add(color)

    # The stated inputs have pairwise partial overlaps.  Thus every pair must
    # provide one, and only one, visible precedence relation.
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            ab = b in above[a]
            ba = a in above[b]
            assert ab != ba, "every ring pair must reveal exactly one occlusion order"

    # Unique topological sort.  Its position is the number of rings above the
    # object, i.e. the required displacement distance.
    indegree = {color: 0 for color in colors}
    for a in colors:
        for b in above[a]:
            indegree[b] += 1
    remaining = set(colors)
    order: list[int] = []
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        assert len(ready) == 1, "occlusion evidence must determine one unique layer order"
        top = ready[0]
        order.append(top)
        remaining.remove(top)
        for lower in above[top]:
            indegree[lower] -= 1

    return bg, objects, order


def _render_solution(
    shape: tuple[int, int], bg: int, objects: list[dict], order: list[int]
) -> np.ndarray:
    h, w = map(int, shape)
    by_color = {int(obj["color"]): obj for obj in objects}
    out = np.full((h, w), int(bg), dtype=int)

    # Paint bottom-to-top so the recovered original occlusion order is kept.
    for depth in range(len(order) - 1, -1, -1):
        color = int(order[depth])
        obj = by_color[color]
        dr, dc = map(int, obj["direction"])
        sr, sc = depth * dr, depth * dc
        for r, c in obj["border"]:
            if (r, c) == obj["gap"]:
                continue
            rr, cc = int(r + sr), int(c + sc)
            assert 0 <= rr < h and 0 <= cc < w, "translated ring must remain in the canvas"
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each reconstructed gapped ring by its occlusion depth in the gap direction, then restack the complete rings.

    Input structure:
    - The dominant color is the background; every other color identifies one one-pixel-thick hollow rectangle.
    - The rectangles pairwise cross at a few border cells, upper borders hide lower ones, and the remaining fragments uniquely determine every full template.
    - Each odd-sided rectangle has one visible, side-centered single-cell gap on an unoccluded border segment.

    Rule:
    - Recover each full border from its visible color bbox and infer who is above whom whenever another color occupies that border template.
    - Require the pairwise evidence to give one unique top-to-bottom order; a ring at depth k has exactly k rings above it.
    - Move each ring k cells in the cardinal direction from its center through its gap, then redraw all completed borders bottom-to-top, retaining each moved gap.

    Key insight:
    - The gap supplies only direction while the occlusion topology supplies distance, so both independent clues are necessary for every non-top ring's translation.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects, order = _analyze(g)
    return _render_solution(g.shape, bg, objects, order)


# ------------------------------ generator ------------------------------

_COLOR_VALUES = tuple(range(10))
_OBJECT_COUNTS = (5, 6)
_GRID_RANGE = (27, 30)
_SIDE_NAMES = tuple(_DIRS)
_GEN_ATTEMPTS = 1200


def _strict_offsets(n: int) -> list[int]:
    """Random strict offsets whose total span stays below a 7x7 border."""
    picks = sorted(random.sample(range(6), n))
    return [x - picks[0] for x in picks]


def _varied_odd_sides(n: int) -> list[int]:
    split = random.randint(1, n - 1)
    return [7] * split + [9] * (n - split)


def _sample_boxes(h: int, w: int, n: int) -> list[tuple[int, int, int, int]] | None:
    roff, coff = _strict_offsets(n), _strict_offsets(n)
    heights, widths = _varied_odd_sides(n), _varied_odd_sides(n)
    margin = n
    max_r0 = h - margin - roff[-1] - heights[-1]
    max_c0 = w - margin - coff[-1] - widths[-1]
    if max_r0 < margin or max_c0 < margin:
        return None
    base_r = random.randint(margin, max_r0)
    base_c = random.randint(margin, max_c0)
    boxes = [
        (
            base_r + roff[i],
            base_c + coff[i],
            base_r + roff[i] + heights[i] - 1,
            base_c + coff[i] + widths[i] - 1,
        )
        for i in range(n)
    ]

    # Strictly increasing upper-left and lower-right corners make every pair
    # cross twice, never at a corner; the overlap remains visually sparse.
    borders = [_border(box) for box in boxes]
    for i in range(n):
        for j in range(i + 1, n):
            overlap = borders[i] & borders[j]
            if len(overlap) != 2:
                return None
            corners_i = {
                (boxes[i][0], boxes[i][1]), (boxes[i][0], boxes[i][3]),
                (boxes[i][2], boxes[i][1]), (boxes[i][2], boxes[i][3]),
            }
            corners_j = {
                (boxes[j][0], boxes[j][1]), (boxes[j][0], boxes[j][3]),
                (boxes[j][2], boxes[j][1]), (boxes[j][2], boxes[j][3]),
            }
            if overlap & (corners_i | corners_j):
                return None
    return boxes


def _direction_assignments(boxes: list[tuple[int, int, int, int]]) -> list[list[str]]:
    borders = [_border(box) for box in boxes]
    occupied_by_other = [
        set().union(*(borders[j] for j in range(len(boxes)) if j != i))
        for i in range(len(boxes))
    ]
    available = [
        [name for name, p in _side_midpoints(box).items() if p not in occupied_by_other[i]]
        for i, box in enumerate(boxes)
    ]
    if any(not choices for choices in available):
        return []

    solutions: list[list[str]] = []

    def search(i: int, chosen: list[str]) -> None:
        if i == len(boxes):
            if set(chosen) == set(_SIDE_NAMES):
                solutions.append(chosen.copy())
            return
        choices = available[i].copy()
        random.shuffle(choices)
        for name in choices:
            search(i + 1, chosen + [name])

    search(0, [])
    return solutions


def _paint_specs(
    shape: tuple[int, int],
    bg: int,
    specs: list[dict],
    order: list[int],
    distances: list[int],
    forced_direction: tuple[int, int] | None = None,
) -> np.ndarray:
    """Independent generator renderer indexed by object ids rather than colors."""
    out = np.full(shape, int(bg), dtype=int)
    for obj_id in reversed(order):
        spec = specs[obj_id]
        dr, dc = forced_direction if forced_direction is not None else spec["direction"]
        sr, sc = int(distances[obj_id] * dr), int(distances[obj_id] * dc)
        for r, c in _border(spec["box"]):
            if (r, c) == spec["gap"]:
                continue
            rr, cc = r + sr, c + sc
            if not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
                raise ValueError("generator translation escaped the canvas")
            out[rr, cc] = int(spec["color"])
    return out


def _semantic_checks(
    inp: np.ndarray, expected: np.ndarray, bg: int, specs: list[dict], order: list[int]
) -> bool:
    n = len(specs)
    depths = [0] * n
    for depth, obj_id in enumerate(order):
        depths[obj_id] = depth

    if np.array_equal(inp, expected):
        return False
    # Removing depth (all moving rings take one step) must change the answer.
    unit = [0 if d == 0 else 1 for d in depths]
    if np.array_equal(_paint_specs(inp.shape, bg, specs, order, unit), expected):
        return False
    # Removing gap directions (all rings move down) must change the answer.
    if np.array_equal(
        _paint_specs(inp.shape, bg, specs, order, depths, forced_direction=(1, 0)), expected
    ):
        return False
    # The original stacking order must still matter after translation.
    if np.array_equal(_paint_specs(inp.shape, bg, specs, list(reversed(order)), depths), expected):
        return False

    moved_dirs = [specs[obj_id]["name"] for obj_id in order[1:]]
    if len(set(moved_dirs)) == len(moved_dirs):
        return False
    if set(spec["name"] for spec in specs) != set(_SIDE_NAMES):
        return False

    try:
        solved = main(inp)
    except (AssertionError, ValueError):
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_OBJECT_COUNTS)
    h, w = random.randint(*_GRID_RANGE), random.randint(*_GRID_RANGE)
    boxes = _sample_boxes(h, w, n)
    if boxes is None:
        return None

    assignments = _direction_assignments(boxes)
    if not assignments:
        return None
    names = random.choice(assignments)

    # Put a singleton direction on top when necessary, leaving a repeated gap
    # direction among moved rings so equal direction visibly receives unequal
    # depth distances.
    counts = Counter(names)
    top_candidates = [i for i, name in enumerate(names) if counts[name] == 1]
    if not top_candidates:
        top_candidates = list(range(n))
    random.shuffle(top_candidates)
    top = next(
        (
            i for i in top_candidates
            if len({names[j] for j in range(n) if j != i}) < n - 1
        ),
        None,
    )
    if top is None:
        return None
    rest = [i for i in range(n) if i != top]
    random.shuffle(rest)
    order = [top] + rest
    if order == list(range(n)) or order == list(reversed(range(n))):
        return None

    bg = random.choice(_COLOR_VALUES)
    colors = random.sample([c for c in _COLOR_VALUES if c != bg], n)
    specs: list[dict] = []
    for i, (box, name, color) in enumerate(zip(boxes, names, colors)):
        specs.append(
            {
                "id": i,
                "box": box,
                "name": name,
                "direction": _DIRS[name],
                "gap": _side_midpoints(box)[name],
                "color": color,
            }
        )

    zeros = [0] * n
    inp = _paint_specs((h, w), bg, specs, order, zeros)
    if _background(inp) != bg:
        return None
    depths = [0] * n
    for depth, obj_id in enumerate(order):
        depths[obj_id] = depth
    expected = _paint_specs((h, w), bg, specs, order, depths)
    if not _semantic_checks(inp, expected, bg, specs, order):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a unique non-degenerate occlusion-depth ring puzzle")
