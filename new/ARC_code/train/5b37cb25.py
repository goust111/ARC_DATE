import numpy as np

from arc_utils import find_components, in_bounds, paste, random_free_location
import random
from itertools import product


_CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_DIRECTIONS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _uniform_side(side: np.ndarray) -> int:
    vals = np.unique(side)
    assert len(vals) == 1, "expected each frame side to be one solid color"
    return int(vals[0])


def _interior_colors(interior: np.ndarray) -> tuple[int, int]:
    vals, counts = np.unique(interior, return_counts=True)
    max_count = counts.max()
    assert int(np.sum(counts == max_count)) == 1, "expected a unique dominant interior background"
    bg = int(vals[int(np.argmax(counts))])
    obj_colors = [int(v) for v in vals if int(v) != bg]
    assert len(obj_colors) == 1, "expected one interior foreground object color"
    return bg, obj_colors[0]


def _direction_for_notch(g: np.ndarray, r: int, c: int, obj: int) -> list[str]:
    dirs = []
    shape = g.shape
    for name, (dr, dc) in _DIRECTIONS.items():
        pr, pc = -dc, dr
        support = (
            (r - 2 * dr, c - 2 * dc),
            (r - dr + pr, c - dc + pc),
            (r - dr - pr, c - dc - pc),
            (r + 2 * pr, c + 2 * pc),
            (r - 2 * pr, c - 2 * pc),
        )
        open_side = (
            (r + 2 * dr, c + 2 * dc),
            (r + dr + pr, c + dc + pc),
            (r + dr - pr, c + dc - pc),
        )
        if not all(in_bounds(shape, rr, cc) and int(g[rr, cc]) == obj for rr, cc in support):
            continue
        if not all((not in_bounds(shape, rr, cc)) or int(g[rr, cc]) != obj for rr, cc in open_side):
            continue
        dirs.append(name)
    return dirs


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each plus-shaped background notch with the color of the frame side it opens toward.

    Input structure
    - A one-cell-thick rectangular frame has four solid-colored sides and neutral corner pixels.
    - The frame interior contains a dominant background color and one foreground object color.
    - Some object boundaries contain five-pixel plus-shaped background notches.

    Rule
    - Find every background plus whose opposite side is supported by foreground object pixels and whose open side faces one frame side.
    - Recolor the five pixels of that plus with the color of the facing frame side.
    - Leave all other pixels unchanged.

    Key insight
    The frame side colors are direction labels for the open notches in the foreground object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 5, "expected room for a frame and plus notches"

    side_color = {
        "top": _uniform_side(g[0, 1:-1]),
        "bottom": _uniform_side(g[-1, 1:-1]),
        "left": _uniform_side(g[1:-1, 0]),
        "right": _uniform_side(g[1:-1, -1]),
    }
    bg, obj = _interior_colors(g[1:-1, 1:-1])

    out = g.copy()
    painted: set[tuple[int, int]] = set()
    notch_count = 0
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            cross = tuple((r + dr, c + dc) for dr, dc in _CROSS)
            if not all(int(g[rr, cc]) == bg for rr, cc in cross):
                continue

            dirs = _direction_for_notch(g, r, c, obj)
            assert len(dirs) <= 1, "expected each plus notch to open in at most one direction"
            if not dirs:
                continue

            notch_count += 1
            color = side_color[dirs[0]]
            for cell in cross:
                assert cell not in painted, "expected non-overlapping plus notches"
                painted.add(cell)
                out[cell] = color

    assert notch_count > 0, "expected at least one plus notch"
    return out

# --------------- generator (att2) ---------------

# Canvas and palette.
_GEN_GRID_SIZE, _GEN_CORNER_COLOR = 30, 0
_GEN_COLOR_POOL, _GEN_PALETTE_SIZE = range(1, 10), 6

# Object size and total filled-area budget.
_OBJECT_COUNT_MIN, _OBJECT_COUNT_MAX = 3, 4
_OBJECT_DIM_MIN, _OBJECT_DIM_MAX = 5, 20
_OBJECT_AREA_MIN, _OBJECT_AREA_MAX = 30, 180
_TOTAL_OBJECT_AREA_MAX, _MIN_REMAINING_OBJECT_AREA = 360, 30

# Placement gap against frame and other objects.
_PLACEMENT_BORDER, _PLACEMENT_PADDING, _PLACEMENT_PADDING_CONNECTIVITY = 2, 2, 8

# Notch geometry and counts.
_GEN_CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_DIRECTIONS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_EDGES = tuple(_GEN_DIRECTIONS)
_NOTCH_EDGE_MARGIN, _MAX_TOTAL_NOTCHES = 2, 8
_NOTCHES_PER_OBJECT = (1, 2, 3)

# Rejection-sampling retry cap.
_GEN_RETRY_LIMIT = 300
_DIM_PAIRS = [(h, w) for h in range(_OBJECT_DIM_MIN, _OBJECT_DIM_MAX + 1)
              for w in range(_OBJECT_DIM_MIN, _OBJECT_DIM_MAX + 1) if _OBJECT_AREA_MIN <= h * w <= _OBJECT_AREA_MAX]


def _notch_cells(r, c, direction):
    dr, dc = _GEN_DIRECTIONS[direction]
    pr, pc = -dc, dr
    cross = tuple((r + rr, c + cc) for rr, cc in _GEN_CROSS)
    support = (
        (r - 2 * dr, c - 2 * dc),
        (r - dr + pr, c - dc + pc),
        (r - dr - pr, c - dc - pc),
        (r + 2 * pr, c + 2 * pc),
        (r - 2 * pr, c - 2 * pc),
    )
    open_side = (
        (r + 2 * dr, c + 2 * dc),
        (r + dr + pr, c + dc + pc),
        (r + dr - pr, c + dc - pc),
    )
    return {"direction": direction, "cross": cross, "support": support, "open_side": open_side}


def _candidate_notches(r0, c0, h, w, edge):
    if edge in ("top", "bottom"):
        row = r0 if edge == "top" else r0 + h - 1
        centers = [(row, c) for c in range(c0 + _NOTCH_EDGE_MARGIN, c0 + w - _NOTCH_EDGE_MARGIN)]
    else:
        col = c0 if edge == "left" else c0 + w - 1
        centers = [(r, col) for r in range(r0 + _NOTCH_EDGE_MARGIN, r0 + h - _NOTCH_EDGE_MARGIN)]
    return [_notch_cells(r, c, edge) for r, c in centers]


def _choose_notches(r0, c0, h, w, count):
    edges = random.sample(_EDGES, count)
    random.shuffle(edges)
    choices = {edge: _candidate_notches(r0, c0, h, w, edge) for edge in edges}

    def search(i, chosen, carved, protected):
        if i == len(edges):
            return chosen
        options = choices[edges[i]][:]
        random.shuffle(options)
        for notch in options:
            cross = set(notch["cross"])
            support = set(notch["support"])
            if cross & carved or cross & protected or support & carved:
                continue
            result = search(i + 1, chosen + [notch], carved | cross, protected | support)
            if result is not None:
                return result
        return None

    return search(0, [], set(), set())


def _notch_counts(n_objects):
    total = random.randint(n_objects, min(_MAX_TOTAL_NOTCHES, max(_NOTCHES_PER_OBJECT) * n_objects))
    plans = [p for p in product(_NOTCHES_PER_OBJECT, repeat=n_objects) if sum(p) == total]
    plan = list(random.choice(plans))
    random.shuffle(plan)
    return plan


def _place_rectangles(out, obj, n_objects, counts):
    """Place object rectangles and return (r,c,h,w,notch_count) records, or None."""
    placement = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=int)
    rects = []
    total_area = 0
    for i in range(n_objects):
        remaining = n_objects - i - 1
        area_cap = _TOTAL_OBJECT_AREA_MAX - _MIN_REMAINING_OBJECT_AREA * remaining
        valid = [p for p in _DIM_PAIRS if total_area + p[0] * p[1] <= area_cap]
        if not valid:
            return None
        h, w = random.choice(valid)
        patch = np.ones((h, w), dtype=int)
        try:
            r, c = random_free_location(
                placement, patch, transparent=0, border=_PLACEMENT_BORDER, padding=_PLACEMENT_PADDING,
                padding_connectivity=_PLACEMENT_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(placement, patch, r, c, transparent=0)
        paste(out, np.full((h, w), obj, dtype=int), r, c)
        rects.append((r, c, h, w, counts[i]))
        total_area += h * w
    return rects


def _apply_notches(out, notches, side_color, bg):
    """Paint output notches and return the matching carved input grid."""
    for notch in notches:
        color = side_color[notch["direction"]]
        for rr, cc in notch["cross"]:
            out[rr, cc] = color

    inp = out.copy()
    for notch in notches:
        for rr, cc in notch["cross"]:
            inp[rr, cc] = bg
    return inp


def _passes_oracle(inp, out, n_objects, bg, obj):
    """Require the intended components and exact agreement with main()."""
    comps = find_components(inp[1:-1, 1:-1], bg=bg, connectivity=4, monochromatic=True)
    if len(comps) != n_objects or any(comp["color"] != obj for comp in comps):
        return False
    try:
        return np.array_equal(main(inp), out)
    except Exception:
        return False


def _gen_attempt():
    top, bottom, left, right, bg, obj = random.sample(_GEN_COLOR_POOL, _GEN_PALETTE_SIZE)
    side_color = {"top": top, "bottom": bottom, "left": left, "right": right}

    out = np.full((_GEN_GRID_SIZE, _GEN_GRID_SIZE), bg, dtype=int)
    out[0, 1:-1] = top
    out[-1, 1:-1] = bottom
    out[1:-1, 0] = left
    out[1:-1, -1] = right
    out[0, 0] = out[0, -1] = out[-1, 0] = out[-1, -1] = _GEN_CORNER_COLOR

    n_objects = random.randint(_OBJECT_COUNT_MIN, _OBJECT_COUNT_MAX)
    counts = _notch_counts(n_objects)
    rects = _place_rectangles(out, obj, n_objects, counts)
    if rects is None:
        return None

    notches = []
    for r, c, h, w, count in rects:
        chosen = _choose_notches(r, c, h, w, count)
        if chosen is None:
            return None
        notches.extend(chosen)

    inp = _apply_notches(out, notches, side_color, bg)
    if not _passes_oracle(inp, out, n_objects, bg, obj):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5b37cb25 input")
