from __future__ import annotations

import random
from collections import Counter

import numpy as np


# Coordinates and inclusive bboxes follow arc_utils.py: (row, column) and
# (r0, c0, r1, c1).  The few helpers here are kept local so this file can be
# loaded directly with SourceFileLoader without modifying sys.path.
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (r, c)."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_direction(
    marker: tuple[int, int], box: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    """Outward direction when marker is exactly outside an edge midpoint."""
    r, c = marker
    r0, c0, r1, c1 = box
    candidates: list[tuple[int, int]] = []
    if (c1 - c0 + 1) % 2 == 1:
        mid_c = (c0 + c1) // 2
        if (r, c) == (r0 - 1, mid_c):
            candidates.append((-1, 0))
        if (r, c) == (r1 + 1, mid_c):
            candidates.append((1, 0))
    if (r1 - r0 + 1) % 2 == 1:
        mid_r = (r0 + r1) // 2
        if (r, c) == (mid_r, c0 - 1):
            candidates.append((0, -1))
        if (r, c) == (mid_r, c1 + 1):
            candidates.append((0, 1))
    return candidates[0] if len(candidates) == 1 else None


def _solid_rectangles(
    grid: np.ndarray, bg: int, excluded_color: int
) -> list[dict[str, object]] | None:
    rectangles: list[dict[str, object]] = []
    colors = set(map(int, np.unique(grid))) - {int(bg), int(excluded_color)}
    for color in colors:
        for coords in _components(grid == color):
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if len(coords) != area or not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                return None
            rectangles.append(
                {"color": int(color), "box": box, "coords": coords, "marker": None, "dir": None}
            )
    return rectangles if rectangles else None


def _parse(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]], list[int]]:
    """Identify the marker color, rectangles, and the unique successor chain."""
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    candidates: list[tuple[int, list[dict[str, object]]]] = []

    for marker_color in set(map(int, np.unique(g))) - {bg}:
        rectangles = _solid_rectangles(g, bg, marker_color)
        if rectangles is None:
            continue
        markers = [tuple(map(int, p)) for p in np.argwhere(g == marker_color)]
        if len(markers) != len(rectangles):
            continue

        valid = True
        used_rectangles: set[int] = set()
        for marker in markers:
            matches = [
                (i, _marker_direction(marker, rect["box"]))
                for i, rect in enumerate(rectangles)
                if _marker_direction(marker, rect["box"]) is not None
            ]
            if len(matches) != 1:
                valid = False
                break
            i, direction = matches[0]
            if i in used_rectangles:
                valid = False
                break
            used_rectangles.add(i)
            rectangles[i]["marker"] = marker
            rectangles[i]["dir"] = direction
        if valid and len(used_rectangles) == len(rectangles):
            candidates.append((int(marker_color), rectangles))

    assert len(candidates) == 1, "the common marker color and its rectangles must be unique"
    marker_color, rectangles = candidates[0]
    h, w = g.shape

    owner = np.full((h, w), -1, dtype=int)
    for i, rect in enumerate(rectangles):
        coords = rect["coords"]
        owner[coords[:, 0], coords[:, 1]] = i

    successors: list[int | None] = []
    for i, rect in enumerate(rectangles):
        r, c = rect["marker"]
        dr, dc = rect["dir"]
        r, c = r + dr, c + dc
        hit: int | None = None
        while 0 <= r < h and 0 <= c < w:
            if owner[r, c] >= 0:
                hit = int(owner[r, c])
                break
            r, c = r + dr, c + dc
        assert hit != i
        successors.append(hit)

    indegree = [0] * len(rectangles)
    for nxt in successors:
        if nxt is not None:
            indegree[nxt] += 1
    heads = [i for i, degree in enumerate(indegree) if degree == 0]
    tails = [i for i, nxt in enumerate(successors) if nxt is None]
    assert len(heads) == 1 and len(tails) == 1
    assert all(degree == 1 for i, degree in enumerate(indegree) if i != heads[0])

    order: list[int] = []
    current: int | None = heads[0]
    while current is not None and current not in order:
        order.append(current)
        current = successors[current]
    assert current is None and len(order) == len(rectangles), "successors must form one full chain"
    assert order[-1] == tails[0]
    return bg, marker_color, rectangles, order


def _turn_marker_side_down(array: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    turns = {(1, 0): 0, (0, 1): 3, (-1, 0): 2, (0, -1): 1}
    return np.rot90(array, k=turns[direction]).copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Order rectangles by their marker rays, orient them marker-side down, and stack them vertically.

    Input structure:
    - The dominant color is background; the other objects are separated solid monochrome rectangles.
    - A shared marker color supplies one pixel just outside the midpoint of exactly one edge of every rectangle.
    - Each outward marker ray first hits the successor rectangle, except the tail ray, which reaches the boundary.

    Rule:
    - Find the unique rectangle with no incoming ray and traverse first-hit successors to obtain the full order.
    - Rotate each tight rectangle so its marked edge faces down, omit markers, then stack the rectangles top-to-bottom.
    - Center every oriented rectangle horizontally in a background canvas whose width is the maximum oriented width.

    Key insight:
    - Each marker has two inseparable roles: its outward ray selects the next rectangle and its attached edge selects rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, _marker_color, rectangles, order = _parse(g)

    pieces: list[np.ndarray] = []
    for i in order:
        rect = rectangles[i]
        r0, c0, r1, c1 = rect["box"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        pieces.append(_turn_marker_side_down(patch, rect["dir"]))

    out_h = sum(piece.shape[0] for piece in pieces)
    out_w = max(piece.shape[1] for piece in pieces)
    out = np.full((out_h, out_w), bg, dtype=int)
    row = 0
    for piece in pieces:
        ph, pw = piece.shape
        col = (out_w - pw) // 2
        out[row : row + ph, col : col + pw] = piece
        row += ph
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _outward_marker(
    box: tuple[int, int, int, int], direction: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if direction == (-1, 0):
        return r0 - 1, (c0 + c1) // 2
    if direction == (1, 0):
        return r1 + 1, (c0 + c1) // 2
    if direction == (0, -1):
        return (r0 + r1) // 2, c0 - 1
    return (r0 + r1) // 2, c1 + 1


def _known_rectangles(
    grid: np.ndarray, colors: list[int], marker_color: int
) -> list[dict[str, object]]:
    """Recover known generated rectangles without using the solver's chain parser."""
    marker_points = [tuple(map(int, p)) for p in np.argwhere(grid == marker_color)]
    result: list[dict[str, object]] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        box = _bbox(coords)
        matches = [(m, _marker_direction(m, box)) for m in marker_points]
        matches = [(m, d) for m, d in matches if d is not None]
        if len(matches) != 1:
            raise ValueError("generated rectangle does not have one exact marker")
        marker, direction = matches[0]
        result.append({"color": color, "box": box, "marker": marker, "dir": direction})
    return result


def _stack_oracle(
    bg: int, rectangles: list[dict[str, object]], *, orient: bool = True
) -> np.ndarray:
    dims: list[tuple[int, int, int]] = []
    for rect in rectangles:
        r0, c0, r1, c1 = rect["box"]
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if orient and rect["dir"][1] != 0:
            h, w = w, h
        dims.append((h, w, int(rect["color"])))
    out_h = sum(h for h, _w, _color in dims)
    out_w = max(w for _h, w, _color in dims)
    out = np.full((out_h, out_w), bg, dtype=int)
    row = 0
    for h, w, color in dims:
        col = (out_w - w) // 2
        out[row : row + h, col : col + w] = color
        row += h
    return out


def _ray_rectangle_ids(
    shape: tuple[int, int], rectangles: list[dict[str, object]], index: int
) -> list[int]:
    h, w = shape
    owner = np.full(shape, -1, dtype=int)
    for i, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["box"]
        owner[r0 : r1 + 1, c0 : c1 + 1] = i
    r, c = rectangles[index]["marker"]
    dr, dc = rectangles[index]["dir"]
    r, c = r + dr, c + dc
    hits: list[int] = []
    while 0 <= r < h and 0 <= c < w:
        hit = int(owner[r, c])
        if hit >= 0 and (not hits or hits[-1] != hit):
            hits.append(hit)
        r, c = r + dr, c + dc
    return hits


def _semantic_check(
    grid: np.ndarray, bg: int, marker_color: int, colors: list[int], expected: np.ndarray
) -> bool:
    """Check rule semantics, anti-shortcuts, uniqueness, and non-degeneracy."""
    try:
        rectangles = _known_rectangles(grid, colors, marker_color)
    except ValueError:
        return False
    n = len(rectangles)

    # The designed chain is colors[0] -> ... -> colors[-1].  Verify first-hit
    # semantics independently, including a farther decoy behind the first successor.
    ray_hits = [_ray_rectangle_ids(grid.shape, rectangles, i) for i in range(n)]
    if any(not ray_hits[i] or ray_hits[i][0] != i + 1 for i in range(n - 1)):
        return False
    if ray_hits[-1]:
        return False
    if len(ray_hits[0]) < 2:
        return False

    directions = {rect["dir"] for rect in rectangles}
    if len(directions) < 3:
        return False
    if not any(rect["dir"][1] != 0 and (rect["box"][2] - rect["box"][0]) != (rect["box"][3] - rect["box"][1]) for rect in rectangles):
        return False

    # Chain order must not be recoverable by ordinary row/column position sorts.
    chain = list(range(n))
    row_order = sorted(chain, key=lambda i: (rectangles[i]["box"][0], rectangles[i]["box"][1]))
    col_order = sorted(chain, key=lambda i: (rectangles[i]["box"][1], rectangles[i]["box"][0]))
    if chain == row_order or chain == col_order:
        return False

    if not np.array_equal(_stack_oracle(bg, rectangles), expected):
        return False
    if np.array_equal(_stack_oracle(bg, rectangles, orient=False), expected):
        return False
    reversed_output = _stack_oracle(bg, list(reversed(rectangles)))
    if np.array_equal(reversed_output, expected):
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if _background(grid) != bg:
        return False

    try:
        parsed_bg, parsed_marker, parsed_rects, parsed_order = _parse(grid)
        actual = main(grid)
    except (AssertionError, ValueError, IndexError):
        return False
    if parsed_bg != bg or parsed_marker != marker_color or len(parsed_rects) != n or len(parsed_order) != n:
        return False
    return np.array_equal(actual, expected)


def _generation_attempt() -> np.ndarray | None:
    n = random.choice((5, 6))
    palette = random.sample(range(10), n + 2)
    bg, marker_color = palette[:2]
    colors = palette[2:]

    top = random.randint(3, 5)
    vertical_gap = random.randint(7, 10)
    bottom = top + vertical_gap
    left = random.randint(3, 4)
    gap_count = 3 if n == 6 else 2
    gaps = [7 if n == 6 else random.randint(7, 9) for _ in range(gap_count)]
    columns = [left]
    for gap in gaps:
        columns.append(columns[-1] + gap)

    h = bottom + random.randint(4, 6)
    w = columns[-1] + random.randint(4, 6)
    if h > 30 or w > 30:
        return None

    # A -> B -> C -> E -> D [-> F].  A's ray sees B first and D/F farther away.
    centers = [
        (top, columns[0]),
        (top, columns[1]),
        (bottom, columns[1]),
        (bottom, columns[2]),
        (top, columns[2]),
    ]
    if n == 6:
        centers.append((top, columns[3]))

    directions: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(centers, centers[1:]):
        directions.append((int(np.sign(r1 - r0)), int(np.sign(c1 - c0))))
    directions.append((0, 1))  # tail exits through the right boundary

    shape_choices = ((3, 3), (3, 5), (5, 3), (5, 5))
    shapes = [random.choice(shape_choices) for _ in range(n)]
    if not any(d[1] != 0 and sh[0] != sh[1] for sh, d in zip(shapes, directions)):
        shapes[0] = random.choice(((3, 5), (5, 3)))
    if not any(d[0] != 0 and sh[0] != sh[1] for sh, d in zip(shapes, directions)):
        shapes[1] = random.choice(((3, 5), (5, 3)))

    base = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    marker_points: list[tuple[int, int]] = []
    for color, center, shape, direction in zip(colors, centers, shapes, directions):
        cr, cc = center
        rh, rw = shape[0] // 2, shape[1] // 2
        box = (cr - rh, cc - rw, cr + rh, cc + rw)
        r0, c0, r1, c1 = box
        marker = _outward_marker(box, direction)
        mr, mc = marker
        if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w and 0 <= mr < h and 0 <= mc < w):
            return None
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any() or occupied[mr, mc]:
            return None
        base[r0 : r1 + 1, c0 : c1 + 1] = color
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        marker_points.append(marker)

    if len(set(marker_points)) != n:
        return None
    for r, c in marker_points:
        if occupied[r, c]:
            return None
        base[r, c] = marker_color
        occupied[r, c] = True

    # Uniform global dihedral transform balances all four visible marker sides
    # without changing the successor chain.
    transformed = np.rot90(base, k=random.randrange(4)).copy()
    if random.choice((False, True)):
        transformed = np.fliplr(transformed).copy()

    try:
        known = _known_rectangles(transformed, colors, marker_color)
        expected = _stack_oracle(bg, known)
    except (ValueError, IndexError):
        return None
    if not _semantic_check(transformed, bg, marker_color, colors, expected):
        return None
    return transformed


def generate_input() -> np.ndarray:
    """Generate a varied, independently validated instance of the ray-successor chain task."""
    for _ in range(300):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate ray-successor chain")
