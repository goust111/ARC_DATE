from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import get_bg, in_bounds, is_contiguous, neighbors4
import math
import random


def _signal_direction(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    edges = [
        ("top", g[0, :], (-1, 0)),
        ("bottom", g[-1, :], (1, 0)),
        ("left", g[:, 0], (0, -1)),
        ("right", g[:, -1], (0, 1)),
    ]
    signal_edges = []
    for _name, values, direction in edges:
        color = int(values[0])
        if color != int(bg) and bool(np.all(values == color)):
            signal_edges.append((color, direction))

    assert len(signal_edges) == 1, "expected exactly one non-bg solid edge"
    edge_color, (dr, dc) = signal_edges[0]
    return int(edge_color), int(dr), int(dc)


def _role_colors(g: np.ndarray, bg: int, edge_color: int) -> tuple[int, int]:
    counts = Counter(map(int, g.flat))
    role_colors = [c for c in counts if c not in {int(bg), int(edge_color)}]
    assert len(role_colors) == 2, "expected one main object color and one marker color"

    a, b = role_colors
    assert counts[a] != counts[b], "role colors must have distinct frequencies"
    if counts[a] > counts[b]:
        return int(a), int(b)
    return int(b), int(a)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair markers embedded in the main object and copy the remaining markers one pixel toward the solid edge.

    Input structure:
    - A mostly background grid contains one solid non-background edge line.
    - Apart from that edge and the background, there is one large main object
      color and a smaller marker color.
    - Each marker is either embedded in the main object, touching it on at
      least three 4-neighbor sides, or isolated on the background.

    Rule:
    - Embedded markers are recolored to the main object color.
    - Isolated background markers remain in place and also paint the adjacent
      background pixel in the direction of the solid edge line.
    - All other pixels are unchanged.

    Key insight:
    The full edge line is a direction cue: it tells which way the background
    markers extend, while markers surrounded by the main object are simply
    defects to be filled.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    edge_color, dr, dc = _signal_direction(g, bg)
    main_color, marker_color = _role_colors(g, bg, edge_color)

    out = g.copy()
    marker_cells = np.argwhere(g == marker_color)
    assert marker_cells.size > 0, "expected at least one marker"

    for r0, c0 in marker_cells:
        r = int(r0)
        c = int(c0)
        main_neighbors = sum(
            1
            for rr, cc in neighbors4(r, c)
            if in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == int(main_color)
        )

        if main_neighbors >= 3:
            out[r, c] = main_color
        else:
            assert main_neighbors == 0, "marker must be embedded or isolated from the main object"
            nr = r + dr
            nc = c + dc
            assert in_bounds(g.shape, nr, nc), "marker extension must stay on the grid"
            assert int(g[nr, nc]) == int(bg), "marker extension target must be background"
            out[nr, nc] = marker_color

    return out

# --------------- generator (att2) ---------------

EDGES = ["top", "bottom", "left", "right"]
EDGE_DIRECTIONS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Canvas, palette, and edge cue bounds.
GRID_SIZE_MIN, GRID_SIZE_MAX = 8, 13
COLOR_POOL = range(10)

# Main object size and growth shape controls.
MAIN_EDGE_GAP = 2
MAIN_MIN_PIXELS, MAIN_MAX_PIXELS = 10, 51
MAIN_MIN_NON_EDGE_RATIO, MAIN_MAX_NON_EDGE_RATIO = 0.15, 0.30
BLOB_DEGREE_BIAS, BLOB_DEGREE_POWER = 0.35, 3

# Marker count bounds; the margin preserves main_color > marker_color.
EMBEDDED_MIN, EMBEDDED_MAX = 1, 7
ISOLATED_MIN, ISOLATED_MAX = 3, 15
EMBEDDED_MAIN_MARGIN, MARKER_MAIN_COUNT_MARGIN = 4, 1

# Retry budget.
GENERATION_ATTEMPTS = 300

Cell = tuple[int, int]
MarkerPair = tuple[Cell, Cell]


def _edge_direction(edge: str) -> tuple[int, int]:
    return EDGE_DIRECTIONS[edge]


def _paint_edge(g: np.ndarray, edge: str, color: int) -> None:
    if edge == "top":
        g[0, :] = color
    elif edge == "bottom":
        g[-1, :] = color
    elif edge == "left":
        g[:, 0] = color
    else:
        g[:, -1] = color


def _main_allowed(n: int, edge: str) -> list[Cell]:
    r0 = MAIN_EDGE_GAP if edge == "top" else 0
    r1 = n - MAIN_EDGE_GAP if edge == "bottom" else n
    c0 = MAIN_EDGE_GAP if edge == "left" else 0
    c1 = n - MAIN_EDGE_GAP if edge == "right" else n
    return [(r, c) for r in range(r0, r1) for c in range(c0, c1)]


def _degree_in(cells: set[Cell], r: int, c: int) -> int:
    return sum((rr, cc) in cells for rr, cc in neighbors4(r, c))


def _grow_blob(n: int, edge: str, size: int) -> set[Cell] | None:
    allowed = set(_main_allowed(n, edge))
    if len(allowed) < size:
        return None

    start = random.choice(tuple(allowed))
    blob = {start}
    frontier = {
        (rr, cc)
        for rr, cc in neighbors4(*start)
        if (rr, cc) in allowed
    }

    while len(blob) < size and frontier:
        choices = tuple(frontier)
        weights = []
        for r, c in choices:
            k = _degree_in(blob, r, c)
            weights.append((k + BLOB_DEGREE_BIAS) ** BLOB_DEGREE_POWER)
        cell = random.choices(choices, weights=weights, k=1)[0]
        frontier.remove(cell)
        blob.add(cell)
        for rr, cc in neighbors4(*cell):
            if (rr, cc) in allowed and (rr, cc) not in blob:
                frontier.add((rr, cc))

    if len(blob) != size:
        return None
    return blob


def _connected_after_removal(n: int, blob: set[Cell], removed: set[Cell]) -> bool:
    mask = np.zeros((n, n), dtype=int)
    for r, c in blob - removed:
        mask[r, c] = 1
    return bool(mask.any()) and is_contiguous(mask, background=0, connectivity=4)


def _embedded_order(n: int, blob: set[Cell]) -> list[Cell]:
    candidates = [cell for cell in blob if _degree_in(blob, cell[0], cell[1]) >= 3]
    random.shuffle(candidates)
    picked: list[Cell] = []
    removed: set[Cell] = set()

    for cell in candidates:
        if any((rr, cc) in removed for rr, cc in neighbors4(*cell)):
            continue
        trial = removed | {cell}
        if _connected_after_removal(n, blob, trial):
            picked.append(cell)
            removed.add(cell)
    return picked


def _isolated_pairs(g: np.ndarray, blob: set[Cell], bg: int, dr: int, dc: int) -> list[MarkerPair]:
    n = g.shape[0]
    pairs = []
    for r in range(n):
        for c in range(n):
            if int(g[r, c]) != bg:
                continue
            if any((rr, cc) in blob for rr, cc in neighbors4(r, c)):
                continue
            tr, tc = r + dr, c + dc
            if in_bounds(g.shape, tr, tc) and int(g[tr, tc]) == bg:
                pairs.append(((r, c), (tr, tc)))

    random.shuffle(pairs)
    selected = []
    used: set[Cell] = set()
    for marker, target in pairs:
        if marker in used or target in used:
            continue
        selected.append((marker, target))
        used.add(marker)
        used.add(target)
    return selected


def _blob_size_bounds(n: int) -> tuple[int, int] | None:
    non_edge_area = n * n - n
    lo = max(MAIN_MIN_PIXELS, math.ceil(MAIN_MIN_NON_EDGE_RATIO * non_edge_area))
    hi = min(MAIN_MAX_PIXELS, math.floor(MAIN_MAX_NON_EDGE_RATIO * non_edge_area))
    if lo > hi:
        return None
    return lo, hi


def _choose_embedded_sites(n: int, blob: set[Cell]) -> list[Cell] | None:
    embedded_capacity = min(
        EMBEDDED_MAX,
        (len(blob) - EMBEDDED_MAIN_MARGIN) // 2,
    )
    embedded_sites = _embedded_order(n, blob)
    embedded_high = min(embedded_capacity, len(embedded_sites))
    if embedded_high < EMBEDDED_MIN:
        return None
    embedded_count = random.randint(EMBEDDED_MIN, embedded_high)
    return embedded_sites[:embedded_count]


def _choose_isolated_pairs(
    out: np.ndarray,
    blob: set[Cell],
    bg: int,
    dr: int,
    dc: int,
    embedded_count: int,
) -> list[MarkerPair] | None:
    isolated_high_by_count = min(
        ISOLATED_MAX,
        len(blob) - 2 * embedded_count - MARKER_MAIN_COUNT_MARGIN,
    )
    if isolated_high_by_count < ISOLATED_MIN:
        return None
    pairs = _isolated_pairs(out, blob, bg, dr, dc)
    isolated_high = min(isolated_high_by_count, len(pairs))
    if isolated_high < ISOLATED_MIN:
        return None
    isolated_count = random.randint(ISOLATED_MIN, isolated_high)
    return pairs[:isolated_count]


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    edge = random.choice(EDGES)
    colors = tuple(random.sample(COLOR_POOL, 4))
    bg, edge_color, main_color, marker_color = colors
    dr, dc = _edge_direction(edge)

    size_bounds = _blob_size_bounds(n)
    if size_bounds is None:
        return None

    blob = _grow_blob(n, edge, random.randint(*size_bounds))
    if blob is None:
        return None

    embedded_sites = _choose_embedded_sites(n, blob)
    if embedded_sites is None:
        return None

    out = np.full((n, n), bg, dtype=int)
    _paint_edge(out, edge, edge_color)
    for r, c in blob:
        out[r, c] = main_color

    pairs = _choose_isolated_pairs(out, blob, bg, dr, dc, len(embedded_sites))
    if pairs is None:
        return None

    for (r, c), (tr, tc) in pairs:
        out[r, c] = marker_color
        out[tr, tc] = marker_color

    inp = out.copy()
    for r, c in embedded_sites:
        inp[r, c] = marker_color
    for _marker, (tr, tc) in pairs:
        inp[tr, tc] = bg

    if (inp == bg).sum() <= max(
        (inp == edge_color).sum(),
        (inp == main_color).sum(),
        (inp == marker_color).sum(),
    ):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 8886d717 input")
