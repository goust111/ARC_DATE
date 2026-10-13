from __future__ import annotations

import math
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, label_mask


POLY_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW)
FILL_A = Color.GREY
FILL_B = Color.PINK
PALETTE_COLORS = {Color.GREY, Color.PINK}

GRID_MIN = 18
GRID_MAX = 24
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120
DISTRACTOR_ATTEMPTS = 300


def _angle_cycle(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(points, dtype=float)
    center = arr.mean(axis=0)
    return sorted(
        [tuple(map(int, p)) for p in points],
        key=lambda p: (
            math.atan2(p[0] - center[0], p[1] - center[1]),
            (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2,
            p[0],
            p[1],
        ),
    )


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc == 0:
        return [a]
    if dr == 0:
        sr, sc = 0, 1 if dc > 0 else -1
        steps = abs(dc)
    elif dc == 0:
        sr, sc = 1 if dr > 0 else -1, 0
        steps = abs(dr)
    elif abs(dr) == abs(dc):
        sr = 1 if dr > 0 else -1
        sc = 1 if dc > 0 else -1
        steps = abs(dr)
    else:
        return None
    return [(r0 + sr * k, c0 + sc * k) for k in range(steps + 1)]


def _polygon_area2(points: list[tuple[int, int]]) -> int:
    total = 0
    for i, (r0, c0) in enumerate(points):
        r1, c1 = points[(i + 1) % len(points)]
        total += c0 * r1 - c1 * r0
    return int(total)


def _cycle_if_valid(
    points: list[tuple[int, int]],
    blocked_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if len(points) < 3:
        return None
    pts = sorted(set(points))
    if len(pts) != len(points):
        return None
    cycle = _angle_cycle(pts)
    if _polygon_area2(cycle) == 0:
        return None

    occupied_counts: dict[tuple[int, int], int] = defaultdict(int)
    vertex_set = set(cycle)
    prev_end = None

    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i + 1) % len(cycle)]
        cells = _line_cells(a, b)
        if cells is None:
            return None
        for cell in cells[1:-1]:
            if cell in blocked_cells:
                return None
        for j, cell in enumerate(cells):
            occupied_counts[cell] += 1
            if occupied_counts[cell] > 1:
                allow_shared = False
                if cell in vertex_set:
                    if j == 0 and cell == prev_end:
                        allow_shared = True
                    if i == len(cycle) - 1 and j == len(cells) - 1 and cell == cycle[0]:
                        allow_shared = True
                if not allow_shared:
                    return None
        prev_end = b
    return cycle


def _reduce_group(
    points: list[tuple[int, int]],
    non_bg: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    original = sorted(set(points))
    current = original[:]
    removed: list[tuple[int, int]] = []
    while len(current) >= 3:
        ignored = set(original) - set(current)
        cycle = _cycle_if_valid(current, non_bg - ignored)
        if cycle is not None:
            return cycle, removed
        valid_removals = []
        for p in current:
            subset = [q for q in current if q != p]
            ignored = set(original) - set(subset)
            if _cycle_if_valid(subset, non_bg - ignored) is not None:
                valid_removals.append(p)
        if not valid_removals:
            return None
        chosen = min(valid_removals)
        current.remove(chosen)
        removed.append(chosen)
    return None


def _outline_cells(cycle: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for i in range(len(cycle)):
        seg = _line_cells(cycle[i], cycle[(i + 1) % len(cycle)])
        assert seg is not None
        cells.update(seg)
    return cells


def _render_outline(
    shape: tuple[int, int],
    cycles_by_color: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color, cycle in cycles_by_color.items():
        for r, c in _outline_cells(cycle):
            if out[r, c] not in (0, color):
                raise ValueError("outline conflict between different polygon colors")
            out[r, c] = int(color)
    return out


def _region_labels_and_parity(outline_mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    h, w = outline_mask.shape
    scale = 3
    big = np.zeros((h * scale, w * scale), dtype=bool)
    for r, c in np.argwhere(outline_mask):
        rr = int(r) * scale
        cc = int(c) * scale
        big[rr : rr + scale, cc : cc + scale] = True

    labs, _ = label_mask(~big, connectivity=4)
    center_labels = np.zeros((h, w), dtype=int)
    used_labels: set[int] = set()
    for r in range(h):
        for c in range(w):
            if outline_mask[r, c]:
                continue
            lbl = int(labs[r * scale + 1, c * scale + 1])
            center_labels[r, c] = lbl
            used_labels.add(lbl)

    start = int(center_labels[0, 0])
    if start <= 0:
        raise ValueError("(0,0) must lie in the outside region")

    graph: dict[int, set[int]] = {lbl: set() for lbl in used_labels}
    bh, bw = big.shape
    for r in range(bh):
        for c in range(bw):
            if not big[r, c]:
                continue
            around = set()
            if r > 0:
                a = int(labs[r - 1, c])
                if a in used_labels:
                    around.add(a)
            if r + 1 < bh:
                a = int(labs[r + 1, c])
                if a in used_labels:
                    around.add(a)
            if c > 0:
                a = int(labs[r, c - 1])
                if a in used_labels:
                    around.add(a)
            if c + 1 < bw:
                a = int(labs[r, c + 1])
                if a in used_labels:
                    around.add(a)
            around = sorted(around)
            for i, a in enumerate(around):
                for b in around[i + 1 :]:
                    graph[a].add(b)
                    graph[b].add(a)

    parity = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in graph[cur]:
            want = 1 - parity[cur]
            if nxt in parity:
                if parity[nxt] != want:
                    raise ValueError("region adjacency graph is not bipartite")
            else:
                parity[nxt] = want
                queue.append(nxt)
    if set(parity) != used_labels:
        raise ValueError("some regions are not connected through wall adjacencies")
    return center_labels, parity


def _render_solution(
    shape: tuple[int, int],
    cycles_by_color: dict[int, list[tuple[int, int]]],
    palette_cells: dict[tuple[int, int], int],
) -> np.ndarray:
    outline = _render_outline(shape, cycles_by_color)
    labels, parity = _region_labels_and_parity(outline > 0)
    out = np.zeros(shape, dtype=int)
    for r in range(shape[0]):
        for c in range(shape[1]):
            if outline[r, c]:
                out[r, c] = int(outline[r, c])
            elif (r, c) in palette_cells:
                out[r, c] = int(palette_cells[(r, c)])
            else:
                region = int(labels[r, c])
                out[r, c] = FILL_A if parity[region] == 0 else FILL_B
    for pos, val in palette_cells.items():
        out[pos] = int(val)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each valid same-colored vertex set into a closed polygon and alternately recolor the enclosed regions.

    Input structure:
    - The background is black, and 2-3 polygon colors from {blue, green, yellow} mark sparse candidate vertices.
    - Each true polygon color has 3-5 valid vertices; the same color may also contain extra distractor pixels.
    - Two adjacent palette pixels grey and magenta sit near the lower-right corner and are not part of the outlines.

    Rule:
    - For each polygon color, order all same-colored pixels by polar angle and repeatedly remove a pixel whenever deleting it is the only way to recover a closed cycle whose edges are horizontal, vertical, or 45-degree diagonal and whose edge interiors are obstacle-free.
    - Draw the remaining cycle as that color's polygon outline, keeping only these valid outlines and ignoring the removed distractors.
    - Treat the combined outlines as walls, split the remaining cells into regions, build the wall-separated region adjacency graph, and 2-color it starting from the region containing (0,0): grey for the outside side and magenta for the opposite side.

    Key insight:
    - The extra pixels are identified geometrically, not by a local template: only the subset that can close into a clear H/V/45 polygon survives, and the final fill depends on the topology of all surviving outlines together.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    non_bg = {tuple(map(int, rc)) for rc in np.argwhere(g != 0)}
    cycles_by_color: dict[int, list[tuple[int, int]]] = {}
    for color in POLY_COLORS:
        pts = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        if not pts:
            continue
        reduced = _reduce_group(pts, non_bg)
        if reduced is None:
            raise ValueError(f"color {color} cannot be reduced to a valid polygon")
        cycle, _removed = reduced
        cycles_by_color[int(color)] = cycle

    if len(cycles_by_color) < 2:
        raise ValueError("expected at least two valid polygon groups")

    palette_cells = {
        tuple(map(int, rc)): int(g[tuple(map(int, rc))])
        for rc in np.argwhere(np.isin(g, list(PALETTE_COLORS)))
    }
    return _render_solution(g.shape, cycles_by_color, palette_cells)


def _normalize(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return [(r - min_r, c - min_c) for r, c in points]


def _transform(points: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]]:
    arr = np.array(points, dtype=int)
    if rng.random() < 0.5:
        arr = arr[:, ::-1]
    if rng.random() < 0.5:
        arr[:, 0] *= -1
    if rng.random() < 0.5:
        arr[:, 1] *= -1
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return [tuple(map(int, p)) for p in arr]


def _shape_triangle_right(rng: random.Random) -> list[tuple[int, int]]:
    s = rng.randint(4, 7)
    return [(0, 0), (0, s), (s, s)]


def _shape_triangle_iso(rng: random.Random) -> list[tuple[int, int]]:
    s = rng.randint(3, 5)
    return [(0, s), (s, 0), (s, 2 * s)]


def _shape_rectangle(rng: random.Random) -> list[tuple[int, int]]:
    h = rng.randint(5, 9)
    w = rng.randint(5, 9)
    return [(0, 0), (0, w), (h, w), (h, 0)]


def _shape_diamond(rng: random.Random) -> list[tuple[int, int]]:
    a = rng.randint(3, 5)
    return [(0, a), (a, 2 * a), (2 * a, a), (a, 0)]


def _shape_house(rng: random.Random) -> list[tuple[int, int]]:
    roof = rng.randint(3, 4)
    wall = rng.randint(3, 5)
    return [(0, roof), (roof, 0), (roof + wall, 0), (roof + wall, 2 * roof), (roof, 2 * roof)]


SHAPE_BUILDERS = (
    _shape_triangle_right,
    _shape_triangle_iso,
    _shape_rectangle,
    _shape_diamond,
    _shape_house,
)


def _build_random_shape(rng: random.Random) -> list[tuple[int, int]]:
    pts = rng.choice(SHAPE_BUILDERS)(rng)
    pts = _transform(pts, rng)
    return _angle_cycle(_normalize(pts))


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _translate(points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in points]


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    return min(r for r, _ in points), min(c for _, c in points), max(r for r, _ in points), max(c for _, c in points)


def _fits_in_rect(points: list[tuple[int, int]], rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in points)


def _place_disjoint(
    rng: random.Random,
    shape: tuple[int, int],
    cycle: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = shape
    br0, bc0, br1, bc1 = _bbox(cycle)
    sh = br1 - br0 + 1
    sw = bc1 - bc0 + 1
    row_hi = h - sh - 2
    col_hi = w - sw - 2
    if row_hi < 1 or col_hi < 1:
        return None
    for _ in range(PLACE_ATTEMPTS):
        dr = rng.randint(1, row_hi)
        dc = rng.randint(1, col_hi)
        moved = _translate(cycle, dr, dc)
        cells = _outline_cells(moved)
        if (0, 0) in cells:
            continue
        if cells & forbidden:
            continue
        return moved
    return None


def _place_nested_layout(
    rng: random.Random,
    shape: tuple[int, int],
    colors: list[int],
) -> dict[int, list[tuple[int, int]]] | None:
    h, w = shape
    outer = _angle_cycle(_shape_rectangle(rng))
    ob = _bbox(outer)
    oh = ob[2] - ob[0] + 1
    ow = ob[3] - ob[1] + 1
    if oh > h - 6 or ow > w - 8:
        return None
    for _ in range(PLACE_ATTEMPTS):
        dr = rng.randint(1, h - oh - 4)
        dc = rng.randint(1, w - ow - 4)
        outer_moved = _translate(outer, dr, dc)
        outer_cells = _outline_cells(outer_moved)
        if (0, 0) in outer_cells:
            continue

        inner = _build_random_shape(rng)
        ib = _bbox(inner)
        rect = (dr + 2, dc + 2, dr + oh - 3, dc + ow - 3)
        if rect[0] > rect[2] or rect[1] > rect[3]:
            continue
        if ib[2] - ib[0] > rect[2] - rect[0] or ib[3] - ib[1] > rect[3] - rect[1]:
            continue
        placed_inner = None
        for _ in range(PLACE_ATTEMPTS):
            ir = rng.randint(rect[0] - ib[0], rect[2] - ib[2])
            ic = rng.randint(rect[1] - ib[1], rect[3] - ib[3])
            cand = _translate(inner, ir, ic)
            if not _fits_in_rect(_outline_cells_as_list(cand), rect):
                continue
            cells = _outline_cells(cand)
            if cells & outer_cells:
                continue
            placed_inner = cand
            break
        if placed_inner is None:
            continue

        placed: dict[int, list[tuple[int, int]]] = {
            colors[0]: outer_moved,
            colors[1]: placed_inner,
        }
        if len(colors) == 3:
            forbidden = _expand_cells(outer_cells | _outline_cells(placed_inner), h, w, radius=1)
            third = _place_disjoint(rng, shape, _build_random_shape(rng), forbidden)
            if third is None:
                continue
            placed[colors[2]] = third
        return placed
    return None


def _outline_cells_as_list(cycle: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(_outline_cells(cycle))


def _place_disjoint_layout(
    rng: random.Random,
    shape: tuple[int, int],
    colors: list[int],
) -> dict[int, list[tuple[int, int]]] | None:
    h, w = shape
    placed: dict[int, list[tuple[int, int]]] = {}
    forbidden: set[tuple[int, int]] = set()
    palette_halo = {
        (rr, cc)
        for rr in range(h - 2, h)
        for cc in range(w - 3, w)
    }
    forbidden |= palette_halo
    for color in colors:
        cycle = _build_random_shape(rng)
        moved = _place_disjoint(rng, shape, cycle, forbidden)
        if moved is None:
            return None
        placed[color] = moved
        forbidden |= _expand_cells(_outline_cells(moved), h, w, radius=1)
    return placed


def _valid_single_removals(points: list[tuple[int, int]], grid: np.ndarray) -> list[tuple[int, int]]:
    non_bg = {tuple(map(int, rc)) for rc in np.argwhere(grid != 0)}
    out = []
    for p in points:
        subset = [q for q in points if q != p]
        if len(subset) < 3:
            continue
        if _cycle_if_valid(subset, non_bg - {p}) is not None:
            out.append(p)
    return out


def _find_distractor(
    rng: random.Random,
    grid: np.ndarray,
    cycle: list[tuple[int, int]],
    color: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    r0, c0, r1, c1 = _bbox(cycle)
    candidates = [
        (r, c)
        for r in range(max(1, r0 - 3), min(h - 1, r1 + 4))
        for c in range(max(1, c0 - 3), min(w - 1, c1 + 4))
        if grid[r, c] == 0
    ]
    rng.shuffle(candidates)
    vertices = list(cycle)
    for cand in candidates[:DISTRACTOR_ATTEMPTS]:
        trial = grid.copy()
        trial[cand] = int(color)
        pts = vertices + [cand]
        non_bg = {tuple(map(int, rc)) for rc in np.argwhere(trial != 0)}
        if _cycle_if_valid(pts, non_bg) is not None:
            continue
        removals = _valid_single_removals(pts, trial)
        if removals == [cand]:
            return cand
    return None


def _semantic_check(
    inp: np.ndarray,
    expected_cycles: dict[int, list[tuple[int, int]]],
    expected_distractors: dict[int, tuple[int, int]],
    palette_cells: dict[tuple[int, int], int],
) -> bool:
    if inp[0, 0] != 0:
        return False
    if len(expected_cycles) < 2:
        return False
    palette_positions = sorted(palette_cells)
    if len(palette_positions) != 2:
        return False
    if abs(palette_positions[0][0] - palette_positions[1][0]) + abs(palette_positions[0][1] - palette_positions[1][1]) != 1:
        return False

    non_bg = {tuple(map(int, rc)) for rc in np.argwhere(inp != 0)}
    for color, cycle in expected_cycles.items():
        pts = [tuple(map(int, rc)) for rc in np.argwhere(inp == color)]
        reduced = _reduce_group(pts, non_bg)
        if reduced is None:
            return False
        got_cycle, removed = reduced
        if set(got_cycle) != set(cycle):
            return False
        if removed != [expected_distractors[color]]:
            return False
        if _valid_single_removals(pts, inp) != [expected_distractors[color]]:
            return False

    try:
        expected = _render_solution(inp.shape, expected_cycles, palette_cells)
        label_map, _ = _region_labels_and_parity(_render_outline(inp.shape, expected_cycles) > 0)
    except ValueError:
        return False
    if len(np.unique(label_map)) < 3:
        return False
    return np.array_equal(main(inp), expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    palette_cells = {
        (h - 1, w - 2): Color.GREY,
        (h - 1, w - 1): Color.PINK,
    }
    for pos, val in palette_cells.items():
        grid[pos] = int(val)

    group_count = rng.choice((2, 3))
    colors = rng.sample(list(POLY_COLORS), group_count)
    layout = rng.choice(("disjoint", "nested"))
    if layout == "nested":
        cycles_by_color = _place_nested_layout(rng, (h, w), colors)
        if cycles_by_color is None:
            return None
    else:
        cycles_by_color = _place_disjoint_layout(rng, (h, w), colors)
        if cycles_by_color is None:
            return None

    input_grid = np.zeros((h, w), dtype=int)
    for pos, val in palette_cells.items():
        input_grid[pos] = int(val)
    for color, cycle in cycles_by_color.items():
        for r, c in cycle:
            if input_grid[r, c] != 0:
                return None
            input_grid[r, c] = int(color)

    distractors: dict[int, tuple[int, int]] = {}
    for color, cycle in cycles_by_color.items():
        cand = _find_distractor(rng, input_grid, cycle, color)
        if cand is None:
            return None
        input_grid[cand] = int(color)
        distractors[color] = cand

    if not _semantic_check(input_grid, cycles_by_color, distractors, palette_cells):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for new puzzle 5")
