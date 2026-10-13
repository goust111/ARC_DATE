from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(10))

# --------------- solver helpers ---------------


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _unit_step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    assert (dr == 0) != (dc == 0), "edge must be axis-aligned"
    if dr != 0:
        return (1 if dr > 0 else -1, 0)
    return (0, 1 if dc > 0 else -1)


def _edge_len(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def _rotate_point(point: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _shift_cells(cells: set[tuple[int, int]] | list[tuple[int, int]], dr: int, dc: int):
    if isinstance(cells, set):
        return {(r + dr, c + dc) for r, c in cells}
    return [(r + dr, c + dc) for r, c in cells]


def _arrow_cells(center: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    dr, dc = direction
    pr, pc = -dc, dr
    return {
        (r, c),
        (r + dr, c + dc),
        (r + pr, c + pc),
        (r - pr, c - pc),
    }


def _decode_arrow(coords: list[tuple[int, int]], color: int) -> dict:
    cells = {tuple(map(int, cell)) for cell in coords}
    assert len(cells) == 4, "each arrow must occupy four cells"

    degrees = {}
    for r, c in cells:
        deg = 0
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in cells:
                deg += 1
        degrees[(r, c)] = deg

    centers = [cell for cell, deg in degrees.items() if deg == 3]
    leaves = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(centers) == 1 and len(leaves) == 3, "arrow must be a T-shaped tetromino"
    center = centers[0]

    vectors = [(leaf[0] - center[0], leaf[1] - center[1]) for leaf in leaves]
    tip_vecs = [vec for vec in vectors if (-vec[0], -vec[1]) not in vectors]
    assert len(tip_vecs) == 1, "arrow direction must be unique"
    direction = tip_vecs[0]
    tip = (center[0] + direction[0], center[1] + direction[1])

    return {
        "color": int(color),
        "cells": cells,
        "center": center,
        "dir": direction,
        "tip": tip,
    }


def _extract_arrows(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    arrows: list[dict] = []
    for idx, comp in enumerate(_components(g != bg)):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each arrow component must be monochromatic"
        arrow = _decode_arrow(comp, next(iter(colors)))
        arrow["id"] = idx
        arrows.append(arrow)
    assert arrows, "expected at least one arrow"
    return arrows


def _trace_rays(grid: np.ndarray, bg: int, arrows: list[dict]) -> tuple[dict[int, int], dict[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cell_to_arrow: dict[tuple[int, int], int] = {}
    for arrow in arrows:
        for cell in arrow["cells"]:
            cell_to_arrow[cell] = int(arrow["id"])

    successors: dict[int, int] = {}
    segments: dict[int, list[tuple[int, int]]] = {}

    for arrow in arrows:
        rr, cc = arrow["tip"]
        dr, dc = arrow["dir"]
        rr += dr
        cc += dc
        segment: list[tuple[int, int]] = []

        while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == bg:
            segment.append((rr, cc))
            rr += dr
            cc += dc

        assert 0 <= rr < h and 0 <= cc < w, "ray must hit another arrow before leaving the grid"
        assert (rr, cc) in cell_to_arrow, "ray must terminate on an arrow"
        target = int(cell_to_arrow[(rr, cc)])
        assert target != int(arrow["id"]), "ray cannot hit its own arrow"

        successors[int(arrow["id"])] = target
        segments[int(arrow["id"])] = segment

    indegree = Counter(successors.values())
    assert all(indegree[int(arrow["id"])] == 1 for arrow in arrows), "arrow graph must be disjoint cycles"
    return successors, segments


def _cycle_decomposition(successors: dict[int, int]) -> list[list[int]]:
    unseen = set(successors)
    cycles: list[list[int]] = []

    while unseen:
        start = next(iter(unseen))
        cur = start
        cycle: list[int] = []
        while cur not in cycle:
            cycle.append(cur)
            unseen.remove(cur)
            cur = int(successors[cur])
        assert cur == start, "every arrow must lie on a closed cycle"
        cycles.append(cycle)

    return cycles


def _interior_cells(boundary: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(boundary)
    local_h = r1 - r0 + 3
    local_w = c1 - c0 + 3
    blocked = np.zeros((local_h, local_w), dtype=bool)

    for r, c in boundary:
        blocked[r - r0 + 1, c - c0 + 1] = True

    seen = np.zeros((local_h, local_w), dtype=bool)
    q = deque()

    def push(rr: int, cc: int) -> None:
        if blocked[rr, cc] or seen[rr, cc]:
            return
        seen[rr, cc] = True
        q.append((rr, cc))

    for rr in range(local_h):
        push(rr, 0)
        push(rr, local_w - 1)
    for cc in range(local_w):
        push(0, cc)
        push(local_h - 1, cc)

    while q:
        rr, cc = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < local_h and 0 <= nc < local_w and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    inside: set[tuple[int, int]] = set()
    for rr in range(1, local_h - 1):
        for cc in range(1, local_w - 1):
            if blocked[rr, cc] or seen[rr, cc]:
                continue
            inside.add((rr + r0 - 1, cc + c0 - 1))
    return inside


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode T-shaped arrows, trace their closed ray loops, and fill each enclosed
    orthogonal region with the majority arrow color of that loop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    arrows = _extract_arrows(g, bg)
    successors, segments = _trace_rays(g, bg, arrows)
    cycles = _cycle_decomposition(successors)
    arrow_by_id = {int(arrow["id"]): arrow for arrow in arrows}

    out = g.copy()

    for arrow in arrows:
        color = int(arrow["color"])
        for r, c in segments[int(arrow["id"])]:
            out[r, c] = color

    for cycle in cycles:
        boundary: set[tuple[int, int]] = set()
        votes = Counter()
        for arrow_id in cycle:
            arrow = arrow_by_id[int(arrow_id)]
            votes[int(arrow["color"])] += 1
            boundary |= arrow["cells"]
            boundary.update(segments[int(arrow_id)])

        fill_color = int(votes.most_common(1)[0][0])
        for r, c in _interior_cells(boundary):
            if int(g[r, c]) == bg and int(out[r, c]) == bg:
                out[r, c] = fill_color

    return out


# --------------- generator helpers ---------------

GRID_H_RANGE = (22, 36)
GRID_W_RANGE = (22, 36)
LOOP_COUNT_CHOICES = (1, 2, 3)
SHAPE_TYPES = ("rect", "l")
RECT_SIDE_RANGE = (6, 12)
L_ARM_RANGE = (5, 9)
PLACE_RETRIES = 160
GEN_RETRIES = 400


def _make_rect_vertices() -> list[tuple[int, int]]:
    height = random.randint(*RECT_SIDE_RANGE)
    width = random.randint(*RECT_SIDE_RANGE)
    return [(0, 0), (0, width), (height, width), (height, 0)]


def _make_l_vertices() -> list[tuple[int, int]]:
    a = random.randint(*L_ARM_RANGE)
    b = random.randint(*L_ARM_RANGE)
    c = random.randint(*L_ARM_RANGE)
    d = random.randint(*L_ARM_RANGE)
    return [(0, 0), (0, a + b), (d, a + b), (d, a), (d + c, a), (d + c, 0)]


def _build_loop_geometry(vertices: list[tuple[int, int]]) -> dict | None:
    n = len(vertices)
    if n < 4:
        return None

    directions = [_unit_step(vertices[i], vertices[(i + 1) % n]) for i in range(n)]
    lengths = [_edge_len(vertices[i], vertices[(i + 1) % n]) for i in range(n)]
    if any(length < 4 for length in lengths):
        return None

    raw_arrows: list[dict] = []
    raw_segments: list[list[tuple[int, int]]] = []
    boundary: set[tuple[int, int]] = set()

    for center, direction, length in zip(vertices, directions, lengths):
        cells = _arrow_cells(center, direction)
        tip = (center[0] + direction[0], center[1] + direction[1])
        raw_arrows.append({"center": center, "dir": direction, "tip": tip, "cells": cells})
        boundary |= cells

        segment: list[tuple[int, int]] = []
        for step in range(2, length - 1):
            cell = (center[0] + direction[0] * step, center[1] + direction[1] * step)
            segment.append(cell)
            boundary.add(cell)
        raw_segments.append(segment)

    if not boundary:
        return None

    min_r, min_c, max_r, max_c = _bbox(boundary)
    shift_r = -min_r
    shift_c = -min_c
    height = max_r - min_r + 1
    width = max_c - min_c + 1

    arrows: list[dict] = []
    segments: list[list[tuple[int, int]]] = []
    shifted_boundary: set[tuple[int, int]] = set()
    for raw_arrow, raw_segment in zip(raw_arrows, raw_segments):
        cells = _shift_cells(raw_arrow["cells"], shift_r, shift_c)
        tip = (raw_arrow["tip"][0] + shift_r, raw_arrow["tip"][1] + shift_c)
        center = (raw_arrow["center"][0] + shift_r, raw_arrow["center"][1] + shift_c)
        arrows.append({"center": center, "dir": raw_arrow["dir"], "tip": tip, "cells": cells})
        shifted_boundary |= cells

        shifted_segment = _shift_cells(raw_segment, shift_r, shift_c)
        segments.append(shifted_segment)
        shifted_boundary.update(shifted_segment)

    arrow_mask = np.zeros((height, width), dtype=bool)
    for arrow in arrows:
        for r, c in arrow["cells"]:
            if not (0 <= r < height and 0 <= c < width):
                return None
            if arrow_mask[r, c]:
                return None
            arrow_mask[r, c] = True
    if len(_components(arrow_mask)) != n:
        return None

    inside = _interior_cells(shifted_boundary)
    if not inside:
        return None

    return {
        "height": height,
        "width": width,
        "arrows": arrows,
        "segments": segments,
        "boundary": shifted_boundary,
        "inside": inside,
    }


def _sample_local_loop() -> tuple[str, dict] | tuple[None, None]:
    shape_type = random.choice(SHAPE_TYPES)
    if shape_type == "rect":
        vertices = _make_rect_vertices()
    else:
        vertices = _make_l_vertices()

    turns = random.randrange(4)
    vertices = [_rotate_point(vertex, turns) for vertex in vertices]
    offset = random.randrange(len(vertices))
    vertices = vertices[offset:] + vertices[:offset]

    geometry = _build_loop_geometry(vertices)
    if geometry is None:
        return None, None
    return shape_type, geometry


def _assign_arrow_colors(n: int, bg: int) -> tuple[list[int], int]:
    available = [color for color in PALETTE if color != bg]
    majority_color = random.choice(available)
    majority_count = random.randint(n // 2 + 1, n - 1)

    other_pool = [color for color in available if color != majority_color]
    other_count = random.randint(1, min(2, len(other_pool)))
    other_colors = random.sample(other_pool, other_count)

    colors = [majority_color] * majority_count
    for _ in range(n - majority_count):
        colors.append(random.choice(other_colors))
    random.shuffle(colors)
    return colors, majority_color


def _dilate_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))
    return blocked


def _translate_loop(local: dict, origin: tuple[int, int], colors: list[int], majority_color: int) -> dict:
    off_r, off_c = origin
    arrows: list[dict] = []
    segments: list[list[tuple[int, int]]] = []
    boundary: set[tuple[int, int]] = set()
    inside = _shift_cells(local["inside"], off_r, off_c)

    for idx, (arrow, segment) in enumerate(zip(local["arrows"], local["segments"])):
        cells = _shift_cells(arrow["cells"], off_r, off_c)
        tip = (arrow["tip"][0] + off_r, arrow["tip"][1] + off_c)
        center = (arrow["center"][0] + off_r, arrow["center"][1] + off_c)
        arrows.append(
            {
                "center": center,
                "dir": arrow["dir"],
                "tip": tip,
                "cells": cells,
                "color": int(colors[idx]),
            }
        )
        boundary |= cells

        shifted_segment = _shift_cells(segment, off_r, off_c)
        segments.append(shifted_segment)
        boundary.update(shifted_segment)

    return {
        "arrows": arrows,
        "segments": segments,
        "boundary": boundary,
        "inside": inside,
        "majority_color": int(majority_color),
        "shape": (local["height"], local["width"]),
    }


def _render_input(shape: tuple[int, int], bg: int, loops: list[dict]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for loop in loops:
        for arrow in loop["arrows"]:
            for r, c in arrow["cells"]:
                grid[r, c] = int(arrow["color"])
    return grid


def _expected_output(inp: np.ndarray, bg: int, loops: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for loop in loops:
        for arrow, segment in zip(loop["arrows"], loop["segments"]):
            for r, c in segment:
                out[r, c] = int(arrow["color"])
        for r, c in loop["inside"]:
            if int(inp[r, c]) == bg and int(out[r, c]) == bg:
                out[r, c] = int(loop["majority_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_loops = random.choice(LOOP_COUNT_CHOICES)
    min_h = max(GRID_H_RANGE[0], 18 + 4 * (n_loops - 1))
    min_w = max(GRID_W_RANGE[0], 18 + 4 * (n_loops - 1))
    h = random.randint(min_h, GRID_H_RANGE[1])
    w = random.randint(min_w, GRID_W_RANGE[1])
    bg = random.choice(PALETTE)

    placed: list[dict] = []
    blocked: set[tuple[int, int]] = set()

    for _ in range(n_loops):
        accepted = False
        for _ in range(PLACE_RETRIES):
            _shape_type, local = _sample_local_loop()
            if local is None:
                continue

            loop_h, loop_w = local["shape"] if "shape" in local else (local["height"], local["width"])
            if loop_h + 2 > h or loop_w + 2 > w:
                continue

            colors, majority_color = _assign_arrow_colors(len(local["arrows"]), bg)
            origin = (
                random.randint(1, h - loop_h - 1),
                random.randint(1, w - loop_w - 1),
            )
            loop = _translate_loop(local, origin, colors, majority_color)

            if any(cell in blocked for cell in loop["boundary"]):
                continue

            accepted = True
            placed.append(loop)
            blocked |= _dilate_cells(loop["boundary"], h, w, pad=1)
            break

        if not accepted:
            return None

    inp = _render_input((h, w), bg, placed)
    counts = Counter(int(value) for value in inp.ravel())
    if counts.most_common(1)[0][0] != bg:
        return None

    expected = _expected_output(inp, bg, placed)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid arrow-loop input")
