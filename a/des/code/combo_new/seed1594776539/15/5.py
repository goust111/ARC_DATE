from __future__ import annotations

import random

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EDGE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
EDGE_SUBSETS = [
    subset
    for mask in range(1, 1 << 4)
    for subset in [[edge for bit, edge in enumerate(("top", "bottom", "left", "right")) if mask & (1 << bit)]]
]

GEN_ATTEMPTS = 400
OBJECT_COUNT_RANGE = (1, 4)
OBJECT_H_RANGE = (5, 6)
OBJECT_W_RANGE = (5, 6)
HOLE_COUNT_RANGE = (1, 3)
SPEC_ATTEMPTS = 200
SLOT_EXTRA_RANGE = (0, 1)
OUTER_MARGIN_RANGE = (1, 2)
INTER_GAP_RANGE = (1, 2)
NON_BG_COLORS = list(range(1, 10))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if not _in_bounds(mask.shape, nr, nc):
                        continue
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if g[r, c] == BG or seen[r, c]:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _count_holes(window: np.ndarray) -> int:
    holes = 0
    for coords in _components_from_mask(window == BG):
        rows = coords[:, 0]
        cols = coords[:, 1]
        touches_edge = bool(
            np.any((rows == 0) | (rows == window.shape[0] - 1) | (cols == 0) | (cols == window.shape[1] - 1))
        )
        if not touches_edge:
            holes += 1
    return holes


def _boundary_gaps(window: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = window.shape
    gaps: list[tuple[int, int, int, int]] = []
    for c in range(1, w - 1):
        if int(window[0, c]) == BG:
            gaps.append((0, c, -1, 0))
        if int(window[h - 1, c]) == BG:
            gaps.append((h - 1, c, 1, 0))
    for r in range(1, h - 1):
        if int(window[r, 0]) == BG:
            gaps.append((r, 0, 0, -1))
        if int(window[r, w - 1]) == BG:
            gaps.append((r, w - 1, 0, 1))
    return gaps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each object's internal hole count and emit same-colored outward rays from every bounding-box edge gap.

    Input structure:
    - The background is black.
    - The grid contains several separated monochromatic 4-connected objects.
    - Each object nearly fills its axis-aligned bounding box, but may contain internal black holes and one-cell black gaps on some bounding-box edges.

    Rule:
    - For each object, count the black hole components inside its bounding box that do not touch the box boundary.
    - Find every black gap cell on the non-corner boundary cells of that bounding box.
    - Keep the object unchanged, keep the holes black, and from each gap draw a same-colored ray outward perpendicular to that edge with length equal to the hole count.

    Key insight:
    - The internal hole count is the reusable numeric parameter, while the edge gaps determine both ray origins and ray directions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    for obj in _objects(g):
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        ray_len = _count_holes(window)
        if ray_len == 0:
            continue
        for gr, gc, dr, dc in _boundary_gaps(window):
            for step in range(1, ray_len + 1):
                rr = r0 + gr + dr * step
                cc = c0 + gc + dc * step
                if _in_bounds(g.shape, rr, cc):
                    out[rr, cc] = color

    return out


def _is_four_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _sample_holes(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if len(candidates) < count:
        return None
    for _ in range(200):
        holes = random.sample(candidates, count)
        if all(not _is_four_adjacent(holes[i], holes[j]) for i in range(count) for j in range(i + 1, count)):
            return holes
    return None


def _gap_positions(h: int, w: int, edges: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    gaps: dict[str, tuple[int, int]] = {}
    for edge in edges:
        if edge == "top":
            gaps[edge] = (0, random.randint(1, w - 2))
        elif edge == "bottom":
            gaps[edge] = (h - 1, random.randint(1, w - 2))
        elif edge == "left":
            gaps[edge] = (random.randint(1, h - 2), 0)
        else:
            gaps[edge] = (random.randint(1, h - 2), w - 1)
    return gaps


def _object_spec() -> dict[str, object] | None:
    for _ in range(SPEC_ATTEMPTS):
        hole_count = random.randint(*HOLE_COUNT_RANGE)
        h = random.randint(*OBJECT_H_RANGE)
        w = random.randint(*OBJECT_W_RANGE)
        color = random.choice(NON_BG_COLORS)
        edges = tuple(random.choice(EDGE_SUBSETS))
        gaps = _gap_positions(h, w, edges)

        forbidden: set[tuple[int, int]] = set()
        for gr, gc in gaps.values():
            for dr, dc in DIR4:
                nr = gr + dr
                nc = gc + dc
                if 1 <= nr <= h - 2 and 1 <= nc <= w - 2:
                    forbidden.add((nr, nc))

        candidates = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if (r, c) not in forbidden
        ]
        holes = _sample_holes(candidates, hole_count)
        if holes is None:
            continue

        patch = np.full((h, w), color, dtype=int)
        for r, c in holes:
            patch[r, c] = BG
        for r, c in gaps.values():
            patch[r, c] = BG

        input_cells = {(int(r), int(c)) for r in range(h) for c in range(w) if int(patch[r, c]) != BG}
        ray_cells: set[tuple[int, int]] = set()
        for edge, (gr, gc) in gaps.items():
            dr, dc = EDGE_TO_DIR[edge]
            for step in range(1, hole_count + 1):
                ray_cells.add((gr + dr * step, gc + dc * step))

        output_cells = input_cells | ray_cells
        min_r = min(r for r, _c in output_cells)
        max_r = max(r for r, _c in output_cells)
        min_c = min(c for _r, c in output_cells)
        max_c = max(c for _r, c in output_cells)

        return {
            "color": color,
            "patch": patch,
            "hole_count": hole_count,
            "gaps": gaps,
            "input_cells": input_cells,
            "ray_cells": ray_cells,
            "min_r": min_r,
            "max_r": max_r,
            "min_c": min_c,
            "max_c": max_c,
            "out_h": max_r - min_r + 1,
            "out_w": max_c - min_c + 1,
        }
    return None


def _layout_shape(n_objects: int) -> tuple[int, int]:
    if n_objects == 1:
        return 1, 1
    if n_objects == 2:
        return random.choice([(1, 2), (2, 1)])
    return 2, 2


def _slot_assignment(specs: list[dict[str, object]], rows: int, cols: int) -> list[list[dict[str, object] | None]]:
    slots = [[None for _ in range(cols)] for _ in range(rows)]
    indices = list(range(rows * cols))
    random.shuffle(indices)
    shuffled = specs[:]
    random.shuffle(shuffled)
    for spec, idx in zip(shuffled, indices):
        slots[idx // cols][idx % cols] = spec
    return slots


def _slot_sizes(slots: list[list[dict[str, object] | None]]) -> tuple[list[int], list[int]]:
    rows = len(slots)
    cols = len(slots[0])
    row_heights: list[int] = []
    col_widths: list[int] = []
    for r in range(rows):
        req = max(int(slot["out_h"]) for slot in slots[r] if slot is not None)
        row_heights.append(req + random.randint(*SLOT_EXTRA_RANGE))
    for c in range(cols):
        req = max(int(slots[r][c]["out_w"]) for r in range(rows) if slots[r][c] is not None)
        col_widths.append(req + random.randint(*SLOT_EXTRA_RANGE))
    return row_heights, col_widths


def _starts(sizes: list[int], outer_before: int, gaps: list[int]) -> list[int]:
    pos = outer_before
    out: list[int] = []
    for i, size in enumerate(sizes):
        out.append(pos)
        pos += size
        if i < len(gaps):
            pos += gaps[i]
    return out


def _render_case(specs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = _layout_shape(len(specs))
    slots = _slot_assignment(specs, rows, cols)
    row_heights, col_widths = _slot_sizes(slots)

    row_gaps = [random.randint(*INTER_GAP_RANGE) for _ in range(rows - 1)]
    col_gaps = [random.randint(*INTER_GAP_RANGE) for _ in range(cols - 1)]
    top_margin = random.randint(*OUTER_MARGIN_RANGE)
    bottom_margin = random.randint(*OUTER_MARGIN_RANGE)
    left_margin = random.randint(*OUTER_MARGIN_RANGE)
    right_margin = random.randint(*OUTER_MARGIN_RANGE)

    height = top_margin + sum(row_heights) + sum(row_gaps) + bottom_margin
    width = left_margin + sum(col_widths) + sum(col_gaps) + right_margin
    if max(height, width) > 30:
        return None

    row_starts = _starts(row_heights, top_margin, row_gaps)
    col_starts = _starts(col_widths, left_margin, col_gaps)
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)

    for sr in range(rows):
        for sc in range(cols):
            spec = slots[sr][sc]
            if spec is None:
                continue
            extra_h = row_heights[sr] - int(spec["out_h"])
            extra_w = col_widths[sc] - int(spec["out_w"])
            anchor_r = row_starts[sr] + random.randint(0, extra_h)
            anchor_c = col_starts[sc] + random.randint(0, extra_w)
            box_r0 = anchor_r - int(spec["min_r"])
            box_c0 = anchor_c - int(spec["min_c"])

            patch = spec["patch"]
            ph, pw = patch.shape
            inp[box_r0 : box_r0 + ph, box_c0 : box_c0 + pw] = patch
            out[box_r0 : box_r0 + ph, box_c0 : box_c0 + pw] = patch
            for rr, cc in spec["ray_cells"]:
                out[box_r0 + rr, box_c0 + cc] = int(spec["color"])

    return inp, out


def _valid(inp: np.ndarray, expected: np.ndarray, n_objects: int) -> bool:
    if len(_objects(inp)) != n_objects:
        return False
    for obj in _objects(inp):
        r0, c0, r1, c1 = obj["bbox"]
        window = inp[r0 : r1 + 1, c0 : c1 + 1]
        if _count_holes(window) < 1:
            return False
        if not _boundary_gaps(window):
            return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    specs: list[dict[str, object]] = []
    for _ in range(n_objects):
        spec = _object_spec()
        if spec is None:
            return None
        specs.append(spec)

    rendered = _render_case(specs)
    if rendered is None:
        return None
    inp, expected = rendered
    if not _valid(inp, expected, n_objects):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1594776539/15/5 input")
