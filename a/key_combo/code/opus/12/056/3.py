from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
OUTLINE = 3
MARKER_POOL = (1, 2, 7, 8, 9)
SEED_POOL = (4, 5, 6)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 500
PAIR_ATTEMPTS = 220

SEED_MASKS = (
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _component(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    color = int(grid[start])
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _color_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _marker_colors(grid: np.ndarray, bg: int) -> list[int]:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    marker_colors: list[int] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        if coords.shape[0] != 2:
            continue
        comps = _color_components(grid == color)
        if len(comps) == 2 and all(len(comp) == 1 for comp in comps):
            marker_colors.append(int(color))
    return sorted(marker_colors)


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    steps = max(abs(dr), abs(dc))
    assert steps > 0
    assert abs(dr) in (0, steps)
    assert abs(dc) in (0, steps)
    sr = _sign(dr)
    sc = _sign(dc)
    return [(r0 + i * sr, c0 + i * sc) for i in range(steps + 1)]


def _parallelogram_parts(
    p1: tuple[int, int], p2: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    dr = p2[0] - p1[0]
    dc = p2[1] - p1[1]
    assert dr != 0 and dc != 0 and abs(dr) != abs(dc)
    diag_len = min(abs(dr), abs(dc))
    diag = (_sign(dr) * diag_len, _sign(dc) * diag_len)
    axis = (dr - diag[0], dc - diag[1])
    pd = (p1[0] + diag[0], p1[1] + diag[1])
    pa = (p1[0] + axis[0], p1[1] + axis[1])
    return diag, axis, [p1, pd, p2, pa]


def _cross(a: tuple[int, int], b: tuple[int, int], p: tuple[int, int]) -> int:
    ax, ay = a[1], a[0]
    bx, by = b[1], b[0]
    px, py = p[1], p[0]
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _strictly_inside(p: tuple[int, int], vertices: list[tuple[int, int]]) -> bool:
    signs = []
    for i in range(4):
        val = _cross(vertices[i], vertices[(i + 1) % 4], p)
        if val == 0:
            return False
        signs.append(val > 0)
    return all(signs) or not any(signs)


def _boundary_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    p1, pd, p2, pa = vertices
    cells: set[tuple[int, int]] = set()
    for a, b in ((p1, pd), (pd, p2), (p2, pa), (pa, p1)):
        cells.update(_segment_cells(a, b))
    return cells


def _interior_cells(shape: tuple[int, int], vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    rows = [r for r, _c in vertices]
    cols = [c for _r, c in vertices]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    cells: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _in_bounds(shape, r, c) and _strictly_inside((r, c), vertices):
                cells.add((r, c))
    return cells


def _tile_value(pair: dict, cell: tuple[int, int]) -> int:
    r, c = cell
    sr0, sc0, _sr1, _sc1 = pair["seed_bbox"]
    tile = pair["tile"]
    return int(tile[(r - sr0) % tile.shape[0], (c - sc0) % tile.shape[1]])


def _fill_assignments(pairs: list[dict]) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    owners: dict[tuple[int, int], int] = {}
    values: dict[tuple[int, int], int] = {}
    for idx, pair in sorted(enumerate(pairs), key=lambda item: item[1]["p1"]):
        for cell in sorted(pair["interior"]):
            if cell in owners:
                continue
            owners[cell] = idx
            values[cell] = _tile_value(pair, cell)
    return owners, values


def _render_output(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    _owners, values = _fill_assignments(pairs)
    for (r, c), value in values.items():
        out[r, c] = int(value)
    marker_colors = {pair["marker_color"] for pair in pairs}
    for pair in pairs:
        for r, c in pair["boundary"]:
            if int(grid[r, c]) not in marker_colors:
                out[r, c] = OUTLINE
    return out


def _extract_pair(grid: np.ndarray, marker_color: int, marker_colors: set[int], bg: int) -> dict:
    coords = sorted((int(r), int(c)) for r, c in np.argwhere(grid == marker_color))
    assert len(coords) == 2
    p1, p2 = coords
    diag, axis, vertices = _parallelogram_parts(p1, p2)

    below = (p1[0] + 1, p1[1])
    right = (p1[0], p1[1] + 1)
    anchor: tuple[int, int] | None = None
    if _in_bounds(grid.shape, *below):
        below_color = int(grid[below])
        if below_color != bg and below_color not in marker_colors:
            anchor = below
    if anchor is None:
        assert _in_bounds(grid.shape, *right)
        right_color = int(grid[right])
        assert right_color != bg and right_color not in marker_colors
        anchor = right

    seed_cells = _component(grid, anchor)
    seed_color = int(grid[anchor])
    sr0, sc0, sr1, sc1 = _bbox(seed_cells)
    tile = np.full((sr1 - sr0 + 1, sc1 - sc0 + 1), bg, dtype=int)
    for r, c in seed_cells:
        tile[r - sr0, c - sc0] = seed_color

    boundary = _boundary_cells(vertices)
    interior = _interior_cells(grid.shape, vertices)
    return {
        "marker_color": int(marker_color),
        "seed_color": seed_color,
        "p1": p1,
        "p2": p2,
        "diag": diag,
        "axis": axis,
        "vertices": vertices,
        "boundary": boundary,
        "interior": interior,
        "seed_cells": seed_cells,
        "seed_bbox": (sr0, sc0, sr1, sc1),
        "tile": tile,
        "mode": "vertical" if axis[0] != 0 else "horizontal",
        "poly_bbox": _bbox(boundary | interior),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each marker-defined parallelogram with a phase-aligned periodic copy of its nearby seed.

    Input structure
    - A black background contains 2-3 marker pairs, with exactly two isolated pixels of each marker color.
    - For each pair, the raster-earlier marker has a small 4-connected seed cluster next to it, using a non-marker color in a 2x2 or 2x3 bbox.
    - The two markers of a pair lie on different rows and columns, and their offset has unequal absolute row and column parts.

    Rule
    - For each marker pair, decompose the earlier-to-later offset into a maximal 45-degree diagonal vector plus the remaining axis-aligned vector.
    - Use those two vectors as adjacent sides of a parallelogram, draw its four edges in green while keeping all marker pixels unchanged, and treat only the strict interior as fillable.
    - Repeat the seed's bbox as an infinite periodic tile with phase locked to the seed's actual position, keep only the tile cells that land inside the parallelogram interior, and resolve interior overlaps by prioritizing the lexicographically earliest first marker.

    Key insight
    - The markers do not define a single segment: they imply a diagonal-plus-axis decomposition whose parallelogram determines where the nearby seed pattern must be tiled and phase-preserved.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    markers = _marker_colors(g, bg)
    assert 2 <= len(markers) <= 3
    marker_set = set(markers)
    pairs = [_extract_pair(g, color, marker_set, bg) for color in markers]
    return _render_output(g, pairs)


def _seed_cells_from_mask(
    top_left: tuple[int, int], mask: np.ndarray
) -> set[tuple[int, int]]:
    r0, c0 = top_left
    cells: set[tuple[int, int]] = set()
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                cells.add((r0 + r, c0 + c))
    return cells


def _make_pair(
    shape: tuple[int, int],
    marker_color: int,
    seed_color: int,
    mode: str,
    p1: tuple[int, int],
    dc_sign: int,
    diag_len: int,
    straight_len: int,
    seed_mask: np.ndarray,
) -> dict | None:
    if mode == "vertical":
        dr = diag_len + straight_len
        dc = dc_sign * diag_len
        seed_top_left = (p1[0], p1[1] + 1)
    else:
        dr = diag_len
        dc = dc_sign * (diag_len + straight_len)
        seed_top_left = (p1[0] + 1, p1[1])
    p2 = (p1[0] + dr, p1[1] + dc)
    if not _in_bounds(shape, *p1) or not _in_bounds(shape, *p2):
        return None

    seed_cells = _seed_cells_from_mask(seed_top_left, seed_mask)
    if not seed_cells or not all(_in_bounds(shape, r, c) for r, c in seed_cells):
        return None
    if p1 in seed_cells or p2 in seed_cells:
        return None

    diag, axis, vertices = _parallelogram_parts(p1, p2)
    boundary = _boundary_cells(vertices)
    interior = _interior_cells(shape, vertices)
    if not (10 <= len(interior) <= 70):
        return None
    if seed_cells & (boundary | interior):
        return None

    sr0, sc0 = seed_top_left
    sr1 = sr0 + seed_mask.shape[0] - 1
    sc1 = sc0 + seed_mask.shape[1] - 1
    tile = np.where(seed_mask, seed_color, BG).astype(int)
    pair = {
        "marker_color": int(marker_color),
        "seed_color": int(seed_color),
        "p1": p1,
        "p2": p2,
        "diag": diag,
        "axis": axis,
        "vertices": vertices,
        "boundary": boundary,
        "interior": interior,
        "seed_cells": seed_cells,
        "seed_bbox": (sr0, sc0, sr1, sc1),
        "tile": tile,
        "mode": mode,
        "poly_bbox": _bbox(boundary | interior),
    }
    if not _pair_semantics_ok(pair):
        return None
    return pair


def _pair_semantics_ok(pair: dict) -> bool:
    residues = {
        (
            (r - pair["seed_bbox"][0]) % pair["tile"].shape[0],
            (c - pair["seed_bbox"][1]) % pair["tile"].shape[1],
        )
        for r, c in pair["interior"]
    }
    if len(residues) < 3:
        return False

    interior_vals = {_tile_value(pair, cell) for cell in pair["interior"]}
    if pair["seed_color"] not in interior_vals or BG not in interior_vals:
        return False

    pr0, pc0, pr1, pc1 = pair["poly_bbox"]
    outside_candidates = [
        (r, c)
        for r in range(pr0, pr1 + 1)
        for c in range(pc0, pc1 + 1)
        if (r, c) not in pair["boundary"] and (r, c) not in pair["interior"]
    ]
    if not outside_candidates:
        return False
    if not any(_tile_value(pair, cell) == pair["seed_color"] for cell in outside_candidates):
        return False
    return True


def _placement_ok(pair: dict, placed: list[dict], require_overlap: bool) -> bool:
    pair_input = pair["seed_cells"] | {pair["p1"], pair["p2"]}
    saw_overlap = False
    for other in placed:
        other_input = other["seed_cells"] | {other["p1"], other["p2"]}
        if pair_input & other_input:
            return False
        if pair["boundary"] & other_input:
            return False
        if pair["interior"] & other_input:
            return False

        boundary_boundary = pair["boundary"] & other["boundary"]
        boundary_interior = pair["boundary"] & other["interior"]
        interior_boundary = pair["interior"] & other["boundary"]
        if require_overlap:
            if len(boundary_boundary) > 4 or len(boundary_interior) > 6 or len(interior_boundary) > 6:
                return False
        elif boundary_boundary or boundary_interior or interior_boundary:
            return False

        overlap = pair["interior"] & other["interior"]
        if overlap:
            saw_overlap = True
            if len(overlap) > 0.10 * len(pair["interior"]) or len(overlap) > 0.10 * len(other["interior"]):
                return False
    return saw_overlap if require_overlap else True


def _pair_value_difference_on_overlap(earlier: dict, later: dict) -> bool:
    overlap = earlier["interior"] & later["interior"]
    if not overlap:
        return True
    return any(_tile_value(earlier, cell) != _tile_value(later, cell) for cell in overlap)


def _build_grid(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for pair in pairs:
        grid[pair["p1"]] = pair["marker_color"]
        grid[pair["p2"]] = pair["marker_color"]
        for r, c in pair["seed_cells"]:
            grid[r, c] = pair["seed_color"]
    return grid


def _sample_pair(
    rng: random.Random,
    shape: tuple[int, int],
    marker_color: int,
    seed_color: int,
    mode: str,
    placed: list[dict],
    require_overlap: bool,
) -> dict | None:
    h, w = shape
    focus_box = None
    if require_overlap and placed:
        focus_box = placed[0]["poly_bbox"]

    for _ in range(PAIR_ATTEMPTS):
        dc_sign = rng.choice((-1, 1))
        diag_len = rng.randint(3, 6)
        straight_len = rng.randint(2, 5)
        seed_mask = rng.choice(SEED_MASKS)

        if focus_box is None:
            p1 = (rng.randint(1, h - 3), rng.randint(1, w - 4))
        else:
            r0, c0, r1, c1 = focus_box
            lo_r = max(1, r0 - 3)
            hi_r = min(h - 3, r1 + 1)
            lo_c = max(1, c0 - 4)
            hi_c = min(w - 4, c1 + 2)
            p1 = (rng.randint(lo_r, hi_r), rng.randint(lo_c, hi_c))

        pair = _make_pair(
            shape=shape,
            marker_color=marker_color,
            seed_color=seed_color,
            mode=mode,
            p1=p1,
            dc_sign=dc_sign,
            diag_len=diag_len,
            straight_len=straight_len,
            seed_mask=seed_mask,
        )
        if pair is None:
            continue
        if _placement_ok(pair, placed, require_overlap=require_overlap):
            return pair
    return None


def _sample_modes(rng: random.Random, count: int) -> list[str]:
    if count == 2:
        modes = ["horizontal", "vertical"]
        rng.shuffle(modes)
        return modes
    modes = ["horizontal", "vertical", rng.choice(["horizontal", "vertical"])]
    rng.shuffle(modes)
    return modes


def _global_semantics_ok(inp: np.ndarray, pairs: list[dict]) -> bool:
    out = _render_output(inp, pairs)
    solved = main(inp)
    if not np.array_equal(solved, out):
        return False

    owners, values = _fill_assignments(pairs)
    for idx, pair in enumerate(pairs):
        owned = [cell for cell, owner in owners.items() if owner == idx]
        if len(owned) < 8:
            return False
        owned_vals = {values[cell] for cell in owned}
        if pair["seed_color"] not in owned_vals or BG not in owned_vals:
            return False
        visible_seed = sum(int(out[cell] == pair["seed_color"]) for cell in owned)
        visible_bg = sum(int(out[cell] == BG) for cell in owned)
        if visible_seed < 3 or visible_bg < 1:
            return False

    ordered = [pair for _idx, pair in sorted(enumerate(pairs), key=lambda item: item[1]["p1"])]
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            if not _pair_value_difference_on_overlap(ordered[i], ordered[j]):
                return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        shape = (rng.randint(21, 26), rng.randint(21, 26))
        pair_count = 3 if rng.random() < 0.65 else 2
        modes = _sample_modes(rng, pair_count)
        marker_colors = rng.sample(list(MARKER_POOL), pair_count)
        seed_colors = rng.sample(list(SEED_POOL), pair_count)
        overlap_branch = pair_count >= 2 and rng.random() < 0.35

        pairs: list[dict] = []
        ok = True
        for idx in range(pair_count):
            pair = _sample_pair(
                rng=rng,
                shape=shape,
                marker_color=marker_colors[idx],
                seed_color=seed_colors[idx],
                mode=modes[idx],
                placed=pairs,
                require_overlap=bool(overlap_branch and idx == 1),
            )
            if pair is None:
                ok = False
                break
            pairs.append(pair)
        if not ok:
            continue

        inp = _build_grid(shape, pairs)
        if _global_semantics_ok(inp, pairs):
            return inp
    raise RuntimeError("failed to generate a valid ARC input")
