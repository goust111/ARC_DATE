import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400
GEN_PLACEMENT_RETRIES = 80
GEN_LAYOUTS = ((2, 2), (2, 3), (3, 2), (3, 3))
GEN_CELL_SIZES = (5, 6, 7)
GEN_MULTI_RECEIVE_PROB = 0.5
GEN_DECOY_PROB = 0.45


def _normalize(coords):
    coords = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _rotate90(shape):
    return _normalize((c, -r) for r, c in shape)


def _rotate270(shape):
    return _normalize((-c, r) for r, c in shape)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _majority_lines(g, axis):
    wall = g == GREY
    counts = wall.sum(axis=1 - axis)
    threshold = g.shape[1 - axis] / 2.0
    lines = [idx for idx, count in enumerate(counts) if count > threshold]
    assert lines, "expected divider lines"

    merged = []
    run = [lines[0]]
    for idx in lines[1:]:
        if idx == run[-1] + 1:
            run.append(idx)
        else:
            merged.append(run[len(run) // 2])
            run = [idx]
    merged.append(run[len(run) // 2])
    return merged


def _regions_from_grid(g):
    hlines = _majority_lines(g, axis=0)
    vlines = _majority_lines(g, axis=1)
    assert len(hlines) >= 3 and len(vlines) >= 3

    heights = [hlines[i + 1] - hlines[i] - 1 for i in range(len(hlines) - 1)]
    widths = [vlines[j + 1] - vlines[j] - 1 for j in range(len(vlines) - 1)]
    assert min(heights) > 0 and min(widths) > 0
    assert len(set(heights)) == 1 and len(set(widths)) == 1

    regions = {}
    for ri in range(len(hlines) - 1):
        for cj in range(len(vlines) - 1):
            r0 = hlines[ri] + 1
            r1 = hlines[ri + 1] - 1
            c0 = vlines[cj] + 1
            c1 = vlines[cj + 1] - 1
            regions[(ri, cj)] = {
                "ri": ri,
                "cj": cj,
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
            }
    return hlines, vlines, regions


def _neighbor(region_key, side, row_count, col_count):
    ri, cj = region_key
    if side == "top":
        return (ri - 1, cj) if ri > 0 else None
    if side == "bottom":
        return (ri + 1, cj) if ri + 1 < row_count else None
    if side == "left":
        return (ri, cj - 1) if cj > 0 else None
    if side == "right":
        return (ri, cj + 1) if cj + 1 < col_count else None
    raise ValueError(side)


def _hole_segments(g, region, side):
    if side == "top":
        arr = g[region["r0"] - 1, region["c0"] : region["c1"] + 1]
        horizontal = True
    elif side == "bottom":
        arr = g[region["r1"] + 1, region["c0"] : region["c1"] + 1]
        horizontal = True
    elif side == "left":
        arr = g[region["r0"] : region["r1"] + 1, region["c0"] - 1]
        horizontal = False
    elif side == "right":
        arr = g[region["r0"] : region["r1"] + 1, region["c1"] + 1]
        horizontal = False
    else:
        raise ValueError(side)

    holes = []
    start = None
    for idx, value in enumerate(arr):
        if int(value) == BLACK:
            if start is None:
                start = idx
        elif start is not None:
            length = idx - start
            holes.append(
                {
                    "start": start,
                    "length": length,
                    "shape": _normalize(
                        ((0, k) for k in range(length))
                        if horizontal
                        else ((k, 0) for k in range(length))
                    ),
                }
            )
            start = None
    if start is not None:
        length = len(arr) - start
        holes.append(
            {
                "start": start,
                "length": length,
                "shape": _normalize(
                    ((0, k) for k in range(length))
                    if horizontal
                    else ((k, 0) for k in range(length))
                ),
            }
        )
    return holes


def _matches_key(key_shape, hole_shape):
    return key_shape == _rotate90(hole_shape) or key_shape == _rotate270(hole_shape)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each region's key across the uniquely matching lockhole into the adjacent region.

    Input structure:
    - The background is black.
    - Grey one-cell-thick rows and columns partition the grid into equal rectangular regions.
    - Some regions contain one small connected monochrome key shape inside the region.
    - Shared grey borders may contain short black gaps that act as lockholes.

    Rule:
    - For every region that contains a key, inspect the lockholes on its four shared borders.
    - A lockhole matches when its gap shape becomes the key shape after a 90-degree rotation.
    - Copy the key into the adjacent region across that matching border, mirroring the key position across the divider.
    - Keep the original keys, divider lines, and lockholes unchanged; multiple received copies all remain if they do not overlap.

    Key insight:
    - The gap shape determines which border transmits the key, and only after that match is found can the mirrored destination in the neighboring region be determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    hlines, vlines, regions = _regions_from_grid(g)
    row_count = len(hlines) - 1
    col_count = len(vlines) - 1

    out = g.copy()

    for region_key, region in regions.items():
        sub = g[region["r0"] : region["r1"] + 1, region["c0"] : region["c1"] + 1]
        key_mask = (sub != BLACK) & (sub != GREY)
        comps = _components(key_mask)
        if not comps:
            continue
        assert len(comps) == 1, "each keyed region must contain exactly one key"
        coords = comps[0]
        colors = {int(sub[r, c]) for r, c in coords}
        assert len(colors) == 1, "key must be monochrome"
        color = next(iter(colors))
        key_shape = _normalize(coords)

        matches = []
        for side in SIDES:
            neighbor = _neighbor(region_key, side, row_count, col_count)
            if neighbor is None:
                continue
            for hole in _hole_segments(g, region, side):
                if _matches_key(key_shape, hole["shape"]):
                    matches.append((side, neighbor))
        assert len(matches) == 1, "each key must have exactly one matching lockhole"

        side, dest_key = matches[0]
        dest = regions[dest_key]
        for rr, cc in coords:
            if side in ("left", "right"):
                dst_r = rr
                dst_c = region["w"] - 1 - cc
            else:
                dst_r = region["h"] - 1 - rr
                dst_c = cc
            ar = dest["r0"] + dst_r
            ac = dest["c0"] + dst_c
            assert out[ar, ac] == BLACK, "received keys must land on empty cells"
            out[ar, ac] = color

    return out


def _all_regions(rows, cols):
    return [(r, c) for r in range(rows) for c in range(cols)]


def _adjacent_sides(region, rows, cols):
    ri, cj = region
    sides = []
    if ri > 0:
        sides.append("top")
    if ri + 1 < rows:
        sides.append("bottom")
    if cj > 0:
        sides.append("left")
    if cj + 1 < cols:
        sides.append("right")
    return sides


def _edge_key(region, side):
    ri, cj = region
    if side == "top":
        return ("h", ri, cj)
    if side == "bottom":
        return ("h", ri + 1, cj)
    if side == "left":
        return ("v", ri, cj)
    if side == "right":
        return ("v", ri, cj + 1)
    raise ValueError(side)


def _opposite(side):
    return {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left",
    }[side]


def _shape_cells(side, length, h, w, rng):
    if side in ("left", "right"):
        row = rng.randrange(h)
        col = rng.randrange(w - length + 1)
        return [(row, col + k) for k in range(length)]
    col = rng.randrange(w)
    row = rng.randrange(h - length + 1)
    return [(row + k, col) for k in range(length)]


def _mirror_cells(cells, side, h, w):
    mirrored = []
    for r, c in cells:
        if side in ("left", "right"):
            mirrored.append((r, w - 1 - c))
        else:
            mirrored.append((h - 1 - r, c))
    return mirrored


def _choose_transfers(rows, cols, rng):
    regions = _all_regions(rows, cols)
    all_edges = []
    for region in regions:
        for side in _adjacent_sides(region, rows, cols):
            all_edges.append((region, side, _neighbor(region, side, rows, cols)))

    if rng.random() < GEN_MULTI_RECEIVE_PROB:
        candidates = [region for region in regions if len(_adjacent_sides(region, rows, cols)) >= 2]
        if candidates:
            target = rng.choice(candidates)
            options = []
            for side in _adjacent_sides(target, rows, cols):
                src = _neighbor(target, side, rows, cols)
                options.append((src, _opposite(side), target))
            rng.shuffle(options)
            take = 2 if len(options) == 2 else rng.choice((2, 3))
            transfers = options[:take]
            return transfers

    rng.shuffle(all_edges)
    transfer_goal = rng.choice((2, 3))
    transfers = []
    used_sources = set()
    used_targets = set()
    for src, side, dst in all_edges:
        if src in used_sources or dst in used_targets:
            continue
        transfers.append((src, side, dst))
        used_sources.add(src)
        used_targets.add(dst)
        if len(transfers) == transfer_goal:
            break
    return transfers if len(transfers) >= 2 else None


def _place_transfer_specs(rows, cols, cell_h, cell_w, transfers, rng):
    reserved = {region: set() for region in _all_regions(rows, cols)}
    specs = []
    colors = rng.sample(KEY_COLORS, len(transfers))

    for color, (src, side, dst) in zip(colors, transfers):
        placed = False
        for _ in range(GEN_PLACEMENT_RETRIES):
            length = rng.choice((2, 3))
            cells = _shape_cells(side, length, cell_h, cell_w, rng)
            mirrored = _mirror_cells(cells, side, cell_h, cell_w)
            cell_set = set(cells)
            mirrored_set = set(mirrored)
            if reserved[src] & cell_set:
                continue
            if reserved[dst] & mirrored_set:
                continue
            reserved[src] |= cell_set
            reserved[dst] |= mirrored_set
            specs.append(
                {
                    "src": src,
                    "dst": dst,
                    "side": side,
                    "color": color,
                    "cells": cells,
                    "length": length,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return specs


def _hole_start(edge_kind, length, cell_h, cell_w, rng):
    span = cell_w if edge_kind == "h" else cell_h
    return rng.randrange(span - length + 1)


def _draw_grid(rows, cols, cell_h, cell_w, specs, rng):
    height = rows * cell_h + rows + 1
    width = cols * cell_w + cols + 1
    grid = np.full((height, width), BLACK, dtype=int)

    hlines = [i * (cell_h + 1) for i in range(rows + 1)]
    vlines = [j * (cell_w + 1) for j in range(cols + 1)]
    grid[hlines, :] = GREY
    grid[:, vlines] = GREY

    holes = {}
    for spec in specs:
        key = _edge_key(spec["src"], spec["side"])
        holes[key] = {
            "length": spec["length"],
            "start": _hole_start(key[0], spec["length"], cell_h, cell_w, rng),
        }

    unused_edges = []
    for region in _all_regions(rows, cols):
        for side in _adjacent_sides(region, rows, cols):
            key = _edge_key(region, side)
            if key in holes or key in unused_edges:
                continue
            unused_edges.append(key)

    rng.shuffle(unused_edges)
    if unused_edges and all(rng.random() >= GEN_DECOY_PROB for _ in range(len(unused_edges))):
        key = unused_edges[0]
        holes[key] = {"length": 1, "start": _hole_start(key[0], 1, cell_h, cell_w, rng)}
    for key in unused_edges:
        if key in holes:
            continue
        if rng.random() < GEN_DECOY_PROB:
            holes[key] = {"length": 1, "start": _hole_start(key[0], 1, cell_h, cell_w, rng)}

    for key, hole in holes.items():
        kind, a, b = key
        if kind == "h":
            wall_r = hlines[a]
            c0 = vlines[b] + 1 + hole["start"]
            grid[wall_r, c0 : c0 + hole["length"]] = BLACK
        else:
            wall_c = vlines[b]
            r0 = hlines[a] + 1 + hole["start"]
            grid[r0 : r0 + hole["length"], wall_c] = BLACK

    for spec in specs:
        ri, cj = spec["src"]
        base_r = hlines[ri] + 1
        base_c = vlines[cj] + 1
        for r, c in spec["cells"]:
            grid[base_r + r, base_c + c] = spec["color"]

    return grid


def _apply_dihedral(grid, op):
    if op == 0:
        return grid.copy()
    if op == 1:
        return np.rot90(grid, 1)
    if op == 2:
        return np.rot90(grid, 2)
    if op == 3:
        return np.rot90(grid, 3)
    if op == 4:
        return np.fliplr(grid)
    if op == 5:
        return np.flipud(grid)
    if op == 6:
        return np.rot90(np.fliplr(grid), 1)
    if op == 7:
        return np.rot90(np.flipud(grid), 1)
    raise ValueError(op)


def _gen_attempt(rng):
    rows, cols = rng.choice(GEN_LAYOUTS)
    cell_h = rng.choice(GEN_CELL_SIZES)
    cell_w = rng.choice(GEN_CELL_SIZES)
    if rows * cell_h + rows + 1 > 30 or cols * cell_w + cols + 1 > 30:
        return None

    transfers = _choose_transfers(rows, cols, rng)
    if not transfers:
        return None

    specs = _place_transfer_specs(rows, cols, cell_h, cell_w, transfers, rng)
    if specs is None:
        return None

    grid = _draw_grid(rows, cols, cell_h, cell_w, specs, rng)
    grid = _apply_dihedral(grid, rng.randrange(8))

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lockhole-key puzzle")
