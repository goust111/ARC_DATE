# Combo variant of a47bf94d (index 11)
# combo_spec: D8/3+D16/0
# Name: 分裂复制+一对多映射

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]

_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}

TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 24)
TOKEN_ROW_OPTIONS = [2, 3]
GEN_MAX_RETRIES = 80


# Design analysis for the fused variant:
# - Concrete behavior: each connected wire component now has one source token at the
#   top, one color-9 square splitter in the middle, and 2 or 3 empty leaf endpoints
#   at the bottom. A 2x2 splitter yields two leaves; a 3x3 splitter yields three.
#   Solid sources still collapse to a diamond in place; X-sources stay put. The emitted
#   complementary shape is copied to every reachable leaf, not just one far endpoint.
# - Complexity calibration: every sample includes both splitter sizes, so "size controls
#   fanout count" is visible within a single grid. One source is solid and one is X, so
#   the solver must infer both local shape change and replicated transmission.
# - Broken original assumptions: the old tracer assumed 1:1 routing and stopped at the
#   first opposite endpoint. That no longer holds, so traversal now treats a 9-rectangle
#   as a hyper-node that branches to all non-entry faces and collects every leaf.
# - Placement legality: the generator uses two disjoint splitter templates whose target
#   centers are spaced by at least three columns, preventing token overlap. Only the
#   9-rectangle branches; all ordinary 8-cells remain simple paths.
# - Visual observability: the left/right or left/center/right leaves make the 1:many
#   copy visible by eye, and the two square sizes repeat across examples. No hidden
#   ordering by color value or other non-visual feature is used.
# - New solver assumptions checked by the generator: each component has exactly one
#   token-bearing endpoint, 2 or 3 bare leaf endpoints, and a square 9-rectangle that
#   is the unique branch point. The constructor enforces those constraints directly.


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col in (bg, 5, 8, 9) or np.any(sub != col):
                continue
            maximal = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                maximal = False
                                break
                    if not maximal:
                        break
                if not maximal:
                    break
            if maximal:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 X patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col in (bg, 5, 8, 9):
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _X_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[dict]:
    """Find color-9 rectangles plus the adjacent wire cells visible on each face."""
    h, w = g.shape
    rects: list[dict] = []
    for comp9 in components_mask(g == 9, connectivity=4):
        cells = {(int(r), int(c)) for r, c in comp9}
        if not cells:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        faces: dict[str, list[tuple[int, int]]] = {}
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in cells or not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) not in (5, 8):
                    continue
                face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                faces.setdefault(face, []).append((nr, nc))
        rects.append(
            {
                "cells": cells,
                "bbox": (r0, c0, r1, c1),
                "size": (r1 - r0 + 1, c1 - c0 + 1),
                "faces": {face: sorted(set(pts)) for face, pts in faces.items()},
            }
        )
    return rects


def _build_track_mask(
    g: np.ndarray,
    bg: int,
    solids: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    xs: dict[tuple[int, int], tuple[int, tuple[int, int]]],
) -> np.ndarray:
    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs:
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < g.shape[0] and 0 <= cc2 < g.shape[1]:
                track_mask[rr, cc2] = False
    return track_mask


def _component_endpoints(comp: set[tuple[int, int]]) -> list[tuple[int, int]]:
    endpoints = []
    for r, c in comp:
        deg = 0
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in comp:
                deg += 1
        if deg <= 1:
            endpoints.append((r, c))
    return endpoints


def _stamp_token(out: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    cr, cc = center
    for dr, dc in offsets:
        out[cr + dr, cc + dc] = color


def _trace_leaves(
    comp: set[tuple[int, int]],
    cur: tuple[int, int],
    d: tuple[int, int],
    rect_lookup: dict[tuple[int, int], int],
    rects_9: list[dict],
    visited: frozenset[tuple[tuple[int, int], tuple[int, int]]],
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    state = (cur, d)
    if state in visited:
        return set()
    visited = visited | {state}

    ridx = rect_lookup.get(cur)
    if ridx is not None:
        rect = rects_9[ridx]
        entry_face = _DIR2FACE[d]
        leaves: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for face, pts in rect["faces"].items():
            if face == entry_face:
                continue
            for nxt in pts:
                leaves |= _trace_leaves(comp, nxt, _FACE2DIR[face], rect_lookup, rects_9, visited)
        return leaves

    rev = (-d[0], -d[1])
    forward: list[tuple[tuple[int, int], tuple[int, int]]] = []
    r, c = cur
    for nr, nc in neighbors4(r, c):
        if (nr, nc) not in comp:
            continue
        step = (nr - r, nc - c)
        if step == rev:
            continue
        forward.append(((nr, nc), step))

    if not forward:
        return {(cur, d)}

    leaves: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for nxt, nd in forward:
        leaves |= _trace_leaves(comp, nxt, nd, rect_lookup, rects_9, visited)
    return leaves


def _analyze_sources(grid: np.ndarray) -> tuple[dict, dict, list[dict], dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    comps = _components(_build_track_mask(g, bg, solids, xs))
    rects_9 = _find_9rects(g)
    rect_lookup = {cell: i for i, rect in enumerate(rects_9) for cell in rect["cells"]}

    sources: list[dict] = []
    h, w = g.shape
    for pts in comps:
        comp = set(pts)
        for ep in _component_endpoints(comp):
            found = None
            for odr, odc in _N4:
                cen = (ep[0] + 2 * odr, ep[1] + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue
                if cen in solids:
                    col, _ = solids[cen]
                    found = {"kind": "solid", "color": col, "center": cen, "endpoint": ep, "dir_in": (-odr, -odc)}
                    break
                if cen in xs:
                    col, _ = xs[cen]
                    found = {"kind": "x", "color": col, "center": cen, "endpoint": ep, "dir_in": (-odr, -odc)}
                    break
            if found is not None:
                found["comp"] = comp
                sources.append(found)
    return solids, xs, sources, rect_lookup


def _fanout_counts(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    solids, xs, sources, rect_lookup = _analyze_sources(g)
    del solids, xs
    rects_9 = _find_9rects(g)
    counts = []
    for src in sources:
        leaves = _trace_leaves(src["comp"], src["endpoint"], src["dir_in"], rect_lookup, rects_9, frozenset())
        leaves = {leaf for leaf in leaves if leaf[0] != src["endpoint"]}
        counts.append(len(leaves))
    return sorted(counts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each endpoint token through a wire component whose maroon square may fan out into several exits.

    Input structure:
    - Background cells are empty. Non-background cells form isolated wire trees, with one source token near the
      top endpoint of each tree and a square color-9 block embedded mid-route as the only branch node.

    Rule:
    - A solid 3x3 source collapses into a diamond at its original location and emits X-copies.
    - An X-shaped source remains in place and emits diamond-copies.
    - When the transmission enters a color-9 square, it is duplicated onto every exit face other than the face
      it came from; a 2x2 square therefore feeds two targets and a 3x3 square feeds three.

    Key insight:
    - The original shape-complement rule still applies, but routing is no longer a permutation. The maroon square
      is a visible splitter, so solving requires collecting all reachable leaf endpoints after the split.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    solids, xs, sources, rect_lookup = _analyze_sources(g)
    rects_9 = _find_9rects(g)

    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        _stamp_token(out, (cr, cc), col, _DIAMOND_OFF)

    for src in sources:
        leaves = _trace_leaves(src["comp"], src["endpoint"], src["dir_in"], rect_lookup, rects_9, frozenset())
        for end, d in leaves:
            if end == src["endpoint"]:
                continue
            dest = (end[0] + 2 * d[0], end[1] + 2 * d[1])
            if src["kind"] == "solid":
                _stamp_token(out, dest, src["color"], _X_OFF)
            else:
                _stamp_token(out, dest, src["color"], _DIAMOND_OFF)

    return out


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segments must be orthogonal")


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) not in (0, color):
            return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _group_geometry(split_size: int, rect_r0: int, rect_c0: int) -> dict:
    if split_size == 2:
        return {
            "split_size": 2,
            "rect_cells": [(rect_r0 + dr, rect_c0 + dc) for dr in range(2) for dc in range(2)],
            "source_col": rect_c0,
            "target_cols": [rect_c0 - 2, rect_c0 + 3],
            "branch_starts": [(rect_r0 + 1, rect_c0 - 1), (rect_r0 + 1, rect_c0 + 2)],
        }
    return {
        "split_size": 3,
        "rect_cells": [(rect_r0 + dr, rect_c0 + dc) for dr in range(3) for dc in range(3)],
        "source_col": rect_c0 + 1,
        "target_cols": [rect_c0 - 3, rect_c0 + 1, rect_c0 + 5],
        "branch_starts": [(rect_r0 + 1, rect_c0 - 1), (rect_r0 + 3, rect_c0 + 1), (rect_r0 + 1, rect_c0 + 3)],
    }


def _draw_group(
    grid: np.ndarray,
    split_size: int,
    rect_r0: int,
    rect_c0: int,
    top_ep_row: int,
    bot_ep_row: int,
) -> dict | None:
    geom = _group_geometry(split_size, rect_r0, rect_c0)
    source_col = geom["source_col"]

    if not _paint(grid, _segment((top_ep_row, source_col), (rect_r0 - 1, source_col)), 8):
        return None
    if not _paint(grid, geom["rect_cells"], 9):
        return None

    for start, dest_col in zip(geom["branch_starts"], geom["target_cols"]):
        if start[1] != dest_col:
            if not _paint(grid, _segment(start, (start[0], dest_col)), 8):
                return None
        if not _paint(grid, _segment((start[0], dest_col), (bot_ep_row, dest_col)), 8):
            return None

    return {
        "split_size": split_size,
        "source_center": (top_ep_row - 2, source_col),
        "target_centers": [(bot_ep_row + 2, c) for c in geom["target_cols"]],
        "rect_bbox": (
            rect_r0,
            rect_c0,
            rect_r0 + split_size - 1,
            rect_c0 + split_size - 1,
        ),
    }


def _make_layout(top_ep_row: int) -> tuple[int, list[tuple[int, int, int]]]:
    shift = random.randint(0, 1)
    if random.random() < 0.5:
        group_defs = [(2, top_ep_row + random.randint(3, 4), 4 + shift), (3, top_ep_row + random.randint(3, 5), 13 + shift)]
        max_col = 19 + shift
    else:
        group_defs = [(3, top_ep_row + random.randint(3, 5), 5 + shift), (2, top_ep_row + random.randint(3, 4), 16 + shift)]
        max_col = 19 + shift
    width = max(22, max_col + 2)
    return width, group_defs


def _gen_attempt() -> np.ndarray | None:
    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + 2
    height = random.randint(*GRID_H_RANGE)
    bot_token_row = height - top_token_row - 1
    bot_ep_row = bot_token_row - 2
    if bot_ep_row - top_ep_row < 8:
        return None

    width, group_defs = _make_layout(top_ep_row)
    grid = np.zeros((height, width), dtype=int)

    groups = []
    for split_size, rect_r0, rect_c0 in group_defs:
        geom = _draw_group(grid, split_size, rect_r0, rect_c0, top_ep_row, bot_ep_row)
        if geom is None:
            return None
        groups.append(geom)

    colors = random.sample(TOKEN_COLORS_POOL, len(groups))
    shapes = ["solid", "x"]
    random.shuffle(shapes)

    inp = grid.copy()
    out = grid.copy()
    for group, color, shape in zip(groups, colors, shapes):
        if shape == "solid":
            _stamp_token(inp, group["source_center"], color, _SOLID_OFF)
            _stamp_token(out, group["source_center"], color, _DIAMOND_OFF)
            for center in group["target_centers"]:
                _stamp_token(out, center, color, _X_OFF)
        else:
            _stamp_token(inp, group["source_center"], color, _X_OFF)
            _stamp_token(out, group["source_center"], color, _X_OFF)
            for center in group["target_centers"]:
                _stamp_token(out, center, color, _DIAMOND_OFF)

    if _fanout_counts(inp) != [2, 3]:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
