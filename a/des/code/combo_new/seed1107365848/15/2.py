from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
PATH_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
CANVAS_SIDE = 30
GEN_ATTEMPTS = 400

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LEFT_TURN = {
    (-1, 0): (0, -1),
    (1, 0): (0, 1),
    (0, -1): (1, 0),
    (0, 1): (-1, 0),
}
RIGHT_TURN = {v: k for k, v in LEFT_TURN.items()}

PATTERN_MASKS = [
    np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=bool),
]


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    result: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            result.append((rr, cc))
    return result


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, (h, w)):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr != 0:
        dr //= abs(dr)
    if dc != 0:
        dc //= abs(dc)
    assert (dr == 0) != (dc == 0), "path steps must be orthogonal"
    return dr, dc


def _trace_path(path_cells: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in path_cells:
        neighbors[cell] = [nb for nb in _neighbors4(*cell, shape) if nb in path_cells]
        assert len(neighbors[cell]) <= 2, "grey component must be a simple path"
    endpoints = [cell for cell, nbs in neighbors.items() if len(nbs) == 1]
    assert len(endpoints) == 2, "grey component must have exactly two endpoints"

    ordered = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while True:
        nxts = [nb for nb in neighbors[cur] if nb != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "grey path may not branch"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    assert len(ordered) == len(path_cells), "grey path may not self-intersect"
    assert ordered[-1] == endpoints[1], "grey path tracing must end at the other endpoint"
    return ordered


def _count_turns(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    dirs = [_direction(path[i], path[i + 1]) for i in range(len(path) - 1)]
    return sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _anchor_topleft(endpoint: tuple[int, int], anchor_to_endpoint: tuple[int, int]) -> tuple[int, int]:
    r, c = endpoint
    if anchor_to_endpoint == (0, 1):
        return r - 1, c - 3
    if anchor_to_endpoint == (0, -1):
        return r - 1, c + 1
    if anchor_to_endpoint == (1, 0):
        return r - 3, c - 1
    if anchor_to_endpoint == (-1, 0):
        return r + 1, c - 1
    raise AssertionError("invalid anchor direction")


def _find_anchors(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG_COLOR) & (grid != PATH_COLOR)
    anchors: list[dict] = []
    for comp in _components(mask):
        if len(comp) != 9:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if bool(((patch == BG_COLOR) | (patch == PATH_COLOR)).any()):
            continue

        vals, cnts = np.unique(patch, return_counts=True)
        if len(vals) not in (1, 2):
            continue

        entry = {
            "top_left": (int(r0), int(c0)),
            "cells": set(comp),
        }
        if len(vals) == 1:
            entry["is_source"] = False
            entry["base_color"] = int(vals[0])
            entry["pattern_color"] = None
            entry["pattern_mask"] = np.zeros((3, 3), dtype=bool)
        else:
            base_idx = int(np.argmax(cnts))
            pattern_idx = 1 - base_idx
            entry["is_source"] = True
            entry["base_color"] = int(vals[base_idx])
            entry["pattern_color"] = int(vals[pattern_idx])
            entry["pattern_mask"] = patch == int(vals[pattern_idx])
        anchors.append(entry)

    assert anchors and len(anchors) % 2 == 0, "expected paired 3x3 anchors"
    return anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each source-anchor pattern to its paired empty anchor, rotated by the path's turn count.

    Input structure:
    - Background is black. Grey one-pixel-wide orthogonal paths appear as separate 4-connected components.
    - Each path endpoint touches a filled 3x3 colored anchor block. A source anchor contains a second color forming
      a small non-symmetric motif; its paired target anchor is a uniform 3x3 block of a different color.
    - Paths do not branch: every grey component is a simple chain with exactly two endpoints.

    Rule:
    - Identify every 3x3 anchor and every grey path, then pair anchors to a path by endpoint adjacency.
    - For each pair, choose the anchor with two colors as the source and follow the grey cells from that anchor to the
      uniform anchor, counting how many times the path changes between horizontal and vertical.
    - Rotate the source motif clockwise by 90 degrees per turn and paint those motif cells into the target anchor using
      the source motif color, while leaving the target anchor's base color, the source anchor, and the grey path intact.

    Key insight:
    - The hidden quantity is not stored locally in either anchor: it must be recovered by tracing the connecting path,
      and that turn count alone determines the motif's rotation before it is stamped onto the destination anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    anchors = _find_anchors(g)
    anchor_lookup: dict[tuple[int, int], int] = {}
    for idx, anchor in enumerate(anchors):
        for cell in anchor["cells"]:
            anchor_lookup[cell] = idx

    path_components = _components(g == PATH_COLOR)
    assert path_components, "expected at least one grey path"
    usage = [0] * len(anchors)

    for comp in path_components:
        path_set = set(comp)
        ordered = _trace_path(path_set, g.shape)
        endpoints = (ordered[0], ordered[-1])

        endpoint_anchor_ids: list[int] = []
        for endpoint in endpoints:
            ids = {
                anchor_lookup[nb]
                for nb in _neighbors4(*endpoint, g.shape)
                if nb in anchor_lookup
            }
            assert len(ids) == 1, "each grey endpoint must touch exactly one anchor"
            endpoint_anchor_ids.append(ids.pop())

        a0, a1 = endpoint_anchor_ids
        assert a0 != a1, "a path must connect two distinct anchors"
        usage[a0] += 1
        usage[a1] += 1

        if anchors[a0]["is_source"]:
            src_idx, dst_idx = a0, a1
        else:
            src_idx, dst_idx = a1, a0
        assert anchors[src_idx]["is_source"] and not anchors[dst_idx]["is_source"], (
            "each path must connect one patterned anchor and one uniform anchor"
        )

        src_endpoint = endpoints[0] if endpoint_anchor_ids[0] == src_idx else endpoints[1]
        ordered_from_source = ordered if ordered[0] == src_endpoint else list(reversed(ordered))
        turns = _count_turns(ordered_from_source)

        src = anchors[src_idx]
        dst = anchors[dst_idx]
        rotated = np.rot90(src["pattern_mask"], -(turns % 4))
        rr, cc = dst["top_left"]
        out[rr : rr + 3, cc : cc + 3][rotated] = int(src["pattern_color"])

    assert all(count == 1 for count in usage), "every anchor must belong to exactly one path"
    return out


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = [points[0]]
    for start, end in zip(points, points[1:]):
        dr, dc = _direction(start, end)
        steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        for _ in range(steps):
            rr = cells[-1][0] + dr
            cc = cells[-1][1] + dc
            cells.append((rr, cc))
    assert len(cells) == len(set(cells)), "generator paths must not self-intersect"
    return cells


def _build_canonical_path(turns: int) -> list[tuple[int, int]]:
    if turns == 0:
        a = random.randint(2, 7)
        points = [(0, 0), (0, a)]
    elif turns == 1:
        a = random.randint(2, 6)
        b = random.randint(2, 6)
        s = random.choice([-1, 1])
        points = [(0, 0), (0, a), (s * b, a)]
    elif turns == 2:
        a = random.randint(5, 8)
        b = random.randint(2, 5)
        c = random.randint(2, a - 2)
        s = random.choice([-1, 1])
        points = [(0, 0), (0, a), (s * b, a), (s * b, a - c)]
    else:
        assert turns == 3
        a = random.randint(5, 8)
        b = random.randint(2, 4)
        c = random.randint(2, a - 2)
        d = random.randint(2, 4)
        s = random.choice([-1, 1])
        points = [(0, 0), (0, a), (s * b, a), (s * b, a - c), (s * (b + d), a - c)]
    return _polyline_cells(points)


def _pattern_mask() -> np.ndarray:
    mask = random.choice(PATTERN_MASKS).copy()
    return np.rot90(mask, -random.randrange(4))


def _crop_to_content(grid: np.ndarray, border: int) -> np.ndarray:
    pts = np.argwhere(grid != BG_COLOR)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0 = max(0, int(r0) - border)
    c0 = max(0, int(c0) - border)
    r1 = min(grid.shape[0] - 1, int(r1) + border)
    c1 = min(grid.shape[1] - 1, int(c1) + border)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _build_pair(turns: int) -> np.ndarray:
    for _ in range(100):
        path = _build_canonical_path(turns)
        if random.random() < 0.5:
            path = list(reversed(path))

        first_dir = _direction(path[0], path[1])
        last_dir = _direction(path[-2], path[-1])
        src_tl = _anchor_topleft(path[0], first_dir)
        dst_tl = _anchor_topleft(path[-1], (-last_dir[0], -last_dir[1]))

        source_cells = {
            (src_tl[0] + dr, src_tl[1] + dc)
            for dr in range(3)
            for dc in range(3)
        }
        target_cells = {
            (dst_tl[0] + dr, dst_tl[1] + dc)
            for dr in range(3)
            for dc in range(3)
        }
        occupied = set(path) | source_cells | target_cells
        if not source_cells.isdisjoint(target_cells):
            continue
        if not set(path).isdisjoint(source_cells | target_cells):
            continue

        r0, c0, r1, c1 = _bbox(occupied)
        shift_r = 1 - r0
        shift_c = 1 - c0
        h = r1 - r0 + 3
        w = c1 - c0 + 3
        grid = np.full((h, w), BG_COLOR, dtype=int)

        for r, c in path:
            grid[r + shift_r, c + shift_c] = PATH_COLOR

        src_base, dst_base, pat_color = random.sample(PALETTE, 3)
        mask = _pattern_mask()
        sr, sc = src_tl[0] + shift_r, src_tl[1] + shift_c
        tr, tc = dst_tl[0] + shift_r, dst_tl[1] + shift_c
        grid[sr : sr + 3, sc : sc + 3] = src_base
        grid[sr : sr + 3, sc : sc + 3][mask] = pat_color
        grid[tr : tr + 3, tc : tc + 3] = dst_base

        grid = np.rot90(grid, random.randrange(4))
        return _crop_to_content(grid, border=1)
    raise RuntimeError("failed to build a non-overlapping anchor-path pair")


def _place_local(canvas: np.ndarray, blocked: np.ndarray, local: np.ndarray) -> bool:
    h, w = canvas.shape
    lh, lw = local.shape
    if lh > h or lw > w:
        return False
    positions = [(r, c) for r in range(h - lh + 1) for c in range(w - lw + 1)]
    random.shuffle(positions)
    for r, c in positions:
        if bool(blocked[r : r + lh, c : c + lw].any()):
            continue
        region = canvas[r : r + lh, c : c + lw]
        mask = local != BG_COLOR
        if bool((region[mask] != BG_COLOR).any()):
            continue
        region[mask] = local[mask]
        blocked[r : r + lh, c : c + lw] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    canvas = np.full((CANVAS_SIDE, CANVAS_SIDE), BG_COLOR, dtype=int)
    blocked = np.zeros_like(canvas, dtype=bool)

    n_pairs = random.randint(1, 3)
    locals_ = [_build_pair(random.randrange(4)) for _ in range(n_pairs)]
    locals_.sort(key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True)

    for local in locals_:
        if not _place_local(canvas, blocked, local):
            return None

    result = _crop_to_content(canvas, border=random.randint(0, 2))
    try:
        out = main(result)
    except Exception:
        return None
    if np.array_equal(out, result):
        return None
    return result


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
