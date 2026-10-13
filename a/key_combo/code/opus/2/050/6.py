"""Transmit and cycle token shapes through disjoint gray paths.

Input structure:
  A black background with 2-3 thin gray (5) paths connecting two ends.
  One end near each path is a complete 3x3 token (solid block, X, or diamond).
  The opposite end is a 3x3-area fragment of the cycled token color.

Rule:
  On each path, replace the complete source token with its cycle
  (solid→X, X→diamond, diamond→solid) in the same color. At the destination end,
  paint only the missing cells of the destination cycled token shape inside the
  fragment 3x3 window using the fragment color.

Key insight:
  The transported shape is determined first by source shape cycle, while observed
  fragment occupancy acts as a mask of already-present cells.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import numpy as np


SOLID = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)}
X_SHAPE = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
DIAMOND = {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)}
CYCLE = {tuple(sorted(SOLID)): X_SHAPE, tuple(sorted(X_SHAPE)): DIAMOND, tuple(sorted(DIAMOND)): SOLID}


def _shape_id(shape: Set[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted(shape))


def _connected_components(mask: np.ndarray) -> List[Set[Tuple[int, int]]]:
    H, W = mask.shape
    vis = np.zeros((H, W), dtype=bool)
    comps: List[Set[Tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp: Set[Tuple[int, int]] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not vis[nr, nc] and mask[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_of(comp: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rows = [p[0] for p in comp]
    cols = [p[1] for p in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_from_bbox(grid: np.ndarray, comp: Set[Tuple[int, int]]) -> Tuple[Set[Tuple[int, int]], Tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_of(comp)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h != 3 or w != 3:
        return set(), (r0, c0, r1, c1)
    pts = {(r - r0, c - c0) for r, c in comp}
    return pts, (r0, c0, r1, c1)


def _path_components(grid: np.ndarray) -> List[Set[Tuple[int, int]]]:
    return _connected_components(grid == 5)


def _path_endpoints(comp: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    S = set(comp)
    ends = []
    for r, c in comp:
        deg = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in S:
                deg += 1
        if deg == 1:
            ends.append((r, c))
    return ends


def _find_fragment_window(
    fragment: Set[Tuple[int, int]],
    target_shape: Set[Tuple[int, int]],
) -> Tuple[int, int] | None:
    if not fragment:
        return None
    rs = [p[0] for p in fragment]
    cs = [p[1] for p in fragment]
    r0, c0 = min(rs), min(cs)
    H = 3
    W = 3
    for top in range(r0 - 2, r0 + 1):
        for left in range(c0 - 2, c0 + 1):
            if top < 0 or left < 0:
                continue
            # check target 3x3 inside bounds outside context when called
            locals = {(r - top, c - left) for r, c in fragment if 0 <= r - top < H and 0 <= c - left < W}
            if len(locals) != len(fragment):
                continue
            if locals.issubset(target_shape):
                return top, left
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=np.int64, copy=True)
    if g.ndim != 2:
        return g.copy()
    H, W = g.shape

    path_components = _path_components(g)
    token_comps = []
    frag_comps = []
    for comp in _connected_components(g != 0):
        pts, bbox = _shape_from_bbox(g, comp)
        h = bbox[2] - bbox[0] + 1
        w = bbox[3] - bbox[1] + 1
        if len(comp) == 9 and h == 3 and w == 3 and _shape_id(pts) in CYCLE:
            token_comps.append((comp, pts, bbox))
        elif len(comp) <= 4 and h <= 3 and w <= 3:
            # likely fragment
            frag_comps.append((comp, pts, bbox))

    frag_cells_by_endpoint = {}
    # touch each path component
    out = g.copy()
    for path in path_components:
        ends = _path_endpoints(path)
        if len(ends) != 2:
            continue
        e1, e2 = ends
        e1tokens = [(c, pts, bbox) for c, pts, bbox in token_comps if c & set(_neighbors(g, e1))]
        e2tokens = [(c, pts, bbox) for c, pts, bbox in token_comps if c & set(_neighbors(g, e2))]
        # helper choose side with token
        if e1tokens and not e2tokens:
            token_side = e1
            frag_end = e2
            token_data = e1tokens[0]
        elif e2tokens and not e1tokens:
            token_side = e2
            frag_end = e1
            token_data = e2tokens[0]
        elif e1tokens and e2tokens:
            token_data = e1tokens[0]
            token_side = e1
            frag_end = e2
        else:
            continue

        comp, src_shape, src_bbox = token_data
        target_shape = CYCLE[_shape_id(src_shape)]
        r0, c0, r1, c1 = src_bbox
        # paint source transformed token
        for tr, tc in target_shape:
            rr, cc = r0 + tr, c0 + tc
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = g[r0 + tr, c0 + tc]
        # pick fragment near frag_end
        target_candidates = [f for f in frag_comps if f[0] & set(_neighbors(g, frag_end))]
        if not target_candidates:
            continue
        _, frag_shape, frag_bbox = target_candidates[0]
        frag_set = {p for p in target_candidates[0][0]}
        ws = _find_fragment_window(frag_set, target_shape)
        if ws is None:
            continue
        top, left = ws
        if top < 0 or left < 0 or top + 3 > H or left + 3 > W:
            continue
        frag_local = {(r - top, c - left) for r, c in frag_set}
        # fragment color
        color = 4
        for rr, cc in frag_set:
            color = int(g[rr, cc])
            break
        for tr, tc in target_shape:
            rr, cc = top + tr, left + tc
            if (tr, tc) not in frag_local and (0 <= rr < H and 0 <= cc < W):
                out[rr, cc] = color

    return out


def _neighbors(grid: np.ndarray, p: Tuple[int, int]) -> Set[Tuple[int, int]]:
    H, W = grid.shape
    r, c = p
    out = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = r + dr, c + dc
            if (dr == 0 and dc == 0) or not (0 <= nr < H and 0 <= nc < W):
                continue
            out.add((nr, nc))
    return out


def generate_input() -> np.ndarray:
    H = 16
    W = 16
    g = np.zeros((H, W), dtype=np.int64)

    colors_tokens = [1, 2, 3]
    colors_frags = [4, 7, 6]
    n_paths = random.randint(2, 3)
    used = set()

    def paint_path(path: List[Tuple[int, int]], color: int = 5) -> None:
        for r, c in path:
            g[r, c] = color
            used.add((r, c))

    def gen_l_path() -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
        for _ in range(500):
            r1 = random.randint(1, H - 2)
            c1 = random.randint(1, W - 2)
            r2 = random.randint(1, H - 2)
            c2 = random.randint(1, W - 2)
            if abs(r1 - r2) + abs(c1 - c2) < 6:
                continue
            # L shape with no overlap
            path = []
            ok = True
            if r1 <= r2:
                for r in range(r1, r2 + 1):
                    path.append((r, c1))
            else:
                for r in range(r2, r1 + 1):
                    path.append((r, c1))
            if c1 <= c2:
                for c in range(c1 + 1, c2 + 1):
                    path.append((r2, c))
            else:
                for c in range(c2, c1):
                    path.append((r2, c))
            if any((rr, cc) in used for rr, cc in path):
                ok = False
            if any(r < 0 or c < 0 or r >= H or c >= W for r, c in path):
                ok = False
            if ok:
                return ( (r1, c1), path )
        return ((0, 0), [])

    for _ in range(n_paths):
        (s, path) = gen_l_path()
        if len(path) < 3:
            continue
        paint_path(path)

        # choose endpoint tokens/fragments
        ends = []
        for p in [path[0], path[-1]]:
            if p not in ends:
                ends.append(p)
        if len(ends) < 2:
            continue
        random.shuffle(ends)
        token_end = ends[0]
        frag_end = ends[1]

        # place source token 3x3 near token_end
        src_color = random.choice(colors_tokens)
        token_shape = random.choice([SOLID, X_SHAPE, DIAMOND])
        # choose token placement by trying 4 adjacent offsets
        opts = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        placed = False
        for dr, dc in random.sample(opts, len(opts)):
            tl = (token_end[0] + dr - 1, token_end[1] + dc - 1)
            r0, c0 = tl
            if r0 < 1 or c0 < 1 or r0 + 3 > H - 1 or c0 + 3 > W - 1:
                continue
            area = g[r0 : r0 + 3, c0 : c0 + 3]
            if np.any(area != 0):
                continue
            for tr, tc in token_shape:
                g[r0 + tr, c0 + tc] = src_color
            placed = True
            break
        if not placed:
            continue

        # place fragment near frag_end
        target_shape = CYCLE[_shape_id(token_shape)]
        target_cells = list(target_shape)
        k = random.randint(1, 4)
        random.shuffle(target_cells)
        frag_cells = set(target_cells[:k])
        frag_color = random.choice(colors_frags)
        # put fragment top-left near frag_end
        for dr, dc in random.sample([( -1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)], 8):
            tl = (frag_end[0] + dr - 1, frag_end[1] + dc - 1)
            r0, c0 = tl
            if r0 < 0 or c0 < 0 or r0 + 3 > H or c0 + 3 > W:
                continue
            # clear only if all cells background
            if np.any(g[r0 : r0 + 3, c0 : c0 + 3] != 0):
                continue
            for tr, tc in frag_cells:
                g[r0 + tr, c0 + tc] = frag_color
            break

    return g

