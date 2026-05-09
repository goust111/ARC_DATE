# Combo variant of 7b3084d4 (index 4)
# combo_spec: D2/3+D6/2
# Name: 含干扰块过滤的空洞拼图
#
# Design analysis for the fusion:
# a) Concretization:
#    - The original task has 4 scattered interlocking pieces and one marker cell.
#    - This combo keeps the same 2x2 jigsaw-style square assembly, but 2-3 of the
#      4 valid pieces now contain one real interior background cell each.
#    - The input has 5-6 components total, so 1-2 extra components must be ignored.
#    - The output is still the assembled square, but the interior holes remain 0.
# b) Complexity calibration:
#    - Hole count is limited to 2 or 3, which is visually obvious without making
#      the packing search too unconstrained.
#    - Distractors are deliberately much smaller than the real pieces, so they are
#      visible as scraps yet cannot replace a quadrant-sized piece.
# c) Implicit assumptions broken from the original solver:
#    - "One color == one piece" is no longer reliable enough, so pieces are
#      extracted by 8-connected components.
#    - "Total occupied area exactly equals output area" is false because the
#      assembled square contains preserved holes.
#    - "All components are useful" is false because 1-2 components are distractors.
# d) Placement legality:
#    - Generator scatters tight patches with an 8-neighborhood gap, so components
#      never touch or overlap in the input.
# e) Visual observability:
#    - Across samples, humans can see 5-6 scattered pieces, one grey-corner anchor,
#      and a final square that still has small black holes near the center.
# f) Visual cleanliness:
#    - Holes are only single cells and are chosen from bbox-interior locations, so
#      they read as genuine cavities rather than noisy edge erosion.
# g) Learnability:
#    - The rule stays structural, not color-coded: keep the marker-fixed anchor,
#      ignore the extra scraps, and assemble the four large interlocking pieces
#      while preserving their interior holes.

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
ANCHOR_COLOR = 7
MARKER_COLOR = 5
VALID_COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]
DISTRACTOR_COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]

CANVAS_H = 20
CANVAS_W = 20
MAX_ATTEMPTS = 500
N_MIN = 3
N_MAX = 7
MIN_HOLES = 2
MAX_HOLES = 3
PADDING = 1

TL, TR, BL, BR = 0, 1, 2, 3
TAB_NAMES = ["H_left", "H_right", "V_top", "V_bot"]


def _bg_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _normalize(cells):
    min_r = min(r for r, _, _ in cells)
    min_c = min(c for _, c, _ in cells)
    out = [(int(r - min_r), int(c - min_c), int(col)) for r, c, col in cells]
    out.sort()
    return out


def _rotate_cells(cells, k):
    pts = list(cells)
    for _ in range(k % 4):
        pts = [(c, -r, col) for r, c, col in pts]
    return _normalize(pts)


def _variants(cells):
    seen = set()
    outs = []
    for k in range(4):
        rot = _rotate_cells(cells, k)
        key = tuple(rot)
        if key not in seen:
            seen.add(key)
            outs.append(rot)
    return outs


def _bbox(cells):
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _is_contiguous(mask: np.ndarray, connectivity: int = 8) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    H, W = mask.shape
    neigh = _neighbors8 if connectivity == 8 else _neighbors4
    while q:
        r, c = q.popleft()
        for nr, nc in neigh(r, c):
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _has_enclosed_hole(mask: np.ndarray) -> bool:
    H, W = mask.shape
    outside = np.zeros((H, W), dtype=bool)
    q = deque()
    for r in range(H):
        for c in (0, W - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return bool((~mask & ~outside).any())


def _components(grid):
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    counts = Counter(map(int, g.ravel()))
    for sr, sc in np.argwhere(g != bg):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        abs_cells = []
        while q:
            r, c = q.popleft()
            abs_cells.append((int(r), int(c), int(g[r, c])))
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        local = _normalize(abs_cells)
        bbox = _bbox(abs_cells)
        holey = len(local) < (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        comps.append(
            {
                "cells": local,
                "abs": abs_cells,
                "bbox": bbox,
                "area": len(local),
                "holey": holey,
            }
        )
    return comps, bg, counts


def _marker_point(cells, global_counts):
    pts = [(r, c) for r, c, col in cells if col == MARKER_COLOR and global_counts.get(MARKER_COLOR, 0) == 1]
    if len(pts) == 1:
        return pts[0]
    return None


def _anchor_index(comps, global_counts):
    for i, comp in enumerate(comps):
        if _marker_point(comp["cells"], global_counts) is not None:
            return i
    return None


def _piece_hw(cells):
    max_r = max(r for r, _, _ in cells)
    max_c = max(c for _, c, _ in cells)
    return int(max_r + 1), int(max_c + 1)


def _candidate_sides(area, comps):
    min_side = max(
        int(np.ceil(area ** 0.5)),
        max(max(_piece_hw(comp["cells"])) for comp in comps),
    )
    sides = []
    for S in range(min_side, 2 * N_MAX + 1):
        holes = S * S - area
        if MIN_HOLES <= holes <= MAX_HOLES:
            sides.append(S)
    return sides


def _placeable(board, placement):
    H, W = board.shape
    for r, c, _ in placement:
        if not (0 <= r < H and 0 <= c < W) or board[r, c] != -1:
            return False
    return True


def _apply(board, owner, placement, piece_id, erase=False):
    if erase:
        for r, c, _ in placement:
            board[r, c] = -1
            owner[r, c] = -1
        return
    for r, c, col in placement:
        board[r, c] = col
        owner[r, c] = piece_id


def _all_placements(var, S):
    ph, pw = _piece_hw(var)
    out = []
    for tr in range(S - ph + 1):
        for tc in range(S - pw + 1):
            out.append([(r + tr, c + tc, col) for r, c, col in var])
    return out


def _touches_existing(owner, placement):
    if np.all(owner == -1):
        return True
    H, W = owner.shape
    for r, c, _ in placement:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and owner[nr, nc] != -1:
                return True
    return False


def _occupied_connected(board):
    occ = board != -1
    return _is_contiguous(occ, connectivity=4)


def _piece_graph_ok(owner, n_pieces):
    adj = [set() for _ in range(n_pieces)]
    H, W = owner.shape
    for r in range(H):
        for c in range(W):
            a = int(owner[r, c])
            if a == -1:
                continue
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if 0 <= nr < H and 0 <= nc < W:
                    b = int(owner[nr, nc])
                    if b != -1 and b != a:
                        adj[a].add(b)
                        adj[b].add(a)
    if any(len(nei) == 0 for nei in adj):
        return False
    seen = {0}
    q = deque([0])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == n_pieces


def _final_ok(board, owner, n_pieces):
    occ = board != -1
    if not occ[0, :].all() or not occ[-1, :].all() or not occ[:, 0].all() or not occ[:, -1].all():
        return False
    hole_count = int((~occ).sum())
    if not (MIN_HOLES <= hole_count <= MAX_HOLES):
        return False
    if not _occupied_connected(board):
        return False
    return _piece_graph_ok(owner, n_pieces)


def _solve_subset_square(comps, global_counts, S):
    board = np.full((S, S), -1, dtype=int)
    owner = np.full((S, S), -1, dtype=int)
    anchor = _anchor_index(comps, global_counts)
    assert anchor is not None

    piece_vars = [_variants(comp["cells"]) for comp in comps]
    placement_cache = []
    for vars_for_piece in piece_vars:
        per_piece = []
        for var in vars_for_piece:
            per_piece.append(_all_placements(var, S))
        placement_cache.append(per_piece)

    remaining = [i for i in range(len(comps)) if i != anchor]
    anchor_piece_id = 0

    def backtrack(rem, piece_id_map):
        if not rem:
            return _final_ok(board, owner, len(comps))

        best_pi = None
        best_opts = None
        for pi in rem:
            opts = []
            for vi, placements in enumerate(placement_cache[pi]):
                for placement in placements:
                    if not _placeable(board, placement):
                        continue
                    if not _touches_existing(owner, placement):
                        continue
                    opts.append((vi, placement))
            if not opts:
                return False
            if best_opts is None or len(opts) < len(best_opts):
                best_pi = pi
                best_opts = opts
        next_rem = [x for x in rem if x != best_pi]
        next_piece_id = len(piece_id_map)
        for _, placement in best_opts:
            _apply(board, owner, placement, next_piece_id)
            piece_id_map.append(best_pi)
            if backtrack(next_rem, piece_id_map):
                return True
            piece_id_map.pop()
            _apply(board, owner, placement, next_piece_id, erase=True)
        return False

    for var in piece_vars[anchor]:
        marker = _marker_point(var, global_counts)
        if marker is None:
            continue
        mr, mc = marker
        placement = [(r - mr, c - mc, col) for r, c, col in var]
        if not _placeable(board, placement):
            continue
        _apply(board, owner, placement, anchor_piece_id)
        if backtrack(remaining, [anchor]):
            return np.where(board == -1, BG, board).astype(int)
        _apply(board, owner, placement, anchor_piece_id, erase=True)
    return None


def _solve_packing(comps, global_counts):
    anchor = _anchor_index(comps, global_counts)
    assert anchor is not None, "input must contain exactly one anchor component with the marker"
    if len(comps) < 4:
        raise AssertionError("input must contain at least four separated pieces")

    combos = []
    for idxs in itertools.combinations(range(len(comps)), 4):
        if anchor not in idxs:
            continue
        subset = [comps[i] for i in idxs]
        combos.append((sum(comp["area"] for comp in subset), idxs, subset))
    combos.sort(key=lambda item: (-item[0], item[1]))

    for _, _idxs, subset in combos:
        area = sum(comp["area"] for comp in subset)
        for S in _candidate_sides(area, subset):
            solved = _solve_subset_square(subset, global_counts, S)
            if solved is not None:
                return solved
    raise AssertionError("no 4-piece square assembly with preserved holes was found")


def main(grid):
    """
    Assemble the four real puzzle pieces into a square while keeping their interior cavities visible.

    Input structure:
    - bg is the most common color. The grid contains 5-6 separated 8-connected pieces.
    - Exactly one piece is the anchor: it contains the unique grey marker pixel at one corner.
    - The four valid pieces are large interlocking shapes; 2-3 of them have a true interior hole.
    - The remaining 1-2 pieces are distractors that do not belong to the final square.

    Rule:
    - Extract connected components, identify the anchor by the unique marker, and test 4-piece subsets that include it.
    - Rotate pieces in 90-degree steps and place them into a single SxS square with the marker fixed at output position (0,0).
    - Keep only assemblies whose outer border is fully occupied and whose interior still contains the expected preserved holes.
    - Output that assembled square, leaving the cavity cells as background 0.

    Key insight:
    - The extra scraps never participate in any legal square assembly, while the real four pieces uniquely form a border-complete square with 2-3 internal holes and the marker pinned to the top-left.
    """
    comps, _bg, global_counts = _components(np.asarray(grid, dtype=int))
    assert comps, "input must contain non-background pieces"
    return _solve_packing(comps, global_counts)


def _build_ownership(N, p, tab_specs):
    S = 2 * N
    own = np.zeros((S, S), dtype=int)
    own[:N, :N] = TL
    own[:N, N:] = TR
    own[N:, :N] = BL
    own[N:, N:] = BR

    for kind, col, up_owner, down_owner in [
        ("H_left", p, BL, TL),
        ("H_right", S - 1 - p, BR, TR),
    ]:
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == "up":
            own[N - depth : N, col] = up_owner
        else:
            own[N : N + depth, col] = down_owner

    for kind, row, left_owner, right_owner in [
        ("V_top", p, TR, TL),
        ("V_bot", S - 1 - p, BR, BL),
    ]:
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == "left":
            own[row, N - depth : N] = left_owner
        else:
            own[row, N : N + depth] = right_owner

    for q in (TL, TR, BL, BR):
        mask = own == q
        if mask.sum() == 0 or not _is_contiguous(mask, connectivity=8):
            return None
    return own


def _random_tab_specs(N):
    specs = {}
    for name in TAB_NAMES:
        depth = random.randint(1, N - 1)
        direction = random.choice(["up", "down"] if name.startswith("H") else ["left", "right"])
        specs[name] = (depth, direction)
    return specs


def _pick_hole_cell(own, q, N, forbidden):
    coords = np.argwhere(own == q)
    if len(coords) == 0:
        return None
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    targets = {
        TL: (N - 1, N - 1),
        TR: (N - 1, N),
        BL: (N, N - 1),
        BR: (N, N),
    }
    target = targets[q]
    candidates = []
    for r, c in coords:
        r, c = int(r), int(c)
        if (r, c) in forbidden:
            continue
        if not (min_r < r < max_r and min_c < c < max_c):
            continue
        candidates.append((abs(r - target[0]) + abs(c - target[1]), r, c))
    candidates.sort()

    for _, r, c in candidates:
        trial = own.copy()
        trial[r, c] = -1
        mask = trial == q
        if mask.sum() == 0:
            continue
        if not _is_contiguous(mask, connectivity=8):
            continue
        rows, cols = np.where(mask)
        local = np.zeros((rows.max() - rows.min() + 1, cols.max() - cols.min() + 1), dtype=bool)
        local[rows - rows.min(), cols - cols.min()] = True
        if _has_enclosed_hole(local):
            return (r, c)
    return None


def _carve_holes(own, N):
    hole_count = random.choice([2, 3])
    order = [TL, TR, BL, BR]
    random.shuffle(order)
    selected = order[:hole_count]
    forbidden = {(0, 0)}
    holes = []
    carved = own.copy()
    for q in selected:
        cell = _pick_hole_cell(carved, q, N, forbidden)
        if cell is None:
            return None
        r, c = cell
        carved[r, c] = -1
        holes.append((r, c))
    for q in (TL, TR, BL, BR):
        if not _is_contiguous(carved == q, connectivity=8):
            return None
    return carved, holes


def _render_output(own, colors):
    out = np.zeros(own.shape, dtype=int)
    for q, col in colors.items():
        out[own == q] = int(col)
    out[0, 0] = MARKER_COLOR
    return out


def _extract_piece_cells(own, q, colors):
    coords = np.argwhere(own == q)
    min_r, min_c = coords.min(axis=0)
    cells = []
    for r, c in coords:
        r, c = int(r), int(c)
        col = MARKER_COLOR if (r, c) == (0, 0) else int(colors[q])
        cells.append((r - int(min_r), c - int(min_c), col))
    return _normalize(cells)


def _cells_to_patch(cells):
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = int(col)
    return patch


def _mask_can_place(occupied, mask, r0, c0):
    H, W = occupied.shape
    for pr, pc in np.argwhere(mask):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        if occupied[rr, cc]:
            return False
        rr0 = max(0, rr - PADDING)
        rr1 = min(H, rr + PADDING + 1)
        cc0 = max(0, cc - PADDING)
        cc1 = min(W, cc + PADDING + 1)
        if occupied[rr0:rr1, cc0:cc1].any():
            return False
    return True


def _scatter_patches(patches):
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    occupied = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)

    for idx in order:
        patch = patches[idx]
        mask = patch != 0
        ph, pw = patch.shape
        placed = False
        for _ in range(300):
            r0 = random.randint(0, CANVAS_H - ph)
            c0 = random.randint(0, CANVAS_W - pw)
            if not _mask_can_place(occupied, mask, r0, c0):
                continue
            region = canvas[r0 : r0 + ph, c0 : c0 + pw]
            region[mask] = patch[mask]
            occupied[r0 : r0 + ph, c0 : c0 + pw][mask] = True
            placed = True
            break
        if not placed:
            return None
    return canvas


def _make_distractor_patch(color):
    shapes = [
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 3),
    ]
    h, w = random.choice(shapes)
    if random.random() < 0.5:
        h, w = w, h
    return np.full((h, w), int(color), dtype=int)


def _gen_attempt():
    N = random.randint(N_MIN, N_MAX)
    p = random.randint(1, N - 1)

    own = _build_ownership(N, p, _random_tab_specs(N))
    if own is None:
        return None

    carved = _carve_holes(own, N)
    if carved is None:
        return None
    own, holes = carved
    if not (MIN_HOLES <= len(holes) <= MAX_HOLES):
        return None

    pool = VALID_COLOR_POOL.copy()
    random.shuffle(pool)
    colors = {TL: ANCHOR_COLOR, TR: pool[0], BL: pool[1], BR: pool[2]}
    expected = _render_output(own, colors)

    pieces = [_extract_piece_cells(own, q, colors) for q in (TL, TR, BL, BR)]
    pieces = [random.choice(_variants(piece)) for piece in pieces]

    used_colors = {ANCHOR_COLOR, MARKER_COLOR, colors[TR], colors[BL], colors[BR]}
    distractor_palette = [c for c in DISTRACTOR_COLOR_POOL if c not in used_colors]
    if len(distractor_palette) < 2:
        distractor_palette = DISTRACTOR_COLOR_POOL[:]
    random.shuffle(distractor_palette)
    distractor_count = random.choice([1, 2])
    distractors = [_make_distractor_patch(distractor_palette[i]) for i in range(distractor_count)]

    patches = [_cells_to_patch(piece) for piece in pieces] + distractors
    canvas = _scatter_patches(patches)
    if canvas is None:
        return None

    try:
        out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return canvas


def generate_input():
    for _ in range(MAX_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
