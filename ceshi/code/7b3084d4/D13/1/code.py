# Variant of 7b3084d4
# Dimension: D13 — 输出验证/对称性约束 (meta)
# Change: 输出具有旋转对称性
#
# Design notes:
# The original task scatters several colored puzzle pieces and asks for the
# tight assembled rectangle, with a marker-bearing anchor fixing orientation.
# This variant keeps that visible ARC structure but changes the generated
# evidence according to FEATURE below.  The solver extracts separated
# 8-connected pieces rather than assuming one color per piece, so it supports
# multi-color pieces, connector pixels, extra marker pixels, flipped pieces,
# and distractor filtering.  The generator is output-first: it builds a target
# tiling, extracts and transforms the pieces, scatters them, and retries until
# main() can solve the generated input.

import itertools
import random
from collections import Counter, deque

import numpy as np


FEATURE = 'rot_sym'
BG = 0
ANCHOR_COLOR = 7
MARKER_COLOR = 5
CANVAS_H = 20
CANVAS_W = 20
MAX_ATTEMPTS = 400
TL, TR, BL, BR = 0, 1, 2, 3


def _bg_color(g):
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors8(r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _is_contiguous(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _normalize(cells):
    min_r = min(r for r, _, _ in cells)
    min_c = min(c for _, c, _ in cells)
    out = [(r - min_r, c - min_c, col) for r, c, col in cells]
    out.sort()
    return out


def _rotate_cells(cells, k):
    pts = list(cells)
    for _ in range(k % 4):
        pts = [(c, -r, col) for r, c, col in pts]
    return _normalize(pts)


def _reflect_cells(cells):
    return _normalize([(r, -c, col) for r, c, col in cells])


def _variants(cells, mode="rot"):
    bases = [cells]
    if mode in ("flip", "reflect_anchor"):
        bases.append(_reflect_cells(cells))
    if mode == "fixed":
        rots = [0]
    else:
        rots = [0, 1, 2, 3]
    seen = set()
    outs = []
    for base in bases:
        for k in rots:
            v = _rotate_cells(base, k)
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                outs.append(v)
    return outs


def _components(grid):
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    h, w = g.shape
    seen = np.zeros(g.shape, dtype=bool)
    comps = []
    color_counts = Counter(map(int, g.ravel()))
    for sr, sc in np.argwhere(g != bg):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        abs_cells = []
        while q:
            r, c = q.popleft()
            abs_cells.append((r, c, int(g[r, c])))
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        local = _normalize(abs_cells)
        comps.append({"cells": local, "abs": abs_cells, "bbox": _bbox(abs_cells)})
    return comps, bg, color_counts


def _bbox(cells):
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _dims(area):
    out = []
    for h in range(1, int(area ** 0.5) + 1):
        if area % h == 0:
            w = area // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    out.sort(key=lambda t: (abs(t[0] - t[1]), t[0], t[1]))
    return out


def _marker_point(cells, color_counts):
    for wanted in (MARKER_COLOR, 6, 8, 9, 4, 3, 2, 1):
        pts = [(r, c) for r, c, col in cells if col == wanted and color_counts.get(col, 0) == 1]
        if pts:
            return pts[0], wanted
    singles = [(r, c, col) for r, c, col in cells if color_counts.get(col, 0) == 1]
    if singles:
        r, c, col = singles[0]
        return (r, c), col
    return None, None


def _anchor_index(comps, color_counts):
    for i, comp in enumerate(comps):
        if any(col == MARKER_COLOR for _, _, col in comp["cells"]):
            return i
    for i, comp in enumerate(comps):
        pt, _ = _marker_point(comp["cells"], color_counts)
        if pt is not None:
            return i
    return None


def _placeable(board, placement):
    h, w = board.shape
    for r, c, _ in placement:
        if not (0 <= r < h and 0 <= c < w) or board[r, c] != -1:
            return False
    return True


def _apply(board, placement, val=None):
    for r, c, col in placement:
        board[r, c] = col if val is None else val


def _all_placements(var, H, W):
    ph = max(r for r, _, _ in var) + 1
    pw = max(c for _, c, _ in var) + 1
    out = []
    for tr in range(H - ph + 1):
        for tc in range(W - pw + 1):
            out.append([(r + tr, c + tc, col) for r, c, col in var])
    return out


def _corner_target(input_comp, H, W):
    cells = input_comp["cells"]
    (mr, mc), _ = _marker_point(cells, Counter(col for _, _, col in cells))
    max_r = max(r for r, _, _ in cells)
    max_c = max(c for _, c, _ in cells)
    if mr == 0 and mc == 0:
        return (0, 0)
    if mr == 0 and mc == max_c:
        return (0, W - 1)
    if mr == max_r and mc == 0:
        return (H - 1, 0)
    return (H - 1, W - 1)


def _solve_subset(comps, color_counts, H, W, transform_mode, anchor_mode):
    board = np.full((H, W), -1, dtype=int)
    anchor = _anchor_index(comps, color_counts)
    modes = []
    for i, comp in enumerate(comps):
        if FEATURE == "partial_rotate" and i != anchor:
            area = len(comp["cells"])
            modes.append("rot" if area % 2 else "fixed")
        elif FEATURE == "reflect_anchor" and i == anchor:
            modes.append("flip")
        else:
            modes.append(transform_mode)
    piece_vars = [_variants(comp["cells"], modes[i]) for i, comp in enumerate(comps)]

    def backtrack(rem):
        if not rem:
            return True
        best_i = None
        best_opts = None
        for i in rem:
            opts = []
            for var in piece_vars[i]:
                for placement in _all_placements(var, H, W):
                    if _placeable(board, placement):
                        opts.append(placement)
            if not opts:
                return False
            if best_opts is None or len(opts) < len(best_opts):
                best_i, best_opts = i, opts
        next_rem = [x for x in rem if x != best_i]
        for placement in best_opts:
            _apply(board, placement)
            if backtrack(next_rem):
                return True
            _apply(board, placement, -1)
        return False

    if anchor is None or anchor_mode == "none":
        return board if backtrack(list(range(len(comps)))) else None

    targets = []
    if anchor_mode == "same_corner":
        targets = [_corner_target(comps[anchor], H, W)]
    else:
        targets = [(0, 0)]

    remaining = [i for i in range(len(comps)) if i != anchor]
    for var in piece_vars[anchor]:
        pt, _ = _marker_point(var, color_counts)
        if pt is None:
            continue
        mr, mc = pt
        for trgt_r, trgt_c in targets:
            placed = [(r + trgt_r - mr, c + trgt_c - mc, col) for r, c, col in var]
            if not _placeable(board, placed):
                continue
            _apply(board, placed)
            if backtrack(remaining):
                return board
            _apply(board, placed, -1)
    return None


def _solve_packing(comps, color_counts):
    if FEATURE == "packed_grid":
        return _packed_grid_output(comps)
    transform_mode = "flip" if FEATURE == "flip" else "rot"
    anchor_mode = "same_corner" if FEATURE == "same_corner" else "tl"
    if FEATURE == "cross5":
        anchor_mode = "none"
    subset_sizes = [len(comps)]
    if FEATURE == "distractors" and len(comps) > 4:
        subset_sizes = [4]
    for size in subset_sizes:
        combos = itertools.combinations(range(len(comps)), size)
        for idxs in combos:
            if FEATURE == "distractors":
                anchor = _anchor_index(comps, color_counts)
                if anchor is not None and anchor not in idxs:
                    continue
            subset = [comps[i] for i in idxs]
            area = sum(len(c["cells"]) for c in subset)
            for H, W in _dims(area):
                solved = _solve_subset(subset, color_counts, H, W, transform_mode, anchor_mode)
                if solved is not None:
                    return solved.astype(int)
    raise AssertionError("packing failed for all candidate dimensions")


def _packed_grid_output(comps):
    rendered = [_cells_to_patch(c["cells"]) for c in comps]
    order = sorted(range(len(comps)), key=lambda i: (0 if np.any(rendered[i] == MARKER_COLOR) else 1, comps[i]["bbox"][1], comps[i]["bbox"][0], -rendered[i].size))
    heights = [rendered[i].shape[0] for i in order]
    widths = [rendered[i].shape[1] for i in order]
    H = max(heights)
    W = sum(widths) + len(order) - 1
    out = np.zeros((H, W), dtype=int)
    c0 = 0
    for i in order:
        patch = rendered[i]
        mask = patch != 0
        out[:patch.shape[0], c0:c0 + patch.shape[1]][mask] = patch[mask]
        c0 += patch.shape[1] + 1
    return out


def main(grid):
    """Solve the variant by extracting separated pieces and assembling them."""
    comps, _bg, color_counts = _components(np.asarray(grid, dtype=int))
    assert comps, "input must contain at least one non-background piece"
    return _solve_packing(comps, color_counts)


def _build_tab_ownership(N, plain=False):
    S = 2 * N
    own = np.zeros((S, S), dtype=int)
    own[:N, :N] = TL
    own[:N, N:] = TR
    own[N:, :N] = BL
    own[N:, N:] = BR
    if plain:
        return own
    p = random.randint(1, N - 1)
    specs = {}
    for name in ("H_left", "H_right", "V_top", "V_bot"):
        specs[name] = (random.randint(1, N - 1), random.choice(["up", "down"] if name.startswith("H") else ["left", "right"]))
    if N == 2:
        specs[random.choice(list(specs))] = None
    for kind, col, up_owner, down_owner in [("H_left", p, BL, TL), ("H_right", S - 1 - p, BR, TR)]:
        spec = specs[kind]
        if spec is None:
            continue
        depth, direction = spec
        if direction == "up":
            own[N - depth:N, col] = up_owner
        else:
            own[N:N + depth, col] = down_owner
    for kind, row, left_owner, right_owner in [("V_top", p, TR, TL), ("V_bot", S - 1 - p, BR, BL)]:
        spec = specs[kind]
        if spec is None:
            continue
        depth, direction = spec
        if direction == "left":
            own[row, N - depth:N] = left_owner
        else:
            own[row, N:N + depth] = right_owner
    for q in (TL, TR, BL, BR):
        if not _is_contiguous(own == q):
            return _build_tab_ownership(N, plain=plain)
    return own


def _ownership_to_cells(own, colors, marker_corner=(0, 0)):
    output = np.zeros(own.shape, dtype=int)
    for q, col in colors.items():
        output[own == q] = col
    pieces = []
    for q in sorted(colors):
        coords = []
        rows, cols = np.where(own == q)
        min_r, min_c = int(rows.min()), int(cols.min())
        for r, c in zip(rows, cols):
            r, c = int(r), int(c)
            col = int(output[r, c])
            coords.append((r - min_r, c - min_c, col))
        pieces.append(coords)
    ar, ac = marker_corner
    output[ar, ac] = MARKER_COLOR
    pieces[0] = [(r, c, MARKER_COLOR if (r + min(rr for rr, _, _ in pieces[0]), c + min(cc for _, cc, _ in pieces[0])) == (0, 0) else col) for r, c, col in pieces[0]]
    return output, pieces


def _cells_to_patch(cells):
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _pieces_from_labeled_output(labels, color_for_label, marker_positions, cell_grid=None):
    output = np.zeros(labels.shape, dtype=int)
    pieces = []
    for label in sorted(color_for_label):
        coords_abs = np.argwhere(labels == label)
        min_r, min_c = coords_abs.min(axis=0)
        cells = []
        for r, c in coords_abs:
            r, c = int(r), int(c)
            col = int(cell_grid[r, c]) if cell_grid is not None else color_for_label[label]
            if (r, c) in marker_positions:
                col = marker_positions[(r, c)]
            output[r, c] = col
            cells.append((r - int(min_r), c - int(min_c), col))
        pieces.append(cells)
    return output, pieces


def _base_output_and_pieces():
    if FEATURE == "variable_count":
        n = random.randint(2, 4)
        count = random.choice([3, 5, 6])
        labels = np.zeros((2 * n, count * n), dtype=int)
        for i in range(count):
            labels[:, i * n:(i + 1) * n] = i
        colors = {i: [ANCHOR_COLOR, 1, 2, 3, 4, 6][i] for i in range(count)}
        return _pieces_from_labeled_output(labels, colors, {(0, 0): MARKER_COLOR})
    if FEATURE == "cross5":
        n = random.randint(2, 4)
        labels = np.zeros((3 * n, 3 * n), dtype=int)
        labels[:n, n:2*n] = 1
        labels[n:2*n, :n] = 2
        labels[n:2*n, n:2*n] = 0
        labels[n:2*n, 2*n:] = 3
        labels[2*n:, n:2*n] = 4
        colors = {0: ANCHOR_COLOR, 1: 1, 2: 2, 3: 3, 4: 4}
        return _pieces_from_labeled_output(labels, colors, {(n, n): MARKER_COLOR})
    n = random.randint(2, 7)
    plain = FEATURE in ("plain_split", "mirror_sym", "rot_sym")
    own = _build_tab_ownership(n, plain=plain)
    pool = [1, 2, 3, 4] if FEATURE == "marker_color" else [1, 2, 3, 4, 6, 8, 9]
    random.shuffle(pool)
    colors = {TL: ANCHOR_COLOR, TR: pool[0], BL: pool[1], BR: pool[2]}
    if FEATURE == "mirror_sym":
        colors = {TL: ANCHOR_COLOR, TR: ANCHOR_COLOR, BL: pool[0], BR: pool[0]}
    if FEATURE == "rot_sym":
        colors = {TL: ANCHOR_COLOR, TR: pool[0], BL: pool[0], BR: ANCHOR_COLOR}
    output = np.zeros(own.shape, dtype=int)
    if FEATURE == "cavities" and n >= 3:
        # Make a visible notch/cavity in the anchor region that is filled by
        # the right-hand piece through a one-cell-wide connected tendril.
        own[1, 1:n + 1] = TR
    for q, col in colors.items():
        output[own == q] = col
    if FEATURE in ("edge_stripes", "connectors"):
        mark_cols = [6, 8, 9]
        for r in range(output.shape[0]):
            for c in range(output.shape[1]):
                for dr, dc in ((1, 0), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if nr < output.shape[0] and nc < output.shape[1] and own[r, c] != own[nr, nc]:
                        if FEATURE == "edge_stripes" or (r + c) % 3 == 0:
                            col = mark_cols[(own[r, c] + own[nr, nc]) % len(mark_cols)]
                            output[r, c] = col
                            output[nr, nc] = col
    marker_positions = {(0, 0): MARKER_COLOR}
    if FEATURE == "same_corner":
        marker_positions = {random.choice([(0, 0), (0, output.shape[1]-1), (output.shape[0]-1, 0), (output.shape[0]-1, output.shape[1]-1)]): MARKER_COLOR}
    if FEATURE == "double_marker":
        marker_positions = {(0, 0): MARKER_COLOR, (0, min(output.shape[1] - 1, n - 1)): 6}
    if FEATURE == "multi_markers":
        marker_positions = {(0, 0): MARKER_COLOR, (0, n): 6, (n, 0): 8, (n, n): 9}
    if FEATURE == "edge_marker":
        marker_positions = {(0, max(0, n // 2)): MARKER_COLOR}
    return _pieces_from_labeled_output(own, colors, marker_positions, output)


def _apply_piece_features(pieces):
    out = [list(p) for p in pieces]
    if FEATURE == "multicolor":
        palette = [1, 2, 3, 4, 6, 8, 9]
        for i, cells in enumerate(out):
            for j, (r, c, col) in enumerate(cells):
                if col != MARKER_COLOR:
                    cells[j] = (r, c, palette[(r + 2 * c + i) % len(palette)])
    if FEATURE == "marker_color":
        marker_col = random.choice([6, 8, 9])
        out = [[(r, c, 1 if col == marker_col else col) for r, c, col in cells] for cells in out]
        out[0] = [(r, c, marker_col if col == MARKER_COLOR else col) for r, c, col in out[0]]
    return out


def _transform_for_input(pieces):
    transformed = []
    for i, cells in enumerate(pieces):
        mode = "rot"
        if FEATURE == "flip" or (FEATURE == "reflect_anchor" and i == 0):
            mode = "flip"
        if FEATURE == "partial_rotate" and i != 0 and len(cells) % 2 == 0:
            mode = "fixed"
        vars_ = _variants(cells, mode)
        transformed.append(random.choice(vars_))
    return transformed


def _scatter(patches, ordered=False, overlap_boxes=False):
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    if ordered:
        r, c = 1, 1
        for patch in patches:
            if c + patch.shape[1] >= CANVAS_W:
                r += max(p.shape[0] for p in patches) + 2
                c = 1
            if r + patch.shape[0] >= CANVAS_H:
                return None
            mask = patch != 0
            region = canvas[r:r+patch.shape[0], c:c+patch.shape[1]]
            if np.any(region[mask] != 0):
                return None
            region[mask] = patch[mask]
            c += patch.shape[1] + 2
        return canvas
    order = list(range(len(patches)))
    random.shuffle(order)
    occupied = np.zeros_like(canvas, dtype=bool)
    for idx in order:
        patch = patches[idx]
        mask = patch != 0
        placed = False
        for _ in range(200):
            r = random.randint(0, CANVAS_H - patch.shape[0])
            c = random.randint(0, CANVAS_W - patch.shape[1])
            area = occupied[r:r+patch.shape[0], c:c+patch.shape[1]]
            if np.any(area & mask):
                continue
            if not overlap_boxes:
                rr0 = max(0, r - 1); cc0 = max(0, c - 1)
                rr1 = min(CANVAS_H, r + patch.shape[0] + 1); cc1 = min(CANVAS_W, c + patch.shape[1] + 1)
                if np.any(occupied[rr0:rr1, cc0:cc1]):
                    continue
            region = canvas[r:r+patch.shape[0], c:c+patch.shape[1]]
            region[mask] = patch[mask]
            occupied[r:r+patch.shape[0], c:c+patch.shape[1]][mask] = True
            placed = True
            break
        if not placed:
            return None
    return canvas


def _distractor_patches():
    patches = []
    for _ in range(random.choice([1, 2])):
        h, w = random.randint(1, 3), random.randint(2, 4)
        col = random.choice([1, 2, 3, 4, 6, 8, 9])
        p = np.full((h, w), col, dtype=int)
        if h * w > 2:
            p[random.randrange(h), random.randrange(w)] = 0
        patches.append(p)
    return patches


def _gen_attempt():
    _expected, pieces = _base_output_and_pieces()
    pieces = _apply_piece_features(pieces)
    pieces = _transform_for_input(pieces)
    patches = [_cells_to_patch(c) for c in pieces]
    if FEATURE == "distractors":
        patches += _distractor_patches()
    canvas = _scatter(
        patches,
        ordered=(FEATURE in ("ordered_layout", "area_order")),
        overlap_boxes=(FEATURE == "overlap_boxes"),
    )
    if canvas is None:
        return None
    try:
        main(canvas)
    except Exception:
        return None
    return canvas


def generate_input():
    for _ in range(MAX_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
