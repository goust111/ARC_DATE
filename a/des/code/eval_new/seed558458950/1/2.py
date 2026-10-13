import random
from collections import deque
import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _connected_components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or visited[r0, c0]:
                continue
            q = deque([(r0, c0)])
            visited[r0, c0] = True
            comp = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                if r > 0 and mask[r - 1, c] and not visited[r - 1, c]:
                    visited[r - 1, c] = True
                    q.append((r - 1, c))
                if r + 1 < h and mask[r + 1, c] and not visited[r + 1, c]:
                    visited[r + 1, c] = True
                    q.append((r + 1, c))
                if c > 0 and mask[r, c - 1] and not visited[r, c - 1]:
                    visited[r, c - 1] = True
                    q.append((r, c - 1))
                if c + 1 < w and mask[r, c + 1] and not visited[r, c + 1]:
                    visited[r, c + 1] = True
                    q.append((r, c + 1))
            comps.append(comp)
    return comps


def _touches_border(cells, h, w):
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _norm_shape(cells):
    arr = np.array(list(cells), dtype=int)
    if len(arr) == 0:
        return tuple()
    r0, c0 = int(arr[:, 0].min()), int(arr[:, 1].min())
    normed = [(int(r - r0), int(c - c0)) for r, c in arr]
    return tuple(sorted(normed))


def _rotate_coord(coord, k):
    r, c = coord
    k %= 4
    if k == 0:
        return (r, c)
    if k == 1:
        return (c, -r)
    if k == 2:
        return (-r, -c)
    return (-c, r)


def _shape_rotations(cells):
    arr = list(cells)
    rots = []
    for k in range(4):
        rots.append(_norm_shape([_rotate_coord(p, k) for p in arr]))
    return tuple(rots)


def _rotate_side(side, k):
    # 0 top, 1 right, 2 bottom, 3 left
    return (int(side) + int(k)) % 4


def _side_from_anchor(shape_set, anchor):
    ar, ac = anchor
    counts = [0, 0, 0, 0]
    for r, c in shape_set:
        if ar == r - 1 and ac == c:
            counts[0] += 1
        elif ar == r + 1 and ac == c:
            counts[2] += 1
        elif ar == r and ac == c - 1:
            counts[1] += 1
        elif ar == r and ac == c + 1:
            counts[3] += 1
    best = max(counts)
    if best == 0:
        return None
    for idx in (0, 2, 3, 1):
        if counts[idx] == best:
            return idx
    return None


def _infer_marker_color(grid, bg_color):
    colors = [int(v) for v in np.unique(grid)]
    candidates = []
    h, w = grid.shape

    for color in colors:
        if color == bg_color:
            continue
        mask = grid == color
        comps = _connected_components(mask)
        if not comps:
            continue
        if any(len(c) != 1 for c in comps):
            continue
        if len(comps) < 2:
            continue

        marker_like = 0
        for r, c in np.argwhere(mask):
            r, c = int(r), int(c)
            cnt = 0
            if r > 0 and grid[r - 1, c] != color:
                cnt += 1
            if r + 1 < h and grid[r + 1, c] != color:
                cnt += 1
            if c > 0 and grid[r, c - 1] != color:
                cnt += 1
            if c + 1 < w and grid[r, c + 1] != color:
                cnt += 1
            if cnt >= 2:
                marker_like += 1

        if marker_like >= len(comps):
            candidates.append((marker_like, len(comps), color))

    if not candidates:
        fallback = [c for c in colors if c != bg_color]
        if not fallback:
            return bg_color
        return int(fallback[0])

    candidates.sort(reverse=True)
    return int(candidates[0][2])


def _find_template_color(grid, bg_color, red_color, enclosed_bg):
    h, w = grid.shape
    colors = [int(c) for c in np.unique(grid) if c not in (bg_color, red_color)]
    scores = []

    for color in colors:
        comps = _connected_components(grid == color)
        if not comps:
            continue
        main_comp = max(comps, key=len)
        main_set = set(main_comp)
        if len(main_comp) < 3:
            continue

        touches = 0
        for hole in enclosed_bg:
            hs = set(hole)
            found = False
            for r, c in main_set:
                if found:
                    break
                if r > 0 and (r - 1, c) in hs:
                    found = True
                elif r + 1 < h and (r + 1, c) in hs:
                    found = True
                elif c > 0 and (r, c - 1) in hs:
                    found = True
                elif c + 1 < w and (r, c + 1) in hs:
                    found = True
            if found:
                touches += 1

        score = touches * 1000 + len(main_comp)
        scores.append((score, len(main_comp), color))

    if not scores:
        vals = [int(v) for v in np.unique(grid)]
        for v in vals:
            if v not in (bg_color, red_color):
                return v
        return bg_color
    scores.sort(reverse=True)
    return int(scores[0][2])


def _collect_holes(grid, bg_color, red_color):
    h, w = grid.shape
    holes = []
    enclosed = []
    for comp in _connected_components(grid == bg_color):
        if comp and not _touches_border(comp, h, w):
            enclosed.append(comp)

    for comp in enclosed:
        shape_set = set(comp)
        anchors = set()
        for r, c in comp:
            if r > 0 and grid[r - 1, c] == red_color:
                anchors.add((r - 1, c))
            if r + 1 < h and grid[r + 1, c] == red_color:
                anchors.add((r + 1, c))
            if c > 0 and grid[r, c - 1] == red_color:
                anchors.add((r, c - 1))
            if c + 1 < w and grid[r, c + 1] == red_color:
                anchors.add((r, c + 1))
        if not anchors:
            continue
        anchor = next(iter(anchors))
        side = _side_from_anchor(shape_set, anchor)
        if side is None:
            continue
        holes.append({
            "cells": comp,
            "sig": _norm_shape(comp),
            "side": side,
            "anchor": anchor,
        })
    return holes


def _collect_pieces(grid, bg_color, red_color, template_color):
    pieces = []
    h, w = grid.shape
    for color in [int(v) for v in np.unique(grid) if v not in (bg_color, red_color, template_color)]:
        for comp in _connected_components(grid == color):
            if not comp:
                continue
            shape_set = set(comp)
            anchors = set()
            for r, c in comp:
                if r > 0 and grid[r - 1, c] == red_color:
                    anchors.add((r - 1, c))
                if r + 1 < h and grid[r + 1, c] == red_color:
                    anchors.add((r + 1, c))
                if c > 0 and grid[r, c - 1] == red_color:
                    anchors.add((r, c - 1))
                if c + 1 < w and grid[r, c + 1] == red_color:
                    anchors.add((r, c + 1))
            if len(anchors) != 1:
                continue
            anchor = next(iter(anchors))
            side = _side_from_anchor(shape_set, anchor)
            if side is None:
                continue
            pieces.append(
                {
                    "color": color,
                    "cells": comp,
                    "anchor": anchor,
                    "side": side,
                    "sig": _norm_shape(comp),
                    "rots": _shape_rotations(shape_set),
                }
            )
    return pieces


def _find_assignment(holes, pieces):
    if not holes or not pieces:
        return {}

    h_to_p = []
    for hi, hole in enumerate(holes):
        options = []
        for pi, piece in enumerate(pieces):
            for k, psig in enumerate(piece['rots']):
                if psig == hole['sig'] and _rotate_side(piece['side'], k) == hole['side']:
                    options.append(pi)
                    break
        h_to_p.append((hi, options))

    h_to_p.sort(key=lambda x: len(x[1]))
    used = [False] * len(pieces)
    match = {}

    def dfs(pos):
        if pos == len(h_to_p):
            return True
        hi, opts = h_to_p[pos]
        if not opts:
            return False
        for pi in opts:
            if used[pi]:
                continue
            used[pi] = True
            match[hi] = pi
            if dfs(pos + 1):
                return True
            used[pi] = False
            match.pop(hi, None)
        return False

    if dfs(0):
        return match
    return {}


def main(grid: np.ndarray) -> np.ndarray:
    """Place scattered pieces into matching template holes after angle alignment via anchor side.
Input structure:
- bg is the dominant color.
- A large grey template is placed on the grid; its interior has several closed bg holes.
- Each hole has one red anchor pixel on a hole boundary at one of four sides.
- Outside the template are scattered monochrome fragments; each fragment has one red anchor.
- The fragment body (without red) matches a hole shape up to 0/90/180/270 rotation.
Rule:
- Detect enclosed bg holes, their red anchor pixels, and red anchor side on each hole boundary.
- Detect all candidate fragments by body shape and anchor side, and match each hole by shape under rotation.
- Use anchor-side consistency to pick the proper orientation so fragment and hole anchor lie on the same relative side.
- Fill each matched hole with the fragment color, keep the red anchor as red, and keep the template unchanged.
- Remove the original fragment cells and fragment anchors from their old positions.
Key insight:
- Hole identity comes from shape matching, while the anchor-side relation removes rotational ambiguity.
"""

    if grid.ndim != 2:
        return grid.copy()

    h, w = grid.shape
    out = np.array(grid, copy=True)
    bg = _most_common_color(grid)
    red = _infer_marker_color(grid, bg)

    holes = _collect_holes(grid, bg, red)
    if not holes:
        return out

    enclosed = [hcomp['cells'] for hcomp in holes]
    template = _find_template_color(grid, bg, red, enclosed)
    pieces = _collect_pieces(grid, bg, red, template)
    if not pieces:
        return out

    # remove fragment bodies and their old red anchors
    for p in pieces:
        for r, c in p['cells']:
            out[r, c] = bg
        ar, ac = p['anchor']
        if out[ar, ac] == red:
            out[ar, ac] = bg

    assignment = _find_assignment(holes, pieces)

    for hi, piece_idx in assignment.items():
        hole = holes[hi]
        piece = pieces[piece_idx]
        color = piece['color']
        for r, c in hole['cells']:
            out[r, c] = color
        ar, ac = hole['anchor']
        out[ar, ac] = red

    return out


# -------------------- generator helpers --------------------

def _random_shape(size, rng):
    cells = {(0, 0)}
    frontier = [(0, 0)]
    while len(cells) < size:
        r, c = frontier[rng.integers(0, len(frontier))]
        dr, dc = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        nr, nc = int(r + dr), int(c + dc)
        if (nr, nc) not in cells:
            cells.add((nr, nc))
            frontier.append((nr, nc))
    return cells


def _bounding_box(cells):
    arr = np.array(list(cells), dtype=int)
    return int(arr[:, 0].min()), int(arr[:, 0].max()), int(arr[:, 1].min()), int(arr[:, 1].max())


def _shape_cells_to_set(shape, rot, tx, ty):
    arr = set()
    for r, c in shape:
        rr, cc = _rotate_coord((r, c), rot)
        arr.add((rr + tx, cc + ty))
    r0 = min(r for r, _ in arr)
    c0 = min(c for _, c in arr)
    return {(r - r0, c - c0) for r, c in arr}, (r0, c0)


def _choose_anchor(shape, side):
    rs = [r for r, _ in shape]
    cs = [c for _, c in shape]
    top = min(rs)
    bottom = max(rs)
    left = min(cs)
    right = max(cs)

    if side == 0:
        cand = [(r, c) for r, c in shape if r == top]
        if not cand:
            return None
        r, c = random.choice(cand)
        return (r - 1, c)
    if side == 2:
        cand = [(r, c) for r, c in shape if r == bottom]
        if not cand:
            return None
        r, c = random.choice(cand)
        return (r + 1, c)
    if side == 3:
        cand = [(r, c) for r, c in shape if c == left]
        if not cand:
            return None
        r, c = random.choice(cand)
        return (r, c - 1)
    cand = [(r, c) for r, c in shape if c == right]
    if not cand:
        return None
    r, c = random.choice(cand)
    return (r, c + 1)


def _place_hole(grid, template_mask, shape, rng, side):
    h, w = grid.shape
    shape_set = set(shape)
    top, bottom, left, right = _bounding_box(shape_set)
    sh = bottom - top + 1
    sw = right - left + 1

    base_r_candidates = []
    base_c_candidates = []
    for r0 in range(h):
        for c0 in range(w):
            base_r_candidates.append(r0)
            base_c_candidates.append(c0)

    r_lim0, r_lim1 = int(np.min(np.where(template_mask)[0])) + 1, int(np.max(np.where(template_mask)[0])) - 1
    c_lim0, c_lim1 = int(np.min(np.where(template_mask)[1])) + 1, int(np.max(np.where(template_mask)[1])) - 1

    tries = 0
    while tries < 250:
        tries += 1
        tx = int(rng.integers(r_lim0, max(r_lim0 + 1, r_lim1 - sh + 2)))
        ty = int(rng.integers(c_lim0, max(c_lim0 + 1, c_lim1 - sw + 2)))
        translated = set()
        ok = True
        for r, c in shape_set:
            tr, tc = r - top + tx, c - left + ty
            if not (0 <= tr < h and 0 <= tc < w):
                ok = False
                break
            if not template_mask[tr, tc]:
                ok = False
                break
            translated.add((tr, tc))
        if not ok:
            continue
        anchor = _choose_anchor(shape_set, side)
        if anchor is None:
            continue
        ar, ac = anchor
        ar, ac = ar - top + tx, ac - left + ty

        if not (0 <= ar < h and 0 <= ac < w):
            continue
        if not template_mask[ar, ac]:
            continue
        if grid[ar, ac] != 1:
            continue
        for tr, tc in translated:
            if grid[tr, tc] != 1:
                ok = False
                break
            grid[tr, tc] = 0
        if not ok:
            continue
        grid[ar, ac] = 2
        return translated
    return None


def _place_piece(grid, available, shape, color, red):
    h, w = grid.shape
    shape_set = set(shape)
    rs = [r for r, _ in shape_set]
    cs = [c for _, c in shape_set]
    sh = max(rs) - min(rs) + 1
    sw = max(cs) - min(cs) + 1
    side = random.randrange(4)

    tries = 0
    while tries < 400:
        tries += 1
        tx = int(np.random.default_rng().integers(0, h - sh + 1))
        ty = int(np.random.default_rng().integers(0, w - sw + 1))
        translated = {(r - min(rs) + tx, c - min(cs) + ty) for r, c in shape_set}
        if any(not available[r, c] for r, c in translated):
            continue
        if any(grid[r, c] != 0 for r, c in translated):
            continue

        anchor = _choose_anchor(shape_set, side)
        if anchor is None:
            continue
        ar, ac = anchor
        ar, ac = ar - min(rs) + tx, ac - min(cs) + ty
        if not (0 <= ar < h and 0 <= ac < w):
            continue
        if not available[ar, ac] or grid[ar, ac] != 0:
            continue

        for rr, cc in translated:
            grid[rr, cc] = color
        grid[ar, ac] = red
        return translated
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    base_shapes = [
        {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)},
        {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)},
        {(0, 0), (1, 0), (2, 0), (2, 1), (1, 2), (0, 2)},
        {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (1, 2)},
        {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
        {(0, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)},
        {(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 1)},
        {(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 2)},
    ]

    def _candidates(shape, mask_allowed, cell_value, anchor_value, anchor_side):
        rows = [r for r, _ in shape]
        cols = [c for _, c in shape]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        h_shape = max_r - min_r + 1
        w_shape = max_c - min_c + 1

        anchors = _choose_anchor(shape, anchor_side)
        if anchors is None:
            return []

        options = []
        h, w = mask_allowed.shape
        ar_rel, ac_rel = anchors
        for tr in range(0, h - h_shape + 1):
            for tc in range(0, w - w_shape + 1):
                translated = {(r - min_r + tr, c - min_c + tc) for r, c in shape}
                ok = True
                for rr, cc in translated:
                    if not mask_allowed[rr, cc] or grid[rr, cc] != cell_value:
                        ok = False
                        break
                if not ok:
                    continue

                ar = ar_rel - min_r + tr
                ac = ac_rel - min_c + tc
                if not (0 <= ar < h and 0 <= ac < w):
                    continue
                if not mask_allowed[ar, ac] or grid[ar, ac] != anchor_value:
                    continue
                options.append((translated, (ar, ac)))

        return options

    for _ in range(250):
        H = int(rng.integers(20, 31))
        W = int(rng.integers(20, 31))

        bg = 0
        grey = 1
        red = 2

        template_h = int(rng.integers(10, max(11, H - 4)))
        template_w = int(rng.integers(10, max(11, W - 4)))
        if template_h > H - 3 or template_w > W - 3:
            continue

        top = int(rng.integers(2, H - template_h - 1))
        left = int(rng.integers(2, W - template_w - 1))
        bottom = top + template_h
        right = left + template_w

        grid = np.full((H, W), bg, dtype=np.int64)
        template_mask = np.zeros((H, W), dtype=bool)
        template_mask[top:bottom, left:right] = True
        grid[template_mask] = grey

        hole_count = int(rng.integers(2, 5))
        hole_templates = []
        selected_rots = set()

        for _ in range(hole_count):
            shape_pool = list(base_shapes)
            rng.shuffle(shape_pool)
            placed = False
            for shape in shape_pool:
                shape_rots = _shape_rotations(shape)
                if any(tuple(sorted(sig)) in selected_rots for sig in shape_rots):
                    continue

                side = int(rng.integers(0, 4))
                candidates = _candidates(shape, template_mask, grey, grey, side)
                # remove occupied template positions by checking current bg/hole markers
                if not candidates:
                    continue
                selected = random.choice(candidates)
                hole_cells, (ar, ac) = selected

                for r, c in hole_cells:
                    if grid[r, c] != grey:
                        break
                else:
                    for r, c in hole_cells:
                        grid[r, c] = bg
                    grid[ar, ac] = red
                    hole_templates.append(shape)
                    selected_rots.update(shape_rots)
                    placed = True
                    break

            if not placed:
                break

        if len(hole_templates) != hole_count:
            continue

        # finalize template object
        grid[template_mask] = grey

        outside = np.ones((H, W), dtype=bool)
        outside[max(0, top - 1):min(H, bottom + 2), max(0, left - 1):min(W, right + 2)] = False

        colors = list(range(3, 50))
        rng.shuffle(colors)
        piece_colors = colors[:hole_count]
        ok = True

        for i, shape in enumerate(hole_templates):
            piece_color = piece_colors[i]
            rot = int(rng.integers(0, 4))
            piece_shape = {_rotate_coord(p, rot) for p in shape}
            norm_piece = {(r - min(r for r, _ in piece_shape), c - min(c for _, c in piece_shape)) for r, c in piece_shape}
            side = int(rng.integers(0, 4))
            candidates = _candidates(norm_piece, outside, bg, bg, side)
            if not candidates:
                ok = False
                break
            piece_cells, (ar, ac) = random.choice(candidates)
            for r, c in piece_cells:
                grid[r, c] = piece_color
            grid[ar, ac] = red
            for r, c in piece_cells:
                outside[r, c] = False
            outside[ar, ac] = False

        if not ok:
            continue

        try:
            _ = main(grid)
        except Exception:
            continue

        return grid

    # deterministic fallback sample
    g = np.full((22, 22), 0, dtype=np.int64)
    g[4:18, 4:18] = 1
    g[8:11, 8] = 0
    g[8, 9] = 2
    g[11:13, 14:16] = 0
    g[12, 16] = 2
    g[2:5, 1:3] = 3
    g[1, 2] = 2
    g[2:5, 0:1] = 4
    g[1, 0] = 2
    return g
