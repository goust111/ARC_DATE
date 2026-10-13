import random
from collections import defaultdict, deque

import numpy as np


BG = 0
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 120


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_shape(bbox):
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _bbox_area(bbox):
    h, w = _bbox_shape(bbox)
    return h * w


def _find_components_by_color(grid, bg=BG):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    by_color = defaultdict(list)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue
            visited[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            by_color[color].append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": bbox,
                }
            )
    return dict(by_color)


def _intervals_overlap(a0, a1, b0, b1):
    return max(a0, b0) <= min(a1, b1)


def _pair_axis(comp_a, comp_b):
    a_r0, a_c0, a_r1, a_c1 = comp_a["bbox"]
    b_r0, b_c0, b_r1, b_c1 = comp_b["bbox"]
    row_overlap = _intervals_overlap(a_r0, a_r1, b_r0, b_r1)
    col_overlap = _intervals_overlap(a_c0, a_c1, b_c0, b_c1)
    if row_overlap and not col_overlap:
        return "h"
    if col_overlap and not row_overlap:
        return "v"
    if row_overlap and col_overlap:
        row_gap = abs((a_r0 + a_r1) - (b_r0 + b_r1))
        col_gap = abs((a_c0 + a_c1) - (b_c0 + b_c1))
        return "h" if col_gap >= row_gap else "v"
    row_center_gap = abs((a_r0 + a_r1) - (b_r0 + b_r1))
    col_center_gap = abs((a_c0 + a_c1) - (b_c0 + b_c1))
    return "h" if col_center_gap >= row_center_gap else "v"


def _fixed_and_moving(comp_a, comp_b):
    key_a = (_bbox_area(comp_a["bbox"]), len(comp_a["coords"]), comp_a["bbox"])
    key_b = (_bbox_area(comp_b["bbox"]), len(comp_b["coords"]), comp_b["bbox"])
    if key_a > key_b:
        return comp_a, comp_b
    if key_b > key_a:
        return comp_b, comp_a
    return (comp_a, comp_b) if comp_a["bbox"] <= comp_b["bbox"] else (comp_b, comp_a)


def _shift_for_alignment(fixed_comp, moving_comp, axis):
    fr0, fc0, fr1, fc1 = fixed_comp["bbox"]
    mr0, mc0, mr1, mc1 = moving_comp["bbox"]
    fh, fw = _bbox_shape((fr0, fc0, fr1, fc1))
    mh, mw = _bbox_shape((mr0, mc0, mr1, mc1))
    if axis == "h":
        target_c0 = fc0 + (fw - mw) // 2
        return 0, target_c0 - mc0
    target_r0 = fr0 + (fh - mh) // 2
    return target_r0 - mr0, 0


def _scan_rows_candidate(grid, top, left, bg):
    h, w = grid.shape
    if w < 2:
        return None
    cols = (0, 1) if left else (w - 2, w - 1)
    row_iter = range(0, h) if top else range(h - 1, -1, -1)
    rows = []
    for r in row_iter:
        if int(grid[r, cols[0]]) != bg and int(grid[r, cols[1]]) != bg:
            rows.append(r)
        else:
            break
    if not rows:
        return None
    rows = sorted(rows)
    left_col, right_col = cols
    mapping = {}
    mask = np.zeros_like(grid, dtype=bool)
    for r in rows:
        src = int(grid[r, left_col])
        dst = int(grid[r, right_col])
        if src == bg or dst == bg or src in mapping:
            return None
        mapping[src] = dst
        mask[r, left_col] = True
        mask[r, right_col] = True
    return {"mapping": mapping, "mask": mask}


def _scan_cols_candidate(grid, top, left, bg):
    h, w = grid.shape
    if h < 2:
        return None
    rows = (0, 1) if top else (h - 2, h - 1)
    col_iter = range(0, w) if left else range(w - 1, -1, -1)
    cols = []
    for c in col_iter:
        if int(grid[rows[0], c]) != bg and int(grid[rows[1], c]) != bg:
            cols.append(c)
        else:
            break
    if not cols:
        return None
    cols = sorted(cols)
    upper_row, lower_row = sorted(rows)
    mapping = {}
    mask = np.zeros_like(grid, dtype=bool)
    for c in cols:
        src = int(grid[upper_row, c])
        dst = int(grid[lower_row, c])
        if src == bg or dst == bg or src in mapping:
            return None
        mapping[src] = dst
        mask[upper_row, c] = True
        mask[lower_row, c] = True
    return {"mapping": mapping, "mask": mask}


def _palette_score(grid, candidate, bg):
    work = np.array(grid, copy=True)
    work[candidate["mask"]] = bg
    by_color = _find_components_by_color(work, bg)
    outside_colors = set(by_color)
    palette_colors = set(candidate["mapping"])
    exact = outside_colors == palette_colors and all(len(comps) == 2 for comps in by_color.values())
    matched = sum(1 for color, comps in by_color.items() if color in palette_colors and len(comps) == 2)
    return (1 if exact else 0, matched, len(palette_colors), -int(candidate["mask"].sum()))


def _extract_palette(grid, bg=BG):
    g = np.asarray(grid, dtype=int)
    corners = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
    candidates = []
    for top, left in corners:
        corner_r = 0 if top else g.shape[0] - 1
        corner_c = 0 if left else g.shape[1] - 1
        if int(g[corner_r, corner_c]) == bg:
            continue
        for builder in (_scan_rows_candidate, _scan_cols_candidate):
            candidate = builder(g, top, left, bg)
            if candidate is not None and candidate["mapping"]:
                candidates.append(candidate)
    assert candidates, "no palette candidate found"
    best = max(candidates, key=lambda cand: _palette_score(g, cand, bg))
    return best["mapping"], best["mask"]


def _apply_pair(out, fixed_comp, moving_comp, mix_color, axis):
    color = fixed_comp["color"]
    dr, dc = _shift_for_alignment(fixed_comp, moving_comp, axis)
    moving_coords = moving_comp["coords"]
    out[moving_coords[:, 0], moving_coords[:, 1]] = BG
    fixed_cells = {tuple(map(int, cell)) for cell in fixed_comp["coords"]}
    for r, c in moving_coords:
        rr = int(r) + dr
        cc = int(c) + dc
        if (rr, cc) in fixed_cells:
            out[rr, cc] = mix_color
        else:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, slide the smaller fragment onto the larger one, and recolor only their overlap using a corner palette.

    Input structure:
    - Background is black. Several monochrome sprites are scattered in the grid, and every non-black input color appears in exactly two disconnected sprites.
    - A compact palette sits in one corner as either an N×2 block of horizontal color pairs or a 2×N block of vertical color pairs.
    - In each palette pair, the left or upper color is the sprite-pair color, and the right or lower color is the overlap color to use after stacking.
    - The two sprites of a given color are aligned either horizontally or vertically, so their row spans or column spans overlap.

    Rule:
    - Detect and remove the corner palette, then group the remaining sprites by color.
    - For each color pair, keep the larger sprite fixed and translate the smaller sprite only along the shared alignment axis until its bounding box is centered within the larger sprite's bounding box.
    - Cells occupied by both sprites after the move are recolored with the palette's mapped overlap color; cells occupied by only one sprite keep the original sprite color.
    - The moved sprite is erased from its source location, and the palette itself is erased to black in the final output.

    Key insight:
    - The task combines geometry and lookup: row/column alignment tells you which axis to slide along, while the corner palette supplies the third color used only on the post-translation overlap.
    """
    g = np.asarray(grid, dtype=int)
    mapping, palette_mask = _extract_palette(g, BG)
    work = np.array(g, copy=True)
    work[palette_mask] = BG
    by_color = _find_components_by_color(work, BG)
    out = np.array(work, copy=True)
    for color, comps in by_color.items():
        assert len(comps) == 2, f"expected exactly two sprites for color {color}"
        assert color in mapping, f"missing palette mapping for color {color}"
        axis = _pair_axis(comps[0], comps[1])
        fixed_comp, moving_comp = _fixed_and_moving(comps[0], comps[1])
        _apply_pair(out, fixed_comp, moving_comp, mapping[color], axis)
    return out


def _rect_free(blocked, r0, c0, h, w, pad=1):
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    return not blocked[rr0:rr1, cc0:cc1].any()


def _mark_rect(blocked, r0, c0, h, w, pad=1):
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _manhattan_path(a, b, rng):
    r, c = a
    tr, tc = b
    cells = [(r, c)]
    if rng.random() < 0.5:
        while c != tc:
            c += 1 if tc > c else -1
            cells.append((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            cells.append((r, c))
    else:
        while r != tr:
            r += 1 if tr > r else -1
            cells.append((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            cells.append((r, c))
    return cells


def _random_connected_mask(rng, h, w):
    anchors = {
        (0, rng.randrange(w)),
        (h - 1, rng.randrange(w)),
        (rng.randrange(h), 0),
        (rng.randrange(h), w - 1),
    }
    order = list(anchors)
    rng.shuffle(order)
    cells = {order[0]}
    current = order[0]
    for nxt in order[1:]:
        path = _manhattan_path(current, nxt, rng)
        cells.update(path)
        current = nxt

    frontier = set()

    def add_frontier(cell):
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    for cell in list(cells):
        add_frontier(cell)

    max_target = max(len(cells), min(h * w, len(cells) + max(2, (h * w) // 3)))
    target = rng.randint(len(cells), max_target)
    while frontier and len(cells) < target:
        cell = rng.choice(tuple(frontier))
        frontier.discard(cell)
        if cell in cells:
            continue
        cells.add(cell)
        add_frontier(cell)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _paint_mask(grid, bbox, mask, color):
    r0, c0, _, _ = bbox
    rows, cols = np.where(mask)
    grid[r0 + rows, c0 + cols] = color


def _overlap_count(mask_big, mask_small, axis):
    if axis == "h":
        offset = (mask_big.shape[1] - mask_small.shape[1]) // 2
        region = mask_big[:, offset:offset + mask_small.shape[1]]
    else:
        offset = (mask_big.shape[0] - mask_small.shape[0]) // 2
        region = mask_big[offset:offset + mask_small.shape[0], :]
    return int(np.logical_and(region, mask_small).sum())


def _generate_pair_shapes(rng, axis):
    for _ in range(PAIR_ATTEMPTS):
        if axis == "h":
            h = rng.randint(2, 5)
            w_big = rng.randint(4, 7)
            small_options = [w for w in range(2, w_big) if (w_big - w) % 2 == 0]
            if not small_options:
                continue
            w_small = rng.choice(small_options)
            big_mask = _random_connected_mask(rng, h, w_big)
            small_mask = _random_connected_mask(rng, h, w_small)
        else:
            w = rng.randint(2, 5)
            h_big = rng.randint(4, 7)
            small_options = [hh for hh in range(2, h_big) if (h_big - hh) % 2 == 0]
            if not small_options:
                continue
            h_small = rng.choice(small_options)
            big_mask = _random_connected_mask(rng, h_big, w)
            small_mask = _random_connected_mask(rng, h_small, w)
        overlap = _overlap_count(big_mask, small_mask, axis)
        small_count = int(small_mask.sum())
        if 0 < overlap < small_count:
            return big_mask, small_mask
    raise RuntimeError("failed to generate non-degenerate pair shapes")


def _place_pair(grid, blocked, rng, color, axis):
    big_mask, small_mask = _generate_pair_shapes(rng, axis)
    bh, bw = big_mask.shape
    sh, sw = small_mask.shape
    for _ in range(PAIR_ATTEMPTS):
        gap = rng.randint(2, 5)
        if axis == "h":
            total_w = bw + sw + gap
            if total_w >= grid.shape[1] or bh >= grid.shape[0]:
                continue
            r0 = rng.randint(0, grid.shape[0] - bh)
            c_start = rng.randint(0, grid.shape[1] - total_w)
            move_on_left = rng.random() < 0.5
            if move_on_left:
                moving_bbox = (r0, c_start, r0 + sh - 1, c_start + sw - 1)
                fixed_bbox = (r0, c_start + sw + gap, r0 + bh - 1, c_start + sw + gap + bw - 1)
            else:
                fixed_bbox = (r0, c_start, r0 + bh - 1, c_start + bw - 1)
                moving_bbox = (r0, c_start + bw + gap, r0 + sh - 1, c_start + bw + gap + sw - 1)
        else:
            total_h = bh + sh + gap
            if total_h >= grid.shape[0] or bw >= grid.shape[1]:
                continue
            c0 = rng.randint(0, grid.shape[1] - bw)
            r_start = rng.randint(0, grid.shape[0] - total_h)
            move_on_top = rng.random() < 0.5
            if move_on_top:
                moving_bbox = (r_start, c0, r_start + sh - 1, c0 + sw - 1)
                fixed_bbox = (r_start + sh + gap, c0, r_start + sh + gap + bh - 1, c0 + bw - 1)
            else:
                fixed_bbox = (r_start, c0, r_start + bh - 1, c0 + bw - 1)
                moving_bbox = (r_start + bh + gap, c0, r_start + bh + gap + sh - 1, c0 + sw - 1)
        fr0, fc0, fr1, fc1 = fixed_bbox
        mr0, mc0, mr1, mc1 = moving_bbox
        if not _rect_free(blocked, fr0, fc0, fr1 - fr0 + 1, fc1 - fc0 + 1):
            continue
        if not _rect_free(blocked, mr0, mc0, mr1 - mr0 + 1, mc1 - mc0 + 1):
            continue
        _paint_mask(grid, fixed_bbox, big_mask, color)
        _paint_mask(grid, moving_bbox, small_mask, color)
        _mark_rect(blocked, fr0, fc0, fr1 - fr0 + 1, fc1 - fc0 + 1)
        _mark_rect(blocked, mr0, mc0, mr1 - mr0 + 1, mc1 - mc0 + 1)
        return
    raise RuntimeError("failed to place pair")


def _place_palette(grid, blocked, input_colors, mix_colors, corner, orient):
    n = len(input_colors)
    h, w = grid.shape
    mask = np.zeros_like(grid, dtype=bool)
    top = corner in ("tl", "tr")
    left = corner in ("tl", "bl")
    if orient == "rows":
        rows = list(range(n)) if top else list(range(h - n, h))
        cols = (0, 1) if left else (w - 2, w - 1)
        for idx, r in enumerate(rows):
            grid[r, cols[0]] = input_colors[idx]
            grid[r, cols[1]] = mix_colors[idx]
            mask[r, cols[0]] = True
            mask[r, cols[1]] = True
        r0, c0 = min(rows), min(cols)
        _mark_rect(blocked, r0, c0, n, 2)
    else:
        cols = list(range(n)) if left else list(range(w - n, w))
        rows = (0, 1) if top else (h - 2, h - 1)
        upper, lower = min(rows), max(rows)
        for idx, c in enumerate(cols):
            grid[upper, c] = input_colors[idx]
            grid[lower, c] = mix_colors[idx]
            mask[upper, c] = True
            mask[lower, c] = True
        r0, c0 = upper, min(cols)
        _mark_rect(blocked, r0, c0, 2, n)
    return mask


def _validate_generated(grid):
    try:
        mapping, pal_mask = _extract_palette(grid, BG)
        work = np.array(grid, copy=True)
        work[pal_mask] = BG
        by_color = _find_components_by_color(work, BG)
        if set(by_color) != set(mapping):
            return False
        for color, comps in by_color.items():
            if len(comps) != 2:
                return False
            fixed_comp, moving_comp = _fixed_and_moving(comps[0], comps[1])
            axis = _pair_axis(comps[0], comps[1])
            dr, dc = _shift_for_alignment(fixed_comp, moving_comp, axis)
            fixed_cells = {tuple(map(int, cell)) for cell in fixed_comp["coords"]}
            overlap = 0
            for r, c in moving_comp["coords"]:
                if (int(r) + dr, int(c) + dc) in fixed_cells:
                    overlap += 1
            if overlap == 0:
                return False
        out = main(grid)
        return out.shape == grid.shape and np.any(out != grid)
    except AssertionError:
        return False


def _gen_attempt(rng):
    num_pairs = rng.randint(2, 4)
    h = rng.randint(max(14, 5 * num_pairs + 4), min(28, 5 * num_pairs + 10))
    w = rng.randint(max(14, 5 * num_pairs + 4), min(28, 5 * num_pairs + 10))
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    colors = list(range(1, 10))
    rng.shuffle(colors)
    input_colors = colors[:num_pairs]
    mix_colors = colors[num_pairs:2 * num_pairs]
    rng.shuffle(input_colors)
    rng.shuffle(mix_colors)

    corner = rng.choice(("tl", "tr", "bl", "br"))
    orient = rng.choice(("rows", "cols"))
    _place_palette(grid, blocked, input_colors, mix_colors, corner, orient)

    axes = ["h", "v"] * ((num_pairs + 1) // 2)
    axes = axes[:num_pairs]
    rng.shuffle(axes)
    for color, axis in zip(input_colors, axes):
        _place_pair(grid, blocked, rng, color, axis)

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except RuntimeError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
