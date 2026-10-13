import numpy as np


BG = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r, c, h, w):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _neighbors8(r, c, h, w):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def _mask_components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for nx, ny in _neighbors4(x, y, h, w):
                    if mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def _same_color_components(grid, mask):
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for nx, ny in _neighbors4(x, y, h, w):
                    if mask[nx, ny] and not seen[nx, ny] and int(grid[nx, ny]) == color:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            comps.append((color, comp))
    return comps


def _top_left_key(cells):
    return min(cells)


def _translation_candidates(piece_cells, hole_set):
    cells = list(piece_cells)
    base_r, base_c = cells[0]
    rel = [(r - base_r, c - base_c) for r, c in cells]
    candidates = []
    seen = set()
    for hr, hc in hole_set:
        tr, tc = hr - base_r, hc - base_c
        if (tr, tc) in seen:
            continue
        placed = {(base_r + tr + dr, base_c + tc + dc) for dr, dc in rel}
        if placed <= hole_set:
            seen.add((tr, tc))
            candidates.append(placed)
    return candidates


def _find_tiling(hole_cells, pieces):
    hole_set = set(hole_cells)
    items = []
    for idx, (color, cells) in enumerate(pieces):
        cands = _translation_candidates(cells, hole_set)
        if not cands:
            return None
        items.append((len(cands), idx, color, len(cells), cands))
    items.sort()

    def backtrack(pos, occupied, chosen):
        if pos == len(items):
            return chosen if occupied == hole_set else None
        _, idx, color, size, cands = items[pos]
        for placed in cands:
            if occupied.isdisjoint(placed):
                found = backtrack(pos + 1, occupied | placed, chosen + [(idx, color, size)])
                if found is not None:
                    return found
        return None

    return backtrack(0, set(), [])


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey-board hole with the majority color among its fitted fragments.

    Input structure:
    The background is the most common color.  A large grey board contains several
    background-colored holes, and colored marker pixels sit on board cells
    adjacent to those holes.  Outside the board are same-colored connected
    fragments; each marker color names the fragment that belongs to that hole.

    Rule:
    Find enclosed background components as holes, read their adjacent marker
    colors, remove the marker pixels, and collect the external colored fragments
    of those colors.  For each hole, try all translations of the named fragments
    without rotation or reflection until they exactly tile the hole without
    overlap.  Then recolor the whole hole to the color whose fitted fragment
    contributes the most cells, erase the external fragments, and restore marker
    cells to grey.

    Key insight:
    The markers only establish which pieces belong to which hole; the final hole
    color is a second-stage majority-by-area decision after the pieces are
    translated into the hole.
    """
    arr = np.asarray(grid)
    out = arr.copy()
    h, w = arr.shape
    values, counts = np.unique(arr, return_counts=True)
    bg = int(values[np.argmax(counts)])
    non_bg = values[values != bg]
    if len(non_bg) == 0:
        return out
    non_bg_counts = [int(np.sum(arr == v)) for v in non_bg]
    board_color = int(non_bg[int(np.argmax(non_bg_counts))])

    hole_id = -np.ones((h, w), dtype=int)
    holes = []
    for comp in _mask_components(arr == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        idx = len(holes)
        holes.append(comp)
        for r, c in comp:
            hole_id[r, c] = idx

    marker_mask = np.zeros((h, w), dtype=bool)
    hole_markers = [[] for _ in holes]
    for r in range(h):
        for c in range(w):
            color = int(arr[r, c])
            if color == bg or color == board_color:
                continue
            near_holes = sorted(
                {
                    int(hole_id[nr, nc])
                    for nr, nc in _neighbors8(r, c, h, w)
                    if hole_id[nr, nc] >= 0
                }
            )
            if len(near_holes) == 1:
                hid = near_holes[0]
                marker_mask[r, c] = True
                hole_markers[hid].append((r, c, color))

    piece_mask = (arr != bg) & (arr != board_color) & (~marker_mask)
    color_to_pieces = {}
    for color, comp in _same_color_components(arr, piece_mask):
        color_to_pieces.setdefault(color, []).append(comp)

    out[piece_mask] = bg
    out[marker_mask] = board_color

    for hid, hole in enumerate(holes):
        markers = sorted(hole_markers[hid], key=lambda item: (item[0], item[1]))
        if not markers:
            continue
        pieces = []
        used_components = set()
        for _, _, color in markers:
            comps = color_to_pieces.get(color, [])
            available = [
                (i, comp)
                for i, comp in enumerate(comps)
                if (color, i) not in used_components
            ]
            if not available:
                pieces = []
                break
            i, comp = min(available, key=lambda item: _top_left_key(item[1]))
            used_components.add((color, i))
            pieces.append((color, comp))
        if not pieces:
            continue

        tiling = _find_tiling(hole, pieces)
        if tiling is None and sum(len(cells) for _, cells in pieces) == len(hole):
            tiling = [(i, color, len(cells)) for i, (color, cells) in enumerate(pieces)]
        if tiling is None:
            continue

        area_by_color = {}
        first_marker_order = {color: i for i, (_, _, color) in enumerate(markers)}
        for _, color, size in tiling:
            area_by_color[color] = area_by_color.get(color, 0) + size
        best_color = max(
            area_by_color,
            key=lambda color: (area_by_color[color], -first_marker_order.get(color, 0)),
        )
        for r, c in hole:
            out[r, c] = best_color
    return out


def _make_edge_shape(rng, length, depth, parts):
    for _ in range(100):
        heights = rng.integers(1, depth + 1, size=length).tolist()
        heights[0] = max(heights[0], 1)
        heights[-1] = max(heights[-1], 1)
        if max(heights) == min(heights) and depth > 1:
            heights[int(rng.integers(0, length))] = depth
        cuts = sorted(rng.choice(np.arange(1, length), size=parts - 1, replace=False).tolist())
        bounds = [0] + cuts + [length]
        if any(bounds[i + 1] - bounds[i] < 1 for i in range(parts)):
            continue
        fragments = []
        sizes = []
        for a, b in zip(bounds, bounds[1:]):
            frag = [(d, x) for x in range(a, b) for d in range(heights[x])]
            fragments.append(frag)
            sizes.append(len(frag))
        if sizes.count(max(sizes)) == 1:
            return fragments
    return None


def _to_abs(side, pr, pc, ph, pw, start, d, x):
    if side == "top":
        return pr + 1 + d, pc + start + x
    if side == "bottom":
        return pr + ph - 2 - d, pc + start + x
    if side == "left":
        return pr + start + x, pc + 1 + d
    return pr + start + x, pc + pw - 2 - d


def _marker_cell(side, pr, pc, ph, pw, start, x):
    if side == "top":
        return pr, pc + start + x
    if side == "bottom":
        return pr + ph - 1, pc + start + x
    if side == "left":
        return pr + start + x, pc
    return pr + start + x, pc + pw - 1


def _outside_piece(cells, side, marker, pr, pc, ph, pw):
    mr, mc = marker
    if side == "top":
        edge = max(r for r, _ in cells)
        anchors = [(r, c) for r, c in cells if r == edge]
        ar, ac = min(anchors, key=lambda p: abs(p[1] - mc))
        tr, tc = pr - 1 - ar, mc - ac
    elif side == "bottom":
        edge = min(r for r, _ in cells)
        anchors = [(r, c) for r, c in cells if r == edge]
        ar, ac = min(anchors, key=lambda p: abs(p[1] - mc))
        tr, tc = pr + ph - ar, mc - ac
    elif side == "left":
        edge = max(c for _, c in cells)
        anchors = [(r, c) for r, c in cells if c == edge]
        ar, ac = min(anchors, key=lambda p: abs(p[0] - mr))
        tr, tc = mr - ar, pc - 1 - ac
    else:
        edge = min(c for _, c in cells)
        anchors = [(r, c) for r, c in cells if c == edge]
        ar, ac = min(anchors, key=lambda p: abs(p[0] - mr))
        tr, tc = mr - ar, pc + pw - ac
    return [(r + tr, c + tc) for r, c in cells]


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        height = int(rng.integers(22, 31))
        width = int(rng.integers(22, 31))
        top_margin = int(rng.integers(4, 7))
        bottom_margin = int(rng.integers(4, 7))
        left_margin = int(rng.integers(4, 7))
        right_margin = int(rng.integers(4, 7))
        ph = height - top_margin - bottom_margin
        pw = width - left_margin - right_margin
        if ph < 12 or pw < 12:
            continue

        side_capacity = {
            "top": min(top_margin, ph // 3),
            "bottom": min(bottom_margin, ph // 3),
            "left": min(left_margin, pw // 3),
            "right": min(right_margin, pw // 3),
        }
        usable_sides = [side for side, cap in side_capacity.items() if cap >= 3]
        if len(usable_sides) < 2:
            continue

        hole_count = int(rng.integers(2, min(4, len(usable_sides)) + 1))
        sides = rng.choice(usable_sides, size=hole_count, replace=False).tolist()
        part_counts = [int(rng.choice([2, 3])) for _ in range(hole_count)]
        if sum(part_counts) > len(COLORS):
            continue

        grid = np.zeros((height, width), dtype=int)
        pr, pc = top_margin, left_margin
        grid[pr:pr + ph, pc:pc + pw] = GREY
        colors = COLORS[:]
        rng.shuffle(colors)
        color_pos = 0
        reserved_holes = set()
        reserved_outside = set()
        ok = True

        for side, parts in zip(sides, part_counts):
            axis_len = pw - 4 if side in ("top", "bottom") else ph - 4
            if axis_len < max(5, parts + 2):
                ok = False
                break
            length = int(rng.integers(max(5, parts + 2), min(9, axis_len) + 1))
            depth = int(rng.integers(3, min(5, side_capacity[side]) + 1))
            start_low = 2
            start_high = axis_len - length + 3
            if start_high <= start_low:
                ok = False
                break
            start = int(rng.integers(start_low, start_high))
            fragments = _make_edge_shape(rng, length, depth, parts)
            if fragments is None:
                ok = False
                break

            candidate_hole = []
            piece_sets = []
            markers = []
            for frag in fragments:
                abs_cells = [_to_abs(side, pr, pc, ph, pw, start, d, x) for d, x in frag]
                candidate_hole.extend(abs_cells)
                edge_xs = sorted(x for d, x in frag if d == 0)
                mx = edge_xs[len(edge_xs) // 2]
                marker = _marker_cell(side, pr, pc, ph, pw, start, mx)
                markers.append(marker)
                piece_sets.append(abs_cells)

            expanded = set()
            for r, c in candidate_hole:
                for nr in range(r - 1, r + 2):
                    for nc in range(c - 1, c + 2):
                        expanded.add((nr, nc))
            if reserved_holes & expanded:
                ok = False
                break
            if any(grid[r, c] != GREY for r, c in candidate_hole + markers):
                ok = False
                break

            outside_sets = []
            outside_flat = set()
            for cells, marker in zip(piece_sets, markers):
                placed = _outside_piece(cells, side, marker, pr, pc, ph, pw)
                if any(not (0 <= r < height and 0 <= c < width) for r, c in placed):
                    ok = False
                    break
                if any(grid[r, c] != BG for r, c in placed):
                    ok = False
                    break
                if outside_flat & set(placed) or reserved_outside & set(placed):
                    ok = False
                    break
                outside_flat.update(placed)
                outside_sets.append(placed)
            if not ok:
                break

            for r, c in candidate_hole:
                grid[r, c] = BG
            for marker, cells in zip(markers, outside_sets):
                color = colors[color_pos]
                color_pos += 1
                mr, mc = marker
                grid[mr, mc] = color
                for r, c in cells:
                    grid[r, c] = color
            reserved_holes.update(expanded)
            reserved_outside.update(outside_flat)

        if ok:
            return grid

    raise RuntimeError("failed to generate a valid input")
