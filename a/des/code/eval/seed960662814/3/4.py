import random
from collections import Counter, defaultdict, deque

import numpy as np

BG = 0
GREY = 5
MAROON = 9
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 5000

# Oriented connected masks. Every mask changes under 90/180/270 rotation,
# which keeps the quadrant->rotation rule visible in generated samples.
SHAPE_LIBRARY = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 2), (1, 0), (1, 1), (1, 2), (2, 0)),
]

QUADRANT_ROTATIONS = {
    "ul": 0,
    "ur": 3,  # clockwise 90
    "lr": 2,
    "ll": 1,  # clockwise 270 == counterclockwise 90
}


def _normalize(coords):
    pts = np.asarray(list(coords), dtype=int)
    if pts.size == 0:
        return tuple(), (0, 0)
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))
    return norm, (r0, c0)


def _coords_to_mask(coords):
    norm, _ = _normalize(coords)
    if not norm:
        return np.zeros((0, 0), dtype=bool)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _mask_to_coords(mask):
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _rotate_coords(coords, k):
    mask = _coords_to_mask(coords)
    return _mask_to_coords(np.rot90(mask, k))


def _component_coords(mask, connectivity=4):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    if connectivity == 4:
        dirs = DIR4
    else:
        dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_components_by_color(grid, colors):
    colors = set(colors)
    mask = np.isin(grid, list(colors))
    comps = []
    for comp in _component_coords(mask, connectivity=4):
        vals = {int(grid[r, c]) for r, c in comp}
        if len(vals) != 1:
            continue
        color = vals.pop()
        norm, (r0, c0) = _normalize(comp)
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        comps.append(
            {
                "color": color,
                "coords": tuple(sorted(comp)),
                "shape": norm,
                "anchor": (r0, c0),
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "centroid": (sum(rows) / len(rows), sum(cols) / len(cols)),
            }
        )
    return comps


def _board_and_holes(grid):
    grey_mask = grid == GREY
    grey_comps = _component_coords(grey_mask, connectivity=4)
    if len(grey_comps) != 1:
        raise AssertionError("expected exactly one grey board")
    grey = np.asarray(grey_comps[0], dtype=int)
    r0 = int(grey[:, 0].min())
    c0 = int(grey[:, 1].min())
    r1 = int(grey[:, 0].max())
    c1 = int(grey[:, 1].max())
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    for comp in _component_coords(sub == BG, connectivity=4):
        abs_comp = [(r0 + r, c0 + c) for r, c in comp]
        norm, (hr0, hc0) = _normalize(abs_comp)
        rows = [r for r, _ in abs_comp]
        cols = [c for _, c in abs_comp]
        holes.append(
            {
                "coords": tuple(sorted(abs_comp)),
                "shape": norm,
                "anchor": (hr0, hc0),
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            }
        )
    return (r0, c0, r1, c1), holes


def _detect_axes(grid):
    maroon_pts = np.argwhere(grid == MAROON)
    if len(maroon_pts) == 0:
        raise AssertionError("missing dashed axes")
    row_counts = Counter(int(r) for r, _ in maroon_pts)
    col_counts = Counter(int(c) for _, c in maroon_pts)
    axis_r = max(row_counts.items(), key=lambda kv: (kv[1], -abs(kv[0] - grid.shape[0] / 2)))[0]
    axis_c = max(col_counts.items(), key=lambda kv: (kv[1], -abs(kv[0] - grid.shape[1] / 2)))[0]
    return int(axis_r), int(axis_c)


def _piece_quadrant(piece, axis_r, axis_c):
    cr, cc = piece["centroid"]
    top = cr < axis_r
    left = cc < axis_c
    if top and left:
        return "ul"
    if top and not left:
        return "ur"
    if not top and left:
        return "ll"
    return "lr"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each colored piece into the grey board after rotating it according to
    the quadrant defined by the dashed maroon axes.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    axis_r, axis_c = _detect_axes(g)
    board_bbox, holes = _board_and_holes(g)
    piece_colors = set(map(int, np.unique(g))) - {BG, GREY, MAROON}
    pieces = _find_components_by_color(g, piece_colors)

    if len(pieces) != len(holes):
        raise AssertionError("piece/hole count mismatch")

    hole_map = defaultdict(list)
    for hole in holes:
        hole_map[hole["shape"]].append(hole)

    out = np.full((h, w), BG, dtype=int)
    r0, c0, r1, c1 = board_bbox
    board_view = out[r0 : r1 + 1, c0 : c1 + 1]
    board_view[g[r0 : r1 + 1, c0 : c1 + 1] == GREY] = GREY

    used_holes = set()
    for piece in pieces:
        quadrant = _piece_quadrant(piece, axis_r, axis_c)
        rotated_shape = _rotate_coords(piece["shape"], QUADRANT_ROTATIONS[quadrant])
        candidates = [hole for hole in hole_map.get(rotated_shape, []) if hole["anchor"] not in used_holes]
        if len(candidates) != 1:
            raise AssertionError("shape match is not unique")
        hole = candidates[0]
        used_holes.add(hole["anchor"])
        for r, c in hole["coords"]:
            out[r, c] = piece["color"]

    if len(used_holes) != len(holes):
        raise AssertionError("not all holes were filled")
    return out


def _shape_variants(coords):
    return {_rotate_coords(coords, k) for k in range(4)}


_VALID_SHAPES = [shape for shape in SHAPE_LIBRARY if len(_shape_variants(shape)) == 4]


def _place_holes(board_h, board_w, hole_shapes):
    hole_mask = np.zeros((board_h, board_w), dtype=bool)
    padded = np.zeros((board_h, board_w), dtype=bool)

    for shape in hole_shapes:
        variants = list(_shape_variants(shape))
        random.shuffle(variants)
        sides = ["top", "right", "bottom", "left"]
        random.shuffle(sides)
        success = False
        for side in sides:
            options = []
            for orient in variants:
                rows = [r for r, _ in orient]
                cols = [c for _, c in orient]
                max_r = max(rows)
                max_c = max(cols)
                min_r = min(rows)
                min_c = min(cols)
                if side == "top":
                    offset_r = -min_r
                    for offset_c in range(-min_c, board_w - max_c):
                        options.append((orient, offset_r, offset_c))
                elif side == "bottom":
                    offset_r = board_h - 1 - max_r
                    for offset_c in range(-min_c, board_w - max_c):
                        options.append((orient, offset_r, offset_c))
                elif side == "left":
                    offset_c = -min_c
                    for offset_r in range(-min_r, board_h - max_r):
                        options.append((orient, offset_r, offset_c))
                else:
                    offset_c = board_w - 1 - max_c
                    for offset_r in range(-min_r, board_h - max_r):
                        options.append((orient, offset_r, offset_c))
            random.shuffle(options)
            for orient, offset_r, offset_c in options:
                abs_cells = [(r + offset_r, c + offset_c) for r, c in orient]
                if any(hole_mask[r, c] for r, c in abs_cells):
                    continue
                local = np.zeros_like(hole_mask)
                for r, c in abs_cells:
                    local[r, c] = True
                dilated = local.copy()
                for r, c in abs_cells:
                    for dr, dc in DIR4:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < board_h and 0 <= nc < board_w:
                            dilated[nr, nc] = True
                if np.any(dilated & padded):
                    continue
                hole_mask |= local
                padded |= dilated
                success = True
                break
            if success:
                break
        if not success:
            return None

    board = np.full((board_h, board_w), GREY, dtype=int)
    board[hole_mask] = BG
    if len(_component_coords(board == GREY, connectivity=4)) != 1:
        return None
    hole_shapes_exact = [_normalize(comp)[0] for comp in _component_coords(board == BG, connectivity=4)]
    if len(hole_shapes_exact) != len(hole_shapes):
        return None
    return board, hole_shapes_exact


def _scatter_position(box_h, box_w, axis_r, axis_c, quadrant, occupied, grid_h, grid_w):
    if quadrant == "ul":
        r_lo, r_hi = 1, axis_r - box_h - 1
        c_lo, c_hi = 1, axis_c - box_w - 1
    elif quadrant == "ur":
        r_lo, r_hi = 1, axis_r - box_h - 1
        c_lo, c_hi = axis_c + 2, grid_w - box_w - 2
    elif quadrant == "ll":
        r_lo, r_hi = axis_r + 2, grid_h - box_h - 2
        c_lo, c_hi = 1, axis_c - box_w - 1
    else:
        r_lo, r_hi = axis_r + 2, grid_h - box_h - 2
        c_lo, c_hi = axis_c + 2, grid_w - box_w - 2
    if r_lo > r_hi or c_lo > c_hi:
        return None

    choices = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    random.shuffle(choices)
    for r0, c0 in choices:
        bbox_cells = [(r0 + r, c0 + c) for r in range(box_h) for c in range(box_w)]
        if any((r, c) in occupied for r, c in bbox_cells):
            continue
        return r0, c0
    return None


def _board_top_ranges(grid_h, axis_r, board_h):
    ranges = []
    top_hi = axis_r - board_h - 2
    if top_hi >= 1:
        ranges.append((max(1, top_hi - 4), top_hi))
    bottom_lo = axis_r + 2
    bottom_hi = grid_h - board_h - 2
    if bottom_lo <= bottom_hi:
        ranges.append((bottom_lo, min(bottom_hi, bottom_lo + 4)))
    return ranges


def _gen_attempt():
    grid_h = random.randint(20, 27)
    grid_w = random.randint(22, 31)
    axis_r = random.randint(grid_h // 2 - 2, grid_h // 2 + 2)
    axis_c = random.randint(grid_w // 2 - 1, grid_w // 2 + 2)

    pieces_per_quadrant = random.choice([1, 2])
    quadrants = ["ul", "ur", "lr", "ll"] * pieces_per_quadrant
    random.shuffle(quadrants)
    n_pieces = len(quadrants)

    hole_shapes = random.sample(_VALID_SHAPES, n_pieces)
    board_h = random.randint(max(8, 3 * pieces_per_quadrant + 5), 12 + pieces_per_quadrant)
    board_w = random.randint(max(10, 4 * pieces_per_quadrant + 6), 15 + pieces_per_quadrant)
    if axis_c - board_w - 2 < 1:
        return None

    top_ranges = _board_top_ranges(grid_h, axis_r, board_h)
    if not top_ranges:
        return None
    board_top_lo, board_top_hi = random.choice(top_ranges)
    board_top = random.randint(board_top_lo, board_top_hi)
    board_left = random.randint(1, axis_c - board_w - 2)

    placed = _place_holes(board_h, board_w, hole_shapes)
    if placed is None:
        return None
    board, _ = placed

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[board_top : board_top + board_h, board_left : board_left + board_w] = board

    for c in range(0, grid_w, 2):
        if grid[axis_r, c] == BG:
            grid[axis_r, c] = MAROON
    for r in range(0, grid_h, 2):
        if grid[r, axis_c] == BG:
            grid[r, axis_c] = MAROON

    sub = grid[board_top : board_top + board_h, board_left : board_left + board_w]
    hole_info = []
    for comp in _component_coords(sub == BG, connectivity=4):
        abs_comp = [(board_top + r, board_left + c) for r, c in comp]
        norm, (r0, c0) = _normalize(abs_comp)
        hole_info.append({"shape": norm, "coords": tuple(sorted(abs_comp)), "anchor": (r0, c0)})

    if len(hole_info) != n_pieces:
        return None
    holes_by_shape = {hole["shape"]: hole for hole in hole_info}
    if len(holes_by_shape) != n_pieces:
        return None

    colors = PIECE_COLORS[:]
    random.shuffle(colors)
    while len(colors) < n_pieces:
        colors.append(random.choice(PIECE_COLORS))

    occupied = {(r, c) for r in range(grid_h) for c in range(grid_w) if grid[r, c] != BG}
    piece_specs = []
    for idx, quadrant in enumerate(quadrants):
        hole = hole_info[idx]
        piece_shape = _rotate_coords(hole["shape"], (-QUADRANT_ROTATIONS[quadrant]) % 4)
        box_h = max(r for r, _ in piece_shape) + 1
        box_w = max(c for _, c in piece_shape) + 1
        pos = _scatter_position(box_h, box_w, axis_r, axis_c, quadrant, occupied, grid_h, grid_w)
        if pos is None:
            return None
        pr, pc = pos
        color = colors[idx]
        for dr in range(-1, box_h + 1):
            for dc in range(-1, box_w + 1):
                rr = pr + dr
                cc = pc + dc
                if 0 <= rr < grid_h and 0 <= cc < grid_w:
                    occupied.add((rr, cc))
        coords = []
        for dr, dc in piece_shape:
            rr = pr + dr
            cc = pc + dc
            if grid[rr, cc] != BG:
                return None
            grid[rr, cc] = color
            coords.append((rr, cc))
        piece_specs.append({"quadrant": quadrant, "color": color, "shape": piece_shape, "hole": hole, "coords": tuple(coords)})

    expected = np.full((grid_h, grid_w), BG, dtype=int)
    board_view = expected[board_top : board_top + board_h, board_left : board_left + board_w]
    board_view[board == GREY] = GREY
    for spec in piece_specs:
        for r, c in spec["hole"]["coords"]:
            expected[r, c] = spec["color"]

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Ensure the three non-zero rotations are visibly exercised in every sample.
    seen_rotation_branches = {"ur": 0, "lr": 0, "ll": 0}
    for spec in piece_specs:
        q = spec["quadrant"]
        if q in seen_rotation_branches and _rotate_coords(spec["shape"], QUADRANT_ROTATIONS[q]) != spec["shape"]:
            seen_rotation_branches[q] += 1
    if not all(seen_rotation_branches.values()):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle sample")
