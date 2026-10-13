import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5

GEN_ATTEMPTS = 500
RECT_DIMS = [
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (3, 6),
    (6, 3),
    (4, 4),
    (4, 5),
    (5, 4),
    (4, 6),
    (6, 4),
]

RED_SYMBOL = np.array(
    [
        [GREY, GREY, GREY],
        [BG, GREY, BG],
        [BG, GREY, BG],
    ],
    dtype=int,
)
BLUE_SYMBOL = np.array(
    [
        [GREY, BG, BG],
        [GREY, GREY, BG],
        [GREY, GREY, GREY],
    ],
    dtype=int,
)
SYMBOL_TO_COLOR = {
    tuple(map(tuple, (RED_SYMBOL == GREY).astype(int))): RED,
    tuple(map(tuple, (BLUE_SYMBOL == GREY).astype(int))): BLUE,
}


def bbox(mask: np.ndarray):
    """arc_utils-style inclusive bbox for a boolean mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_coords(coords: np.ndarray):
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    """Minimal arc_utils-style paste helper."""
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """arc_utils-style connected components on a boolean mask."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        comp = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(comp, dtype=int))
    return comps


def _pattern_key(panel: np.ndarray):
    return tuple(map(tuple, (panel == GREY).astype(int)))


def _extract_color_patches(grid: np.ndarray, color: int) -> list[np.ndarray]:
    patches = []
    for coords in components_mask(grid == color, connectivity=4):
        box = bbox_coords(coords)
        patch = np.where(crop_bbox(grid, box) == color, color, BG)
        patches.append(patch)
    assert len(patches) == 2, f"expected exactly two components of color {color}"
    return patches


def _decode_symbol_colors(grid: np.ndarray) -> list[int]:
    symbol_boxes = []
    for coords in components_mask(grid == GREY, connectivity=8):
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        if (r1 - r0 + 1, c1 - c0 + 1) == (3, 3):
            symbol_boxes.append(box)
    assert len(symbol_boxes) == 2, "expected exactly two 3x3 symbol masks"
    symbol_boxes.sort(key=lambda box: (box[0], box[1]))

    decoded = []
    for box in symbol_boxes:
        panel = np.where(crop_bbox(grid, box) == GREY, GREY, BG)
        key = _pattern_key(panel)
        assert key in SYMBOL_TO_COLOR, "unknown symbol pattern"
        decoded.append(SYMBOL_TO_COLOR[key])
    return decoded


def _pack_pair(patches: list[np.ndarray], color: int) -> np.ndarray:
    coords_a = [tuple(map(int, rc)) for rc in np.argwhere(patches[0] != BG)]
    coords_b = [tuple(map(int, rc)) for rc in np.argwhere(patches[1] != BG)]
    set_a = set(coords_a)
    total_cells = len(coords_a) + len(coords_b)

    best_score = None
    best_union = None
    h1, w1 = patches[0].shape
    h2, w2 = patches[1].shape

    for dr in range(-h2, h1 + 1):
        for dc in range(-w2, w1 + 1):
            shifted_b = {(r + dr, c + dc) for r, c in coords_b}
            if set_a & shifted_b:
                continue
            union = set_a | shifted_b
            rows = [r for r, _c in union]
            cols = [c for _r, c in union]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            area = height * width
            holes = area - total_cells
            score = (holes, area, height + width, abs(dr) + abs(dc), dr, dc)
            if best_score is None or score < best_score:
                best_score = score
                best_union = (union, min_r, min_c, height, width)

    assert best_union is not None, "no non-overlapping packing found"
    union, min_r, min_c, height, width = best_union
    out = np.full((height, width), BG, dtype=int)
    for r, c in union:
        out[r - min_r, c - min_c] = color
    return out


def _stack_rectangles(top_rect: np.ndarray, bottom_rect: np.ndarray) -> np.ndarray:
    h1, w1 = top_rect.shape
    h2, w2 = bottom_rect.shape
    out = np.full((h1 + h2, max(w1, w2)), BG, dtype=int)
    paste(out, top_rect, 0, 0, transparent=BG)
    paste(out, bottom_rect, h1, 0, transparent=BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack two complementary same-color piece pairs into rectangles, then stack those rectangles by decoded symbol order.

    Input structure:
    - The background is black.
    - The left side contains exactly two red components and the right side contains exactly two blue components.
    - Each same-color pair is a translation-only complementary partition of a solid rectangle.
    - Between them sit exactly two vertically separated 3x3 grey-on-black symbol masks.

    Rule:
    - For the red pair and for the blue pair independently, try all translation-only non-overlapping placements and keep the one whose cropped bounding box contains the fewest black cells.
    - Decode the two grey symbol masks with a fixed 3x3 pattern table into the colors red and blue.
    - Stack the packed rectangle whose color matches the upper symbol on top, and the rectangle matching the lower symbol directly below it, left-aligned with no gap.
    - Return only the stacked region.

    Key insight:
    - Geometry and symbol decoding are independent subproblems: one reconstructs two solid rectangles, the other decides which reconstructed rectangle goes on top.
    """
    g = np.asarray(grid, dtype=int)
    red_rect = _pack_pair(_extract_color_patches(g, RED), RED)
    blue_rect = _pack_pair(_extract_color_patches(g, BLUE), BLUE)
    top_color, bottom_color = _decode_symbol_colors(g)

    rects = {
        RED: red_rect,
        BLUE: blue_rect,
    }
    return _stack_rectangles(rects[top_color], rects[bottom_color])


def _neighbors4(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _is_connected(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        cell = stack.pop()
        for nxt in _neighbors4(cell, h, w):
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _is_irregular(cells: set[tuple[int, int]]) -> bool:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    return height > 1 and width > 1 and height * width > len(cells)


def _grow_connected_subset(all_cells: set[tuple[int, int]], h: int, w: int, size: int) -> set[tuple[int, int]] | None:
    piece = {random.choice(tuple(all_cells))}
    frontier = set(_neighbors4(next(iter(piece)), h, w)) & all_cells
    while len(piece) < size:
        frontier -= piece
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        piece.add(cell)
        frontier |= set(_neighbors4(cell, h, w)) & all_cells
    return piece


def _cells_to_patch(cells: set[tuple[int, int]], h: int, w: int, color: int) -> np.ndarray:
    patch = np.full((h, w), BG, dtype=int)
    for r, c in cells:
        patch[r, c] = color
    return patch


def _partition_rectangle(rect_h: int, rect_w: int, color: int) -> tuple[np.ndarray, np.ndarray]:
    area = rect_h * rect_w
    all_cells = {(r, c) for r in range(rect_h) for c in range(rect_w)}
    size_choices = [
        size
        for size in range(4, min(12, area - 4) + 1)
        if 4 <= area - size <= 12
    ]
    assert size_choices

    for _ in range(800):
        size = random.choice(size_choices)
        piece_a = _grow_connected_subset(all_cells, rect_h, rect_w, size)
        if piece_a is None:
            continue
        piece_b = all_cells - piece_a
        if not _is_connected(piece_b, rect_h, rect_w):
            continue
        if not _is_irregular(piece_a) or not _is_irregular(piece_b):
            continue
        patch_a = _cells_to_patch(piece_a, rect_h, rect_w, color)
        patch_b = _cells_to_patch(piece_b, rect_h, rect_w, color)
        return patch_a, patch_b

    raise RuntimeError("failed to split rectangle into two irregular connected complementary pieces")


def _boxes_clear(box_a, box_b, pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _place_pair_in_region(grid: np.ndarray, patches: list[np.ndarray], col_lo: int, col_hi: int) -> None:
    h = grid.shape[0]
    candidates = []
    for patch in patches:
        ph, pw = patch.shape
        row_lo = 1
        row_hi = h - ph - 1
        if row_hi < row_lo:
            row_lo = 0
            row_hi = h - ph
        patch_positions = [
            (r, c)
            for r in range(row_lo, row_hi + 1)
            for c in range(col_lo, col_hi - pw + 1)
        ]
        candidates.append(patch_positions)

    for _ in range(400):
        pos0 = random.choice(candidates[0])
        pos1 = random.choice(candidates[1])
        box0 = (pos0[0], pos0[1], pos0[0] + patches[0].shape[0] - 1, pos0[1] + patches[0].shape[1] - 1)
        box1 = (pos1[0], pos1[1], pos1[0] + patches[1].shape[0] - 1, pos1[1] + patches[1].shape[1] - 1)
        if not _boxes_clear(box0, box1, pad=1):
            continue
        paste(grid, patches[0], pos0[0], pos0[1], transparent=BG)
        paste(grid, patches[1], pos1[0], pos1[1], transparent=BG)
        return

    raise RuntimeError("failed to place two pieces in the requested region")


def _build_expected(red_rect: np.ndarray, blue_rect: np.ndarray, top_color: int) -> np.ndarray:
    top_rect = red_rect if top_color == RED else blue_rect
    bottom_rect = blue_rect if top_color == RED else red_rect
    return _stack_rectangles(top_rect, bottom_rect)


def _generate_case():
    red_h, red_w = random.choice(RECT_DIMS)
    blue_h, blue_w = random.choice(RECT_DIMS)

    red_pieces = list(_partition_rectangle(red_h, red_w, RED))
    blue_pieces = list(_partition_rectangle(blue_h, blue_w, BLUE))
    random.shuffle(red_pieces)
    random.shuffle(blue_pieces)

    left_w = max(piece.shape[1] for piece in red_pieces) + random.randint(1, 2)
    right_w = max(piece.shape[1] for piece in blue_pieces) + random.randint(1, 2)
    middle_w = 3 + random.randint(0, 1)
    outer_left = random.randint(1, 2)
    outer_right = random.randint(1, 2)
    gap_left = random.randint(1, 2)
    gap_right = random.randint(1, 2)

    grid_w = outer_left + left_w + gap_left + middle_w + gap_right + right_w + outer_right
    grid_h = random.randint(13, 19)
    if grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    left_lo = outer_left
    left_hi = left_lo + left_w
    mid_lo = left_hi + gap_left
    mid_hi = mid_lo + middle_w
    right_lo = mid_hi + gap_right
    right_hi = right_lo + right_w

    try:
        _place_pair_in_region(grid, red_pieces, left_lo, left_hi)
        _place_pair_in_region(grid, blue_pieces, right_lo, right_hi)
    except RuntimeError:
        return None

    top_color = random.choice([RED, BLUE])
    bottom_color = BLUE if top_color == RED else RED
    top_symbol = RED_SYMBOL if top_color == RED else BLUE_SYMBOL
    bottom_symbol = RED_SYMBOL if bottom_color == RED else BLUE_SYMBOL

    sym_gap = random.randint(1, 2)
    sym_top_r = random.randint(1, grid_h - (3 + sym_gap + 3) - 1)
    sym_c = mid_lo + random.randint(0, middle_w - 3)
    paste(grid, top_symbol, sym_top_r, sym_c, transparent=BG)
    paste(grid, bottom_symbol, sym_top_r + 3 + sym_gap, sym_c, transparent=BG)

    red_rect = np.full((red_h, red_w), RED, dtype=int)
    blue_rect = np.full((blue_h, blue_w), BLUE, dtype=int)
    expected = _build_expected(red_rect, blue_rect, top_color)
    swapped_expected = _build_expected(red_rect, blue_rect, bottom_color)

    swapped_grid = grid.copy()
    swapped_grid[sym_top_r : sym_top_r + 3, sym_c : sym_c + 3] = BG
    swapped_grid[sym_top_r + 3 + sym_gap : sym_top_r + 6 + sym_gap, sym_c : sym_c + 3] = BG
    paste(swapped_grid, bottom_symbol, sym_top_r, sym_c, transparent=BG)
    paste(swapped_grid, top_symbol, sym_top_r + 3 + sym_gap, sym_c, transparent=BG)

    return {
        "grid": grid,
        "expected": expected,
        "swapped_grid": swapped_grid,
        "swapped_expected": swapped_expected,
    }


def _valid_generated_case(case: dict) -> bool:
    grid = case["grid"]
    expected = case["expected"]
    swapped_grid = case["swapped_grid"]
    swapped_expected = case["swapped_expected"]

    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    try:
        produced = main(grid)
        swapped_out = main(swapped_grid)
    except Exception:
        return False

    if not np.array_equal(produced, expected):
        return False
    if not np.array_equal(swapped_out, swapped_expected):
        return False
    if np.array_equal(produced, swapped_out):
        return False

    if len(_extract_color_patches(grid, RED)) != 2 or len(_extract_color_patches(grid, BLUE)) != 2:
        return False
    if _decode_symbol_colors(grid) == _decode_symbol_colors(swapped_grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None and _valid_generated_case(case):
            return case["grid"]
    raise RuntimeError("failed to generate a valid complementary-pack symbol-order ARC input")
