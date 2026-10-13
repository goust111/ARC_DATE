from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_ATTEMPTS = 400
GRID_H_RANGE = (11, 16)
GRID_W_RANGE = (11, 16)
PATTERN_SIZES = [(3, 3), (3, 4), (4, 3), (4, 4), (5, 3), (3, 5), (5, 4), (4, 5), (5, 5)]
GREY_COUNTS = (2, 3, 4)
PATTERN_CORNERS = ("tl", "tr", "bl", "br")


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _dense_mask_from_sparse_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    assert np.all(rel % 2 == 0), "sparse sprite pixels must lie on a two-step lattice"

    h = (r1 - r0) // 2 + 1
    w = (c1 - c0) // 2 + 1
    logical = np.zeros((h, w), dtype=bool)
    for dr, dc in rel:
        logical[int(dr) // 2, int(dc) // 2] = True
    return np.kron(logical, np.ones((2, 2), dtype=bool))[: 2 * h - 1, : 2 * w - 1]


def _logical_mask_neighbors(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_connected_logical_mask(max_h: int, max_w: int) -> np.ndarray:
    h = random.randint(1, max_h)
    w = random.randint(1, max_w)
    mode = random.choice(["rect", "rect", "blob", "path"])

    if mode == "rect":
        return np.ones((h, w), dtype=bool)

    total = h * w
    target = 1 if total == 1 else random.randint(2, total)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}

    frontier = set(_logical_mask_neighbors(start, h, w))
    while len(cells) < target and frontier:
        nxt = random.choice(list(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        for nb in _logical_mask_neighbors(nxt, h, w):
            if nb not in cells:
                frontier.add(nb)
        if mode == "path":
            frontier = {
                nb
                for cell in list(cells)[-2:]
                for nb in _logical_mask_neighbors(cell, h, w)
                if nb not in cells
            }
            if not frontier and len(cells) < target:
                break

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _sparse_coords_from_logical(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = []
    for r, c in np.argwhere(mask):
        coords.append((top + 2 * int(r), left + 2 * int(c)))
    return np.array(coords, dtype=int)


def _choose_grey_positions(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(200):
        chosen = random.sample(cells, count)
        coords = np.array(chosen, dtype=int)
        if _bbox_coords(coords) == (0, 0, h - 1, w - 1):
            return sorted(chosen)
    return None


def _pattern_anchor(grid_h: int, grid_w: int, pat_h: int, pat_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, grid_w - pat_w
    if corner == "bl":
        return grid_h - pat_h, 0
    return grid_h - pat_h, grid_w - pat_w


def _place_boxes(
    grid_h: int,
    grid_w: int,
    box_sizes: list[tuple[int, int]],
    blocked: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(box_sizes)), key=lambda i: box_sizes[i][0] * box_sizes[i][1], reverse=True)
    anchors: list[tuple[int, int] | None] = [None] * len(box_sizes)
    placed = blocked[:]

    for idx in order:
        box_h, box_w = box_sizes[idx]
        candidates = []
        for r in range(grid_h - box_h + 1):
            for c in range(grid_w - box_w + 1):
                box = (r, c, r + box_h - 1, c + box_w - 1)
                if all(_boxes_separated(box, other, pad=1) for other in placed):
                    candidates.append((r, c))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        anchors[idx] = (top, left)
        placed.append((top, left, top + box_h - 1, left + box_w - 1))

    if any(anchor is None for anchor in anchors):
        return None
    return [(int(anchor[0]), int(anchor[1])) for anchor in anchors]


def _build_output_from_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    grey_coords = np.argwhere(g == GREY)
    assert len(grey_coords) >= 2, "expected grey control pixels"

    src_r0, src_c0, src_r1, src_c1 = _bbox_coords(grey_coords)
    src = g[src_r0 : src_r1 + 1, src_c0 : src_c1 + 1]
    block_h, block_w = src.shape

    outside = np.ones(g.shape, dtype=bool)
    outside[src_r0 : src_r1 + 1, src_c0 : src_c1 + 1] = False
    sprite_colors = [int(c) for c in np.unique(g[outside]) if int(c) != BLACK]

    sprites = []
    for color in sprite_colors:
        coords = np.argwhere((g == color) & outside)
        assert len(coords) > 0, "each sprite color must appear outside the source pattern"
        bbox = _bbox_coords(coords)
        dense = _dense_mask_from_sparse_coords(coords)
        assert dense.shape[0] <= block_h and dense.shape[1] <= block_w, "sprite exceeds block size"
        sprites.append({"color": color, "bbox": bbox, "dense": dense})

    sprites.sort(key=lambda sprite: (sprite["bbox"][0], sprite["bbox"][1]))
    slots = [(r, c) for r in range(block_h) for c in range(block_w) if int(src[r, c]) == GREY]
    assert len(slots) == len(sprites), "grey slots and sprites must match one-to-one"

    out = np.zeros((block_h * block_h, block_w * block_w), dtype=int)
    for (slot_r, slot_c), sprite in zip(slots, sprites):
        dense = sprite["dense"]
        patch = np.zeros((block_h, block_w), dtype=int)
        ph, pw = dense.shape
        patch_view = patch[:ph, :pw]
        patch_view[dense] = int(sprite["color"])
        out[
            slot_r * block_h : (slot_r + 1) * block_h,
            slot_c * block_w : (slot_c + 1) * block_w,
        ] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Densify sparse sprites and stamp them into output blocks selected by the source pattern's grey pixels.

    Input structure:
    - The grid has a black background, one corner source pattern containing grey control pixels, and several detached sparse monochrome sprites elsewhere.
    - Each sprite uses one distinct color and sits on a two-step lattice inside its own bounding box.
    - The grey pixels' bounding box is the source pattern rectangle; the number of grey pixels matches the number of sprites.

    Rule:
    - Take the minimal rectangle containing all grey pixels as the source pattern, with size R x C.
    - Densify every sparse sprite by filling the one-cell gaps implied by its two-step lattice, producing a solidified sprite whose size does not exceed R x C.
    - Sort sprites by the row-major order of their bounding-box top-left corners.
    - Build an output of size (R*R) x (C*C), viewed as an R x C grid of R x C blocks.
    - Scan the source pattern row-major; for each grey pixel, place the next densified sprite into the corresponding block's top-left corner, leaving the rest of that block black. Non-grey source pixels map to all-black blocks.

    Key insight:
    - Two independent orders interact: grey pixels define destination slots in row-major block order, while the sparse sprites must first be densified and then assigned by their spatial top-left order.
    """
    return _build_output_from_grid(grid)


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    pat_h, pat_w = random.choice(PATTERN_SIZES)
    if pat_h > grid_h or pat_w > grid_w:
        return None

    grey_count = random.choice(GREY_COUNTS)
    if grey_count > pat_h * pat_w:
        return None
    grey_positions = _choose_grey_positions(pat_h, pat_w, grey_count)
    if grey_positions is None:
        return None

    logical_h_max = (pat_h + 1) // 2
    logical_w_max = (pat_w + 1) // 2
    logical_masks = [_random_connected_logical_mask(logical_h_max, logical_w_max) for _ in range(grey_count)]
    dense_sizes = [(2 * mask.shape[0] - 1, 2 * mask.shape[1] - 1) for mask in logical_masks]

    corner = random.choice(PATTERN_CORNERS)
    pat_top, pat_left = _pattern_anchor(grid_h, grid_w, pat_h, pat_w, corner)
    pattern_box = (pat_top, pat_left, pat_top + pat_h - 1, pat_left + pat_w - 1)

    sprite_anchors = _place_boxes(grid_h, grid_w, dense_sizes, [pattern_box])
    if sprite_anchors is None or len(sprite_anchors) != grey_count:
        return None

    accent_mode = random.choice(["none", "sparse", "dense"])
    need_accent = accent_mode != "none" and grey_count < pat_h * pat_w
    color_count = grey_count + (1 if need_accent else 0)
    if color_count > len(PALETTE):
        return None
    chosen_colors = random.sample(PALETTE, color_count)
    sprite_colors = chosen_colors[:grey_count]
    accent_color = chosen_colors[-1] if need_accent else None

    grid = np.zeros((grid_h, grid_w), dtype=int)

    for dr, dc in grey_positions:
        grid[pat_top + dr, pat_left + dc] = GREY

    if accent_color is not None:
        other_cells = [(r, c) for r in range(pat_h) for c in range(pat_w) if (r, c) not in set(grey_positions)]
        if accent_mode == "dense":
            fill_n = random.randint(max(1, (2 * len(other_cells)) // 3), len(other_cells))
        else:
            fill_n = random.randint(1, max(1, len(other_cells) // 2))
        for dr, dc in random.sample(other_cells, fill_n):
            grid[pat_top + dr, pat_left + dc] = int(accent_color)

    for color, mask, (top, left) in zip(sprite_colors, logical_masks, sprite_anchors):
        coords = _sparse_coords_from_logical(mask, top, left)
        for r, c in coords:
            grid[int(r), int(c)] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse-sprite stamping input")
