from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
PALETTE = (3, 4, 6, 7, 8, 9)
QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
GEN_RETRIES = 400
PATCH_RETRIES = 200


def _dominant_value(values: np.ndarray) -> tuple[int, int]:
    uniq, counts = np.unique(values, return_counts=True)
    idx = int(np.argmax(counts))
    return int(uniq[idx]), int(counts[idx])


def _find_cut_axis(grid: np.ndarray, axis: int) -> tuple[int, int]:
    size = int(grid.shape[axis])
    line_len = int(grid.shape[1 - axis])
    candidates: list[tuple[int, int]] = []
    for idx in range(size):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        color, count = _dominant_value(line)
        if color in (RED, BLUE) and count >= line_len - 1:
            candidates.append((idx, color))
    assert len(candidates) == 1, f"expected one cut line on axis {axis}, got {candidates}"
    return candidates[0]


def _marker_quadrant(block: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = np.argwhere(block == GREY)
    assert coords.shape == (1, 2), "each block must contain exactly one grey marker"
    mr, mc = map(int, coords[0])
    h, w = map(int, block.shape)
    target = (0 if mr * 2 < h else 1, 0 if mc * 2 < w else 1)
    return target, (mr, mc)


def _apply_piece_transform(piece: np.ndarray, rotate_cw: bool, flip_ud: bool) -> np.ndarray:
    out = piece.copy()
    if rotate_cw:
        out = np.rot90(out, k=3)
    if flip_ud:
        out = np.flipud(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the four cut blocks by marker-indicated destination and line-color-controlled geometry.

    Input structure:
    - bg = black.
    - Exactly one full horizontal cut line and one full vertical cut line split the grid into four rectangular blocks.
    - The cut lines are colored red and/or blue, and each block contains non-line colored pixels plus exactly one grey marker pixel.

    Rule:
    - The grey marker's quadrant inside a block specifies that block's destination quadrant in the output.
    - Remove the grey marker from every block.
    - If the vertical cut line is red, rotate every block 90 degrees clockwise before placement.
    - If the horizontal cut line is red, vertically flip the two blocks that started below that line after any rotation.
    - Place the transformed blocks into their target quadrants and concatenate them directly without cut lines.

    Key insight:
    - The grey marker tells where each block goes, while the horizontal and vertical cut-line colors independently encode the geometric transforms applied before assembly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cut_row, hline_color = _find_cut_axis(g, axis=0)
    cut_col, vline_color = _find_cut_axis(g, axis=1)
    h, w = map(int, g.shape)
    assert 0 < cut_row < h - 1 and 0 < cut_col < w - 1

    top_h = int(cut_row)
    bottom_h = int(h - cut_row - 1)
    left_w = int(cut_col)
    right_w = int(w - cut_col - 1)
    assert top_h == bottom_h and left_w == right_w, "blocks must be same-sized"

    blocks = {
        (0, 0): g[:cut_row, :cut_col],
        (0, 1): g[:cut_row, cut_col + 1 :],
        (1, 0): g[cut_row + 1 :, :cut_col],
        (1, 1): g[cut_row + 1 :, cut_col + 1 :],
    }

    placed: dict[tuple[int, int], np.ndarray] = {}
    for (src_r, src_c), block in blocks.items():
        target, (mr, mc) = _marker_quadrant(block)
        assert target not in placed, f"duplicate destination quadrant {target}"

        piece = block.copy()
        piece[mr, mc] = BG
        piece = _apply_piece_transform(
            piece,
            rotate_cw=(vline_color == RED),
            flip_ud=(hline_color == RED and src_r == 1),
        )
        placed[target] = piece

    assert len(placed) == 4
    tl = placed[(0, 0)]
    tr = placed[(0, 1)]
    bl = placed[(1, 0)]
    br = placed[(1, 1)]
    assert tl.shape[0] == tr.shape[0] and bl.shape[0] == br.shape[0]
    assert tl.shape[1] == bl.shape[1] and tr.shape[1] == br.shape[1]
    return np.block([[tl, tr], [bl, br]])


def _quadrant_indices(length: int, upper_or_left: bool) -> list[int]:
    interior = list(range(1, length - 1))
    if upper_or_left:
        picks = [idx for idx in interior if idx * 2 < length - 1]
        return picks or [1]
    picks = [idx for idx in interior if idx * 2 > length - 1]
    return picks or [length - 2]


def _sample_marker_pos(height: int, width: int, target_idx: int) -> tuple[int, int]:
    tr, tc = QUADRANTS[target_idx]
    row = random.choice(_quadrant_indices(height, upper_or_left=(tr == 0)))
    col = random.choice(_quadrant_indices(width, upper_or_left=(tc == 0)))
    return int(row), int(col)


def _rect_cells(height: int, width: int) -> list[tuple[int, int]]:
    rh = random.randint(1, min(3, height - 2))
    rw = random.randint(1, min(3, width - 2))
    r0 = random.randint(1, height - rh - 1)
    c0 = random.randint(1, width - rw - 1)
    return [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]


def _hbar_cells(height: int, width: int) -> list[tuple[int, int]]:
    length = random.randint(2, min(4, width - 2))
    row = random.randint(1, height - 2)
    col = random.randint(1, width - length - 1)
    return [(row, c) for c in range(col, col + length)]


def _vbar_cells(height: int, width: int) -> list[tuple[int, int]]:
    length = random.randint(2, min(4, height - 2))
    row = random.randint(1, height - length - 1)
    col = random.randint(1, width - 2)
    return [(r, col) for r in range(row, row + length)]


def _lshape_cells(height: int, width: int) -> list[tuple[int, int]]:
    arm_h = random.randint(2, min(4, height - 2))
    arm_w = random.randint(2, min(4, width - 2))
    r0 = random.randint(1, height - arm_h - 1)
    c0 = random.randint(1, width - arm_w - 1)
    orient = random.choice(("dr", "dl", "ur", "ul"))
    cells: set[tuple[int, int]] = set()
    if orient == "dr":
        cells.update((r0 + dr, c0) for dr in range(arm_h))
        cells.update((r0, c0 + dc) for dc in range(arm_w))
    elif orient == "dl":
        cells.update((r0 + dr, c0 + arm_w - 1) for dr in range(arm_h))
        cells.update((r0, c0 + dc) for dc in range(arm_w))
    elif orient == "ur":
        cells.update((r0 + arm_h - 1 - dr, c0) for dr in range(arm_h))
        cells.update((r0 + arm_h - 1, c0 + dc) for dc in range(arm_w))
    else:
        cells.update((r0 + arm_h - 1 - dr, c0 + arm_w - 1) for dr in range(arm_h))
        cells.update((r0 + arm_h - 1, c0 + dc) for dc in range(arm_w))
    return sorted(cells)


def _diag_cells(height: int, width: int) -> list[tuple[int, int]]:
    length = random.randint(2, min(4, height - 2, width - 2))
    r0 = random.randint(1, height - length - 1)
    c0 = random.randint(1, width - length - 1)
    if random.random() < 0.5:
        return [(r0 + k, c0 + k) for k in range(length)]
    return [(r0 + k, c0 + length - 1 - k) for k in range(length)]


def _step_cells(height: int, width: int) -> list[tuple[int, int]]:
    length = random.randint(2, min(4, height - 2, width - 2))
    r0 = random.randint(1, height - length - 1)
    c0 = random.randint(1, width - length - 1)
    cells = []
    for k in range(length):
        cells.append((r0 + k, c0 + k))
        if k + 1 < length:
            cells.append((r0 + k, c0 + k + 1))
    return sorted(set(cells))


def _random_shape(height: int, width: int) -> list[tuple[int, int]]:
    kind = random.choice(("rect", "hbar", "vbar", "lshape", "diag", "step"))
    if kind == "rect":
        return _rect_cells(height, width)
    if kind == "hbar":
        return _hbar_cells(height, width)
    if kind == "vbar":
        return _vbar_cells(height, width)
    if kind == "lshape":
        return _lshape_cells(height, width)
    if kind == "diag":
        return _diag_cells(height, width)
    return _step_cells(height, width)


def _can_place(patch: np.ndarray, cells: list[tuple[int, int]], marker_pos: tuple[int, int]) -> bool:
    for r, c in cells:
        if (r, c) == marker_pos or int(patch[r, c]) != BG:
            return False
    return True


def _make_content(
    height: int,
    width: int,
    marker_pos: tuple[int, int],
    rotate_cw: bool,
    flip_ud: bool,
) -> np.ndarray:
    inner_area = (height - 2) * (width - 2)
    for _ in range(PATCH_RETRIES):
        patch = np.full((height, width), BG, dtype=int)
        colors = random.sample(PALETTE, k=random.randint(2, 4))
        target_shapes = random.randint(2, 4)
        attempts = 0
        while target_shapes > 0 and attempts < 60:
            cells = _random_shape(height, width)
            if len(cells) + int(np.count_nonzero(patch)) > max(7, inner_area // 2):
                attempts += 1
                continue
            if not _can_place(patch, cells, marker_pos):
                attempts += 1
                continue
            color = random.choice(colors)
            for r, c in cells:
                patch[r, c] = color
            target_shapes -= 1
            attempts += 1

        extras = random.randint(0, 2)
        for _extra in range(extras):
            free = [
                (r, c)
                for r in range(1, height - 1)
                for c in range(1, width - 1)
                if (r, c) != marker_pos and int(patch[r, c]) == BG
            ]
            if not free:
                break
            r, c = random.choice(free)
            patch[r, c] = random.choice(colors)

        if int(patch[marker_pos]) != BG:
            continue
        colored = patch[patch != BG]
        if colored.size < 5:
            continue
        if len(np.unique(colored)) < 2:
            continue

        transformed = _apply_piece_transform(patch, rotate_cw=rotate_cw, flip_ud=flip_ud)
        if (rotate_cw or flip_ud) and transformed.shape == patch.shape and np.array_equal(transformed, patch):
            continue
        return patch
    raise RuntimeError("failed to sample a varied block pattern")


def _piece_signature(piece: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, piece.shape)), piece.tobytes()


def _assemble_grid(
    pieces: dict[tuple[int, int], np.ndarray],
    hline_color: int,
    vline_color: int,
    intersection_owner: str,
) -> np.ndarray:
    tl = pieces[(0, 0)]
    tr = pieces[(0, 1)]
    bl = pieces[(1, 0)]
    br = pieces[(1, 1)]
    block_h, block_w = map(int, tl.shape)

    grid = np.full((block_h * 2 + 1, block_w * 2 + 1), BG, dtype=int)
    grid[:block_h, :block_w] = tl
    grid[:block_h, block_w + 1 :] = tr
    grid[block_h + 1 :, :block_w] = bl
    grid[block_h + 1 :, block_w + 1 :] = br
    grid[block_h, :] = int(hline_color)
    grid[:, block_w] = int(vline_color)
    grid[block_h, block_w] = int(hline_color if intersection_owner == "h" else vline_color)
    return grid


def _gen_attempt() -> np.ndarray:
    block_h = random.randint(5, 12)
    block_w = random.randint(5, 12)
    hline_color = random.choice((RED, BLUE))
    vline_color = random.choice((RED, BLUE))
    intersection_owner = random.choice(("h", "v"))

    perm = random.sample(range(4), k=4)
    if perm == [0, 1, 2, 3]:
        return None

    source_pieces: dict[tuple[int, int], np.ndarray] = {}
    transformed_sigs: set[tuple[tuple[int, int], bytes]] = set()

    for src_idx, (src_r, src_c) in enumerate(QUADRANTS):
        target_idx = int(perm[src_idx])
        marker_pos = _sample_marker_pos(block_h, block_w, target_idx)
        content = _make_content(
            block_h,
            block_w,
            marker_pos=marker_pos,
            rotate_cw=(vline_color == RED),
            flip_ud=(hline_color == RED and src_r == 1),
        )
        transformed = _apply_piece_transform(
            content,
            rotate_cw=(vline_color == RED),
            flip_ud=(hline_color == RED and src_r == 1),
        )
        sig = _piece_signature(transformed)
        if sig in transformed_sigs:
            return None
        transformed_sigs.add(sig)

        piece = content.copy()
        piece[marker_pos] = GREY
        source_pieces[(src_r, src_c)] = piece

    grid = _assemble_grid(source_pieces, hline_color, vline_color, intersection_owner)
    out = main(grid)
    assert isinstance(out, np.ndarray)
    assert max(grid.shape) <= 30 and max(out.shape) <= 30
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
