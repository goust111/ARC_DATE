import random
import numpy as np


BG = 0
GREY = 5


def _bbox(mask):
    rs, cs = np.where(mask)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _square_mask(mask):
    r0, c0, r1, c1 = _bbox(mask)
    crop = mask[r0:r1 + 1, c0:c1 + 1]
    n = max(crop.shape)
    sq = np.zeros((n, n), dtype=bool)
    ro = (n - crop.shape[0]) // 2
    co = (n - crop.shape[1]) // 2
    sq[ro:ro + crop.shape[0], co:co + crop.shape[1]] = crop
    return sq, r0, c0


def _sym_order(mask):
    sq, _, _ = _square_mask(mask)
    if np.array_equal(sq, np.rot90(sq, 1)):
        return 4
    if np.array_equal(sq, np.rot90(sq, 2)):
        return 2
    return 1


def main(grid: np.ndarray) -> np.ndarray:
    """Use right-panel rotational symmetry order to rotate-copy the left template.

    Input structure: A full grey vertical divider separates two equal panels.
    The left panel has one asymmetric monochrome template; the right panel has
    one monochrome instruction shape whose rotational symmetry order is 1, 2,
    or 4.
    Rule: Output the left panel only. Let k be the right shape's rotational
    symmetry order. Draw the left template rotated around its own bounding-box
    center by the k evenly spaced rotations. Pixels covered by multiple copies
    take the right-panel instruction color.
    Key insight: The right object is an instruction encoded by symmetry order,
    not a sprite to copy directly.
    """
    h, w = grid.shape
    div = [c for c in range(w) if np.count_nonzero(grid[:, c] == GREY) == h][0]
    left = grid[:, :div]
    right = grid[:, div + 1:]
    lcols = [int(x) for x in np.unique(left) if x != BG]
    rcols = [int(x) for x in np.unique(right) if x != BG]
    if not lcols or not rcols:
        return left.copy()
    lc, rc = lcols[0], rcols[0]
    lmask = left == lc
    rmask = right == rc
    k = _sym_order(rmask)
    sq, r0, c0 = _square_mask(lmask)
    counts = np.zeros(left.shape, dtype=int)
    for t in range(k):
        rot = np.rot90(sq, -t * (4 // k))
        for rr, cc in zip(*np.where(rot)):
            nr, nc = r0 + rr, c0 + cc
            if 0 <= nr < h and 0 <= nc < div:
                counts[nr, nc] += 1
    out = np.zeros_like(left)
    out[counts == 1] = lc
    out[counts > 1] = rc
    return out


def generate_input() -> np.ndarray:
    h, panel_w = 15, 13
    grid = np.zeros((h, panel_w * 2 + 1), dtype=int)
    grid[:, panel_w] = GREY
    lc, rc = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2)
    left_shape = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)]
    for dr, dc in left_shape:
        grid[5 + dr, 4 + dc] = lc
    k = random.choice([1, 2, 4])
    if k == 4:
        shape = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
    elif k == 2:
        shape = [(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)]
    else:
        shape = [(0, 0), (1, 0), (1, 1), (2, 1), (3, 2)]
    for dr, dc in shape:
        grid[5 + dr, panel_w + 1 + 4 + dc] = rc
    return grid
