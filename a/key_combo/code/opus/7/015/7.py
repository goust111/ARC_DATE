from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, crop_bbox, find_components, get_bg


K = 5
SLOT = 5
SEP = 1
PITCH = SLOT + SEP
PREFIX_COL = 0
GRID_SHAPE = (K * SLOT + (K - 1) * SEP, 1 + K * SLOT + (K - 1) * SEP)
ROW_CENTERS = tuple(2 + PITCH * i for i in range(K))
ROW_STARTS = tuple(c - 2 for c in ROW_CENTERS)
COL_STARTS = tuple(1 + PITCH * j for j in range(K))
PREFIX_PALETTE = (Color.BLUE, Color.GREEN, Color.ORANGE, Color.PINK, Color.TEAL)
GEN_ATTEMPTS = 1200
GENERATE_ATTEMPTS = 400


def _slot_bbox(i: int, j: int) -> tuple[int, int, int, int]:
    r0 = ROW_STARTS[i]
    c0 = COL_STARTS[j]
    return bbox_coords(np.array([[r0, c0], [r0 + SLOT - 1, c0 + SLOT - 1]], dtype=int))


def _crop_slot(grid: np.ndarray, i: int, j: int) -> np.ndarray:
    return crop_bbox(grid, _slot_bbox(i, j))


def _mask_connected(mask: np.ndarray) -> bool:
    comps = find_components(mask.astype(int), bg=0, colors=[1], connectivity=4)
    return len(comps) == 1


def _read_prefix_rows(grid: np.ndarray) -> tuple[list[int], list[int]]:
    prefix_coords = np.argwhere(np.isin(grid[:, PREFIX_COL], PREFIX_PALETTE))
    assert prefix_coords.shape == (K, 1), "expected one prefix pixel per lattice row"

    rows = sorted(int(r) for r in prefix_coords[:, 0])
    assert rows == list(ROW_CENTERS), "prefix markers must sit at the row centers"
    colors = [int(grid[r, PREFIX_COL]) for r in rows]
    assert sorted(colors) == sorted(PREFIX_PALETTE), "row prefixes must be a permutation of the palette"
    return rows, colors


def _find_occluded_row(grid: np.ndarray) -> int:
    occluded = []
    for i, r0 in enumerate(ROW_STARTS):
        band = grid[r0 : r0 + SLOT, 1:]
        if np.all(band == Color.GREY):
            occluded.append(i)
    assert len(occluded) == 1, "expected exactly one fully grey lattice row"
    return occluded[0]


def _recover_sprite_mask(grid: np.ndarray, occluded_row: int) -> np.ndarray:
    sprite_mask = None
    visible_occupied = 0
    visible_empty = 0

    for i in range(K):
        for j in range(K):
            patch = _crop_slot(grid, i, j)
            if i == occluded_row:
                continue

            patch_colors = set(map(int, np.unique(patch)))
            assert patch_colors <= {Color.BLACK, Color.RED}, "visible slots may only contain black and red"
            has_sprite = np.any(patch == Color.RED)
            if has_sprite:
                candidate = patch == Color.RED
                if sprite_mask is None:
                    sprite_mask = candidate
                else:
                    assert np.array_equal(candidate, sprite_mask), "all visible copies must share the same sprite"
                visible_occupied += 1
            else:
                visible_empty += 1

    assert sprite_mask is not None and visible_occupied > 0 and visible_empty > 0
    assert _mask_connected(sprite_mask), "sprite must be 4-connected"

    area = int(sprite_mask.sum())
    row_sums = sprite_mask.sum(axis=1)
    col_sums = sprite_mask.sum(axis=0)
    assert 10 <= area <= 15, "sprite area must stay in the target irregular range"
    assert np.all((1 <= row_sums) & (row_sums <= 4)), "every sprite row must be informative"
    assert np.all((1 <= col_sums) & (col_sums <= 4)), "every sprite column must be informative"
    assert sprite_mask[occluded_row].any(), "the hidden row must contain at least one copy"

    for i in range(K):
        for j in range(K):
            if i == occluded_row:
                continue
            patch = _crop_slot(grid, i, j)
            if sprite_mask[i, j]:
                assert np.array_equal(patch == Color.RED, sprite_mask), "visible occupied slots must show the sprite"
            else:
                assert np.all(patch == Color.BLACK), "visible empty slots must stay black"

    return sprite_mask


def _render_output(sprite_mask: np.ndarray, row_colors: list[int]) -> np.ndarray:
    out = np.zeros(GRID_SHAPE, dtype=int)
    for i, color in enumerate(row_colors):
        out[ROW_CENTERS[i], PREFIX_COL] = int(color)

    for i in range(K):
        for j in range(K):
            if not sprite_mask[i, j]:
                continue
            r0, c0, _, _ = _slot_bbox(i, j)
            out[r0 : r0 + SLOT, c0 : c0 + SLOT][sprite_mask] = int(row_colors[i])
            out[r0 + i, c0 + j] = Color.YELLOW
    return out


def _render_input(sprite_mask: np.ndarray, row_colors: list[int], occluded_row: int) -> np.ndarray:
    inp = np.zeros(GRID_SHAPE, dtype=int)
    for i, color in enumerate(row_colors):
        inp[ROW_CENTERS[i], PREFIX_COL] = int(color)

    for i in range(K):
        if i == occluded_row:
            continue
        for j in range(K):
            if not sprite_mask[i, j]:
                continue
            r0, c0, _, _ = _slot_bbox(i, j)
            inp[r0 : r0 + SLOT, c0 : c0 + SLOT][sprite_mask] = Color.RED

    r0 = ROW_STARTS[occluded_row]
    inp[r0 : r0 + SLOT, 1:] = Color.GREY
    return inp


def _random_sprite_mask(rng: np.random.Generator) -> tuple[np.ndarray, int] | None:
    for _ in range(GEN_ATTEMPTS):
        area = int(rng.integers(10, 16))
        chosen = rng.choice(K * K, size=area, replace=False)
        mask = np.zeros((K, K), dtype=bool)
        mask.flat[chosen] = True

        if not _mask_connected(mask):
            continue

        row_sums = mask.sum(axis=1)
        col_sums = mask.sum(axis=0)
        if not np.all((1 <= row_sums) & (row_sums <= 4)):
            continue
        if not np.all((1 <= col_sums) & (col_sums <= 4)):
            continue
        if row_sums.max() < 3 or np.count_nonzero(row_sums == row_sums.max()) != 1:
            continue
        if int(mask.sum() - row_sums.max()) < 5:
            continue

        distinct_rows = {tuple(int(v) for v in row) for row in mask.astype(int)}
        distinct_cols = {tuple(int(v) for v in col) for col in mask.astype(int).T}
        if len(distinct_rows) < 3 or len(distinct_cols) < 3:
            continue

        occluded_row = int(np.argmax(row_sums))
        return mask, occluded_row
    return None


def _semantic_check(inp: np.ndarray, sprite_mask: np.ndarray, row_colors: list[int], occluded_row: int) -> bool:
    if inp.shape != GRID_SHAPE:
        return False
    if get_bg(inp) != Color.BLACK:
        return False
    if not np.array_equal(main(inp), _render_output(sprite_mask, row_colors)):
        return False

    row_sums = sprite_mask.sum(axis=1)
    if np.count_nonzero(row_sums == row_sums.max()) != 1:
        return False
    if int(row_sums[occluded_row]) != int(row_sums.max()):
        return False

    solved = _render_output(sprite_mask, row_colors)
    if np.array_equal(inp, solved):
        return False
    if int(np.count_nonzero(solved == Color.YELLOW)) != int(sprite_mask.sum()):
        return False

    visible_rows = [i for i in range(K) if i != occluded_row and row_sums[i] > 0]
    if len(visible_rows) < 2:
        return False
    if len({int(row_colors[i]) for i in visible_rows}) < 2:
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the self-similar sprite lattice, recolor each row by its prefix code, and mark lattice-coordinate pixels.

    Input structure:
    - The background is black, with a 5x5 lattice of 5x5 sprite slots separated by one-pixel black gutters and a one-pixel prefix column on the far left.
    - Visible copies of one 4-connected red sprite appear only in the slots selected by that sprite's own 5x5 mask.
    - One whole lattice row is hidden by a solid grey bar, and each lattice row has one distinct prefix pixel chosen from blue/green/orange/pink/teal.

    Rule:
    - Recover the shared 5x5 sprite mask from the visible red copies, then interpret that same mask as the coarse 5x5 occupancy map of the lattice.
    - Remove the grey bar and redraw the sprite in every occupied slot, including the hidden row; recolor every red sprite pixel in lattice row i to that row's prefix color.
    - In each occupied slot (i,j), overwrite the sprite pixel at local coordinate (i,j) with yellow; keep the black gutters, black background, and prefix pixels unchanged.

    Key insight:
    - The sprite is self-descriptive: its fine 5x5 foreground mask is also the coarse 5x5 placement map, so the hidden row can only be restored by combining self-similarity with the row-wise prefix recoloring and per-slot coordinate marking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, f"expected grid shape {GRID_SHAPE}, got {g.shape}"
    assert get_bg(g) == Color.BLACK

    _rows, row_colors = _read_prefix_rows(g)
    occluded_row = _find_occluded_row(g)
    sprite_mask = _recover_sprite_mask(g, occluded_row)
    return _render_output(sprite_mask, row_colors)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_ATTEMPTS):
        made = _random_sprite_mask(rng)
        if made is None:
            continue
        sprite_mask, occluded_row = made
        row_colors = [int(c) for c in rng.permutation(PREFIX_PALETTE)]
        inp = _render_input(sprite_mask, row_colors, occluded_row)
        if _semantic_check(inp, sprite_mask, row_colors, occluded_row):
            return inp
    raise RuntimeError("failed to generate a valid self-similar lattice instance")
