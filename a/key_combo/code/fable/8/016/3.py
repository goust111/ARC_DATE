from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
COPY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SLOT_COUNT = 5
TILE_SIZE = 4
TOP_BORDER_ROW = 5
BOTTOM_BORDER_ROW = TOP_BORDER_ROW + TILE_SIZE + 1
GRID_H = BOTTOM_BORDER_ROW + 1
GRID_W = SLOT_COUNT * TILE_SIZE + SLOT_COUNT + 1
MAX_GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    b = _bbox(mask)
    assert b is not None, "expected non-empty mask"
    r0, c0, r1, c1 = b
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _reference_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << 9):
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if (bits >> idx) & 1:
                mask[idx // 3, idx % 3] = True
        if mask.sum() not in (4, 5):
            continue
        if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        if len(_components4(mask)) != 1:
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        masks.append(mask)
    assert masks, "expected at least one asymmetric 3x3 reference mask"
    return masks


REFERENCE_MASKS = _reference_masks()


def _slot_cols(slot: int) -> tuple[int, int]:
    left_sep = slot * (TILE_SIZE + 1)
    return left_sep + 1, left_sep + 1 + TILE_SIZE


def _tile_rows() -> tuple[int, int]:
    return TOP_BORDER_ROW + 1, BOTTOM_BORDER_ROW


def _most_common_excluding(grid: np.ndarray, excluded: set[int]) -> int:
    counts = Counter(int(v) for v in grid.reshape(-1) if int(v) not in excluded)
    assert counts, "expected at least one non-excluded color"
    return counts.most_common(1)[0][0]


def _find_layout(grid: np.ndarray) -> tuple[int, int, list[int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best: tuple[int, int, list[int], int] | None = None
    for color in map(int, np.unique(g)):
        full_rows = [r for r in range(h) if bool(np.all(g[r, :] == color))]
        if len(full_rows) < 2:
            continue
        for i, top in enumerate(full_rows):
            for bottom in full_rows[i + 1 :]:
                if bottom - top < 2:
                    continue
                full_cols = [c for c in range(w) if bool(np.all(g[top : bottom + 1, c] == color))]
                if len(full_cols) < 3:
                    continue
                widths = [full_cols[j + 1] - full_cols[j] - 1 for j in range(len(full_cols) - 1)]
                if min(widths) <= 0 or len(set(widths)) != 1:
                    continue
                if top <= 0 or not bool(np.any(g[:top, :] == color)):
                    continue
                score = (len(full_cols), bottom - top)
                if best is None or score > best[:2]:
                    best = (score[0], score[1], full_cols, color)
                    best_top = top
                    best_bottom = bottom
    assert best is not None, "failed to detect separator layout"
    return best_top, best_bottom, best[2], best[3]


def _reference_mask_from_grid(grid: np.ndarray, top: int, sep_color: int) -> np.ndarray:
    ref_area = np.asarray(grid[:top, :] == sep_color, dtype=bool)
    assert ref_area.any(), "expected a reference shape above the tile row"
    return _crop_mask(ref_area)


def _match_rotation(obj_mask: np.ndarray, ref_mask: np.ndarray) -> int:
    matches = [k for k in range(4) if np.array_equal(np.rot90(ref_mask, k), obj_mask)]
    assert len(matches) == 1, "tile shape must match the reference under exactly one rotation"
    return int(matches[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten each tile copy, then cyclically move it right by the number of quarter-turns it had.

    Input structure:
    - Grey separator lines split the lower part of the grid into one ordered row of equal tiles.
    - A separate grey reference shape above the tile row gives the canonical 0-degree orientation.
    - Each non-empty tile contains a single monochrome rotated copy of that reference on the background.

    Rule:
    - Detect the separator color, tile boundaries, and the reference shape above them.
    - For each occupied tile, compare its shape to the reference to read a unique rotation k in {0,1,2,3}.
    - Rotate the copy back to the reference orientation and place it into the tile k slots to the right cyclically, leaving emptied tiles blank while keeping the separators and reference unchanged.

    Key insight:
    - The visible rotation is not an output attribute; it encodes how far the normalized shape must be shifted along the ordered tile row.
    """
    g = np.asarray(grid, dtype=int)
    top, bottom, full_cols, sep_color = _find_layout(g)
    ref_mask = _reference_mask_from_grid(g, top, sep_color)
    bg = _most_common_excluding(g, {sep_color})

    out = np.full_like(g, bg)
    out[g == sep_color] = sep_color

    tile_r0, tile_r1 = top + 1, bottom
    slot_count = len(full_cols) - 1

    for slot in range(slot_count):
        c0 = full_cols[slot] + 1
        c1 = full_cols[slot + 1]
        tile = g[tile_r0:tile_r1, c0:c1]
        fg_mask = (tile != bg) & (tile != sep_color)
        if not fg_mask.any():
            continue

        colors = np.unique(tile[fg_mask])
        assert len(colors) == 1, "each tile copy must be monochrome"
        color = int(colors[0])

        b = _bbox(fg_mask)
        assert b is not None
        r0, cc0, r1, cc1 = b
        obj_mask = fg_mask[r0 : r1 + 1, cc0 : cc1 + 1]
        k = _match_rotation(obj_mask, ref_mask)

        dst_slot = (slot + k) % slot_count
        dst_c0 = full_cols[dst_slot] + 1
        rr0 = tile_r0 + r0
        cc_abs0 = dst_c0 + cc0

        target = out[rr0 : rr0 + ref_mask.shape[0], cc_abs0 : cc_abs0 + ref_mask.shape[1]]
        assert target.shape == ref_mask.shape, "reference placement must stay inside tile bounds"
        target[ref_mask] = color

    return out


def _build_layout_canvas() -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_BORDER_ROW, :] = SEP
    grid[BOTTOM_BORDER_ROW, :] = SEP
    for slot in range(SLOT_COUNT + 1):
        c = slot * (TILE_SIZE + 1)
        grid[TOP_BORDER_ROW : BOTTOM_BORDER_ROW + 1, c] = SEP
    return grid


def _place_reference(grid: np.ndarray, ref_mask: np.ndarray) -> None:
    max_left = GRID_W - ref_mask.shape[1]
    left = random.randint(0, max_left)
    top = random.randint(1, 2)
    patch = grid[top : top + ref_mask.shape[0], left : left + ref_mask.shape[1]]
    assert patch.shape == ref_mask.shape
    patch[ref_mask] = SEP


def _valid_assignment(src_slots: list[int], rotations: list[int]) -> bool:
    dst = [(slot + rot) % SLOT_COUNT for slot, rot in zip(src_slots, rotations)]
    return len(set(dst)) == len(dst)


def _sample_assignment() -> tuple[list[int], list[int]]:
    for _ in range(400):
        src_count = random.choice([4, 5])
        src_slots = sorted(random.sample(range(SLOT_COUNT), src_count))
        if src_count == 4:
            rotations = [0, 1, 2, 3]
            random.shuffle(rotations)
        else:
            rotations = [0, 1, 2, 3, random.randint(0, 3)]
            random.shuffle(rotations)
        if _valid_assignment(src_slots, rotations):
            return src_slots, rotations
    raise RuntimeError("failed to sample a non-conflicting rotation assignment")


def _expected_output(
    ref_mask: np.ndarray,
    src_slots: list[int],
    rotations: list[int],
    placements: dict[int, tuple[int, int]],
    colors: dict[int, int],
    reference_grid: np.ndarray,
) -> np.ndarray:
    out = np.full_like(reference_grid, BG)
    out[reference_grid == SEP] = SEP
    tile_r0, _tile_r1 = _tile_rows()
    for slot, rot in zip(src_slots, rotations):
        dst = (slot + rot) % SLOT_COUNT
        local_r, local_c = placements[slot]
        c0, _c1 = _slot_cols(dst)
        patch = out[tile_r0 + local_r : tile_r0 + local_r + 3, c0 + local_c : c0 + local_c + 3]
        patch[ref_mask] = colors[slot]
    return out


def _gen_attempt() -> np.ndarray | None:
    ref_mask = random.choice(REFERENCE_MASKS).copy()
    grid = _build_layout_canvas()
    _place_reference(grid, ref_mask)

    src_slots, rotations = _sample_assignment()
    available_colors = random.sample(COPY_COLORS, len(src_slots))

    placements: dict[int, tuple[int, int]] = {}
    colors: dict[int, int] = {}
    tile_r0, _tile_r1 = _tile_rows()
    for slot, rot, color in zip(src_slots, rotations, available_colors):
        c0, _c1 = _slot_cols(slot)
        local_r = random.randint(0, 1)
        local_c = random.randint(0, 1)
        placements[slot] = (local_r, local_c)
        colors[slot] = color
        patch = grid[tile_r0 + local_r : tile_r0 + local_r + 3, c0 + local_c : c0 + local_c + 3]
        patch[np.rot90(ref_mask, rot)] = color

    expected = _expected_output(ref_mask, src_slots, rotations, placements, colors, grid)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(actual, grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
