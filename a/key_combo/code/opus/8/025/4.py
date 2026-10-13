from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import Color, bbox, crop_bbox, find_components, paste, random_free_location


PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
GRID_RANGE = (24, 29)
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 200


def _mask_from_coords(size: int, coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


MASK_PARTITIONS: dict[int, list[list[np.ndarray]]] = {
    5: [
        [
            _mask_from_coords(5, [(0, 0), (0, 4), (4, 0), (4, 4)]),
            _mask_from_coords(5, [(0, 2), (2, 0), (2, 4), (4, 2)]),
            _mask_from_coords(5, [(1, 1), (1, 3), (3, 1), (3, 3)]),
            _mask_from_coords(5, [(2, 2)]),
        ],
        [
            _mask_from_coords(5, [(0, 0), (0, 4), (4, 0), (4, 4)]),
            _mask_from_coords(5, [(0, 2), (2, 0), (2, 4), (4, 2)]),
            _mask_from_coords(5, [(1, 2), (2, 1), (2, 3), (3, 2)]),
            _mask_from_coords(5, [(2, 2)]),
        ],
    ],
    7: [
        [
            _mask_from_coords(7, [(0, 0), (0, 6), (6, 0), (6, 6)]),
            _mask_from_coords(7, [(0, 3), (3, 0), (3, 6), (6, 3)]),
            _mask_from_coords(7, [(1, 1), (1, 5), (5, 1), (5, 5)]),
            _mask_from_coords(7, [(2, 3), (3, 2), (3, 3), (3, 4), (4, 3)]),
        ],
        [
            _mask_from_coords(7, [(0, 0), (0, 6), (6, 0), (6, 6)]),
            _mask_from_coords(7, [(0, 3), (3, 0), (3, 6), (6, 3)]),
            _mask_from_coords(7, [(1, 3), (3, 1), (3, 5), (5, 3)]),
            _mask_from_coords(7, [(2, 2), (2, 4), (3, 3), (4, 2), (4, 4)]),
        ],
    ],
}


def _component_count(mask: np.ndarray) -> int:
    comps = find_components(mask.astype(int), bg=Color.BLACK, connectivity=4, monochromatic=False)
    return len(comps)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    b = bbox(mask)
    assert b is not None
    return crop_bbox(mask, b).astype(bool)


def _same_shape(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    return (b1[2] - b1[0], b1[3] - b1[1]) == (b2[2] - b2[0], b2[3] - b2[1])


def _extract_damage_mask(grid: np.ndarray, color: int) -> np.ndarray:
    comps = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True, colors=[color])
    assert len(comps) == 2, f"expected exactly two components for color {color}"
    comps = sorted(comps, key=lambda comp: int(comp["area"]))
    damaged, complete = comps
    assert int(complete["area"]) > int(damaged["area"]), "complete copy must be larger than damaged copy"
    assert _same_shape(complete["bbox"], damaged["bbox"]), "same-color copies must share bbox shape"

    complete_crop = crop_bbox(grid, complete["bbox"]) == int(color)
    damaged_crop = crop_bbox(grid, damaged["bbox"]) == int(color)
    missing = complete_crop & ~damaged_crop
    extra = damaged_crop & ~complete_crop
    assert missing.any(), "expected at least one missing pixel"
    assert not extra.any(), "damaged copy cannot add foreground"

    missing_bbox = bbox(missing)
    assert missing_bbox is not None
    cropped = crop_bbox(missing, missing_bbox)
    assert cropped.shape[0] % 2 == 1 and cropped.shape[1] % 2 == 1, "cropped damage mask must be odd-sized"
    return cropped


def main(grid: np.ndarray) -> np.ndarray:
    """Extract each color's missing cells from complete/damaged motif pairs and center-overlay the cropped damage masks.

    Input structure:
    - The grid has a black background and contains 3 or 4 foreground colors from blue, red, green, and yellow.
    - Each foreground color appears in exactly two separated 4-connected monochrome motifs with the same bbox shape.
    - For each color, one motif is the complete copy and the other is the damaged copy with some cells turned to black.

    Rule:
    - For each color, compare the two same-color motifs after aligning their bboxes by top-left corner.
    - Keep the cells that are foreground in the larger motif and black in the smaller motif, then crop that damage mask to its own bbox.
    - Make a black output canvas whose size is the maximum cropped damage-mask height and width, and paste every cropped mask centered on that canvas in its original color.

    Key insight:
    - The useful signal is not any single motif itself but the per-color missing-cell masks; only after extracting those masks and recentering them into one compact frame does the hidden composite pattern appear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = sorted(int(c) for c in np.unique(g) if int(c) != Color.BLACK)
    assert 3 <= len(colors) <= 4, "expected 3 or 4 foreground colors"

    damage_masks: list[tuple[int, np.ndarray]] = []
    max_h = 0
    max_w = 0
    for color in colors:
        damage = _extract_damage_mask(g, color)
        damage_masks.append((color, damage))
        max_h = max(max_h, int(damage.shape[0]))
        max_w = max(max_w, int(damage.shape[1]))

    out = np.zeros((max_h, max_w), dtype=int)
    occupied = np.zeros((max_h, max_w), dtype=bool)
    for color, damage in damage_masks:
        dh, dw = map(int, damage.shape)
        row_extra = max_h - dh
        col_extra = max_w - dw
        assert row_extra % 2 == 0 and col_extra % 2 == 0, "damage masks must center on a shared canvas"
        r0 = row_extra // 2
        c0 = col_extra // 2
        coords = np.argwhere(damage)
        for r, c in coords:
            rr = r0 + int(r)
            cc = c0 + int(c)
            assert not occupied[rr, cc], "centered damage masks must not overlap"
            occupied[rr, cc] = True
            out[rr, cc] = int(color)
    return out


def _center_overlay(specs: list[tuple[int, np.ndarray]]) -> np.ndarray:
    max_h = max(int(mask.shape[0]) for _color, mask in specs)
    max_w = max(int(mask.shape[1]) for _color, mask in specs)
    out = np.zeros((max_h, max_w), dtype=int)
    for color, mask in specs:
        r0 = (max_h - mask.shape[0]) // 2
        c0 = (max_w - mask.shape[1]) // 2
        colored = np.where(mask, int(color), Color.BLACK).astype(int)
        paste(out, colored, r0, c0, transparent=Color.BLACK)
    return out


def _choose_damage_specs() -> list[tuple[int, np.ndarray]]:
    size = random.choice(sorted(MASK_PARTITIONS))
    variants = MASK_PARTITIONS[size]
    masks = [_tight_mask(mask) for mask in random.choice(variants)]
    n_colors = random.choice((3, 4))

    while True:
        idxs = sorted(random.sample(range(len(masks)), n_colors))
        chosen = [masks[i].copy() for i in idxs]
        if any(mask.shape[0] == size for mask in chosen) and any(mask.shape[0] < size or mask.shape[1] < size for mask in chosen):
            break

    colors = random.sample(PALETTE, n_colors)
    assigned = list(zip(colors, chosen))
    random.shuffle(assigned)
    return assigned


def _make_pair(mask: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray] | None:
    mh, mw = map(int, mask.shape)
    min_h = max(mh, 5)
    min_w = max(mw, 5)
    height_choices = list(range(min_h, 8))
    width_choices = list(range(min_w, 8))

    for _ in range(PAIR_ATTEMPTS):
        bh = random.choice(height_choices)
        bw = random.choice(width_choices)
        r_off = random.randint(0, bh - mh)
        c_off = random.randint(0, bw - mw)

        missing = np.zeros((bh, bw), dtype=bool)
        missing[r_off : r_off + mh, c_off : c_off + mw] = mask

        interior = [
            (r, c)
            for r in range(1, bh - 1)
            for c in range(1, bw - 1)
            if not missing[r, c]
        ]
        if len(interior) < 2:
            continue

        hole_count = random.randint(1, min(4, len(interior)))
        holes = random.sample(interior, hole_count)

        complete_mask = np.ones((bh, bw), dtype=bool)
        for r, c in holes:
            complete_mask[int(r), int(c)] = False
        damaged_mask = complete_mask & ~missing

        if not complete_mask.any() or not damaged_mask.any():
            continue
        if bbox(complete_mask) != (0, 0, bh - 1, bw - 1):
            continue
        if bbox(damaged_mask) != (0, 0, bh - 1, bw - 1):
            continue
        if _component_count(complete_mask) != 1 or _component_count(damaged_mask) != 1:
            continue
        if int(damaged_mask.sum()) >= int(complete_mask.sum()):
            continue

        complete = np.zeros((bh, bw), dtype=int)
        damaged = np.zeros((bh, bw), dtype=int)
        complete[complete_mask] = int(color)
        damaged[damaged_mask] = int(color)
        return complete, damaged
    return None


def _assemble_grid(patches: list[np.ndarray]) -> np.ndarray | None:
    rows = random.randint(*GRID_RANGE)
    cols = random.randint(*GRID_RANGE)
    grid = np.zeros((rows, cols), dtype=int)
    reserved = np.zeros((rows, cols), dtype=int)

    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: int(np.prod(patches[idx].shape)), reverse=True)

    placements: dict[int, tuple[int, int]] = {}
    for idx in order:
        patch = patches[idx]
        rect = np.ones(patch.shape, dtype=int)
        try:
            loc = random_free_location(reserved, rect, transparent=Color.BLACK, border=1, padding=1, padding_connectivity=8)
        except ValueError:
            return None
        placements[idx] = loc
        paste(reserved, rect, loc[0], loc[1], transparent=Color.BLACK)

    for idx, patch in enumerate(patches):
        r0, c0 = placements[idx]
        paste(grid, patch, r0, c0, transparent=Color.BLACK)
    return grid


def _validate_case(grid: np.ndarray, expected: np.ndarray) -> bool:
    colors = sorted(int(c) for c in np.unique(grid) if int(c) != Color.BLACK)
    if not (3 <= len(colors) <= 4):
        return False

    extracted: list[tuple[int, np.ndarray]] = []
    for color in colors:
        comps = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True, colors=[color])
        if len(comps) != 2:
            return False
        comps = sorted(comps, key=lambda comp: int(comp["area"]))
        damaged, complete = comps
        if int(complete["area"]) <= int(damaged["area"]):
            return False
        if not _same_shape(complete["bbox"], damaged["bbox"]):
            return False

        complete_crop = crop_bbox(grid, complete["bbox"]) == color
        damaged_crop = crop_bbox(grid, damaged["bbox"]) == color
        shared_holes = (~complete_crop) & (~damaged_crop)
        if not shared_holes.any():
            return False

        missing = complete_crop & ~damaged_crop
        extra = damaged_crop & ~complete_crop
        if not missing.any() or extra.any():
            return False
        missing_bbox = bbox(missing)
        if missing_bbox is None:
            return False
        cropped = crop_bbox(missing, missing_bbox)
        if cropped.shape[0] % 2 == 0 or cropped.shape[1] % 2 == 0:
            return False
        extracted.append((color, cropped))

    max_h = max(mask.shape[0] for _color, mask in extracted)
    max_w = max(mask.shape[1] for _color, mask in extracted)
    if not any(mask.shape != (max_h, max_w) for _color, mask in extracted):
        return False

    overlay = np.zeros((max_h, max_w), dtype=int)
    occupied = np.zeros((max_h, max_w), dtype=bool)
    for color, mask in extracted:
        r0 = (max_h - mask.shape[0]) // 2
        c0 = (max_w - mask.shape[1]) // 2
        if (max_h - mask.shape[0]) % 2 or (max_w - mask.shape[1]) % 2:
            return False
        coords = np.argwhere(mask)
        for r, c in coords:
            rr = r0 + int(r)
            cc = c0 + int(c)
            if occupied[rr, cc]:
                return False
            occupied[rr, cc] = True
            overlay[rr, cc] = int(color)

    return np.array_equal(overlay, expected) and np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    specs = _choose_damage_specs()
    patches: list[np.ndarray] = []
    for color, mask in specs:
        pair = _make_pair(mask, color)
        if pair is None:
            return None
        complete, damaged = pair
        if random.random() < 0.5:
            patches.extend([complete, damaged])
        else:
            patches.extend([damaged, complete])

    grid = _assemble_grid(patches)
    if grid is None:
        return None

    expected = _center_overlay(specs)
    if not _validate_case(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
