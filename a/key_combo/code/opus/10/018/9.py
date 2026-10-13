from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BLUE = 1
GUIDE = 7
_OFFSETS_2X2 = ((0, 0), (0, 1), (1, 0), (1, 1))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _all_exact_sprite_covers(blue_mask: np.ndarray) -> list[tuple[tuple[tuple[int, int], ...], tuple[frozenset[tuple[int, int]], ...]]]:
    blue_cells = frozenset(map(tuple, np.argwhere(blue_mask)))
    total = len(blue_cells)
    assert total > 0 and total % 4 == 0, "blue pixels must split into four equal sprites"
    sprite_area = total // 4
    assert 1 <= sprite_area <= 4, "each sprite must fit inside a 2x2 mask"

    h, w = map(int, blue_mask.shape)
    solutions: list[tuple[tuple[tuple[int, int], ...], tuple[frozenset[tuple[int, int]], ...]]] = []
    for mask in itertools.combinations(_OFFSETS_2X2, sprite_area):
        placements: list[frozenset[tuple[int, int]]] = []
        for r in range(h - 1):
            for c in range(w - 1):
                cells = frozenset((r + dr, c + dc) for dr, dc in mask)
                if cells <= blue_cells:
                    placements.append(cells)

        for combo in itertools.combinations(placements, 4):
            covered: set[tuple[int, int]] = set()
            ok = True
            for cells in combo:
                if covered & cells:
                    ok = False
                    break
                covered.update(cells)
            if ok and covered == blue_cells:
                solutions.append((tuple(mask), tuple(combo)))
    return solutions


def _find_unique_sprite_mask(blue_mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    solutions = _all_exact_sprite_covers(blue_mask)
    masks = {mask for mask, _combo in solutions}
    assert len(masks) == 1, "expected a unique 2x2 sprite mask"
    return next(iter(masks))


def _guide_orientation(grid: np.ndarray) -> str:
    guide_mask = grid == GUIDE
    comps = _mask_components(guide_mask)
    assert len(comps) == 2, "expected exactly two guide bars"

    orientation = None
    for comp in comps:
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        if height == 1 and 2 <= width <= 3:
            current = "H"
        elif width == 1 and 2 <= height <= 3:
            current = "V"
        else:
            raise AssertionError("guide bars must be straight length-2/3 segments")
        if orientation is None:
            orientation = current
        else:
            assert orientation == current, "guide bars must be parallel"
    assert orientation is not None
    return orientation


def main(grid: np.ndarray) -> np.ndarray:
    """Compress each separated sprite into a line segment and place the four lines in the output corners.

    Input structure:
    - The grid uses black background, blue sprite pixels, and orange guide bars.
    - The blue foreground is exactly coverable by four translated copies of one unique 2x2 sprite mask.
    - The orange guides are exactly two parallel straight bars of length 2 or 3 placed to the right side or below the blue packing.

    Rule:
    - Infer the unique 2x2 sprite mask whose four translated copies cover all blue pixels.
    - Let p be that mask's blue-pixel count, and read the orange bars' orientation: horizontal bars mean horizontal output lines, vertical bars mean vertical output lines.
    - Build a four-corner layout with a one-cell middle gap and place four blue line segments of length p in the output corners; remove the orange guides.

    Key insight:
    - The packed blue region only tells you the sprite area, while the orange guide bars independently choose the compression direction; both cues are required to determine the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BG, BLUE, GUIDE}, "unexpected colors in input"
    assert np.any(g == BLUE), "expected blue sprite pixels"
    assert np.any(g == GUIDE), "expected orange guide bars"

    sprite_mask = _find_unique_sprite_mask(g == BLUE)
    p = len(sprite_mask)
    orientation = _guide_orientation(g)

    if orientation == "H":
        out = np.zeros((3, 2 * p + 1), dtype=int)
        out[0, :p] = BLUE
        out[0, p + 1 :] = BLUE
        out[2, :p] = BLUE
        out[2, p + 1 :] = BLUE
        return out

    out = np.zeros((2 * p + 1, 3), dtype=int)
    out[:p, 0] = BLUE
    out[p + 1 :, 0] = BLUE
    out[:p, 2] = BLUE
    out[p + 1 :, 2] = BLUE
    return out


_GEN_LOCAL_SHAPE = (6, 6)
_GEN_MASKS_BY_AREA = {
    2: (
        ((0, 0), (1, 1)),
        ((0, 1), (1, 0)),
    ),
    3: (
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    ),
}
_GEN_ANCHOR_ROWS = range(4)
_GEN_ANCHOR_COLS = range(5)
_GEN_MAX_ATTEMPTS = 400
_GEN_GUIDE_LENGTHS = (2, 3)
_GEN_GAPS = (1, 2)
_GEN_PAD_VALUES = (0, 1)
_GEN_BAR_GAPS = (1, 2)
_GEN_PACKINGS_CACHE: dict[tuple[tuple[int, int], ...], list[tuple[tuple[int, int], ...]]] = {}


def _gen_local_packing_mask(occupied: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(_GEN_LOCAL_SHAPE, dtype=bool)
    for r, c in occupied:
        mask[int(r), int(c)] = True
    return mask


def _gen_is_compact_packing(occupied: set[tuple[int, int]]) -> bool:
    coords = np.array(sorted(occupied), dtype=int)
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    if not (int(min_r) == 0 and int(min_c) == 0 and int(max_r) >= 3 and int(max_c) >= 4):
        return False
    return len(_mask_components(_gen_local_packing_mask(occupied))) <= 2


def _gen_valid_packings(mask: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    cached = _GEN_PACKINGS_CACHE.get(mask)
    if cached is not None:
        return cached

    anchors: list[tuple[int, int, frozenset[tuple[int, int]]]] = []
    for r in _GEN_ANCHOR_ROWS:
        for c in _GEN_ANCHOR_COLS:
            cells = frozenset((r + dr, c + dc) for dr, dc in mask)
            anchors.append((int(r), int(c), cells))

    valid: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.combinations(anchors, 4):
        occupied: set[tuple[int, int]] = set()
        placements: list[tuple[int, int]] = []
        for r, c, cells in combo:
            if occupied & cells:
                break
            occupied.update(cells)
            placements.append((int(r), int(c)))
        else:
            if not _gen_is_compact_packing(occupied):
                continue
            local_blue = _gen_local_packing_mask(occupied)
            cropped_blue = _crop_nonzero_bbox(local_blue)
            if len(_all_exact_sprite_covers(cropped_blue)) == 1:
                valid.append(tuple(placements))

    _GEN_PACKINGS_CACHE[mask] = valid
    return valid


def _crop_nonzero_bbox(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr != 0)
    assert len(coords) > 0
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    return arr[int(min_r) : int(max_r) + 1, int(min_c) : int(max_c) + 1]


def _gen_patch_from_packing(mask: tuple[tuple[int, int], ...], placements: tuple[tuple[int, int], ...]) -> np.ndarray:
    patch = np.zeros(_GEN_LOCAL_SHAPE, dtype=int)
    for r, c in placements:
        for dr, dc in mask:
            patch[int(r + dr), int(c + dc)] = BLUE
    return patch


def _gen_blue_patch() -> np.ndarray | None:
    areas = list(_GEN_MASKS_BY_AREA)
    random.shuffle(areas)
    for area in areas:
        masks = list(_GEN_MASKS_BY_AREA[area])
        random.shuffle(masks)
        for mask in masks:
            packings = _gen_valid_packings(mask)
            if not packings:
                continue
            patch = _gen_patch_from_packing(mask, random.choice(packings))
            return _crop_nonzero_bbox(patch)
    return None


def _guide_shape(orientation: str, guide_len: int, bar_gap: int) -> tuple[int, int]:
    if orientation == "H":
        return bar_gap + 2, guide_len
    return guide_len, bar_gap + 2


def _draw_guides(
    grid: np.ndarray,
    orientation: str,
    location: str,
    guide_len: int,
    bar_gap: int,
    inner_top: int,
    inner_left: int,
    blue_top: int,
    blue_left: int,
    blue_shape: tuple[int, int],
    gap: int,
    inner_h: int,
    inner_w: int,
) -> None:
    blue_h, blue_w = blue_shape
    guide_h, guide_w = _guide_shape(orientation, guide_len, bar_gap)
    if location == "right":
        base_c = blue_left + blue_w + gap
        row0 = inner_top + random.randint(0, inner_h - guide_h)
        if orientation == "H":
            grid[row0, base_c : base_c + guide_len] = GUIDE
            grid[row0 + bar_gap + 1, base_c : base_c + guide_len] = GUIDE
        else:
            grid[row0 : row0 + guide_len, base_c] = GUIDE
            grid[row0 : row0 + guide_len, base_c + bar_gap + 1] = GUIDE
        return

    base_r = blue_top + blue_h + gap
    col0 = inner_left + random.randint(0, inner_w - guide_w)
    if orientation == "H":
        grid[base_r, col0 : col0 + guide_len] = GUIDE
        grid[base_r + bar_gap + 1, col0 : col0 + guide_len] = GUIDE
    else:
        grid[base_r : base_r + guide_len, col0] = GUIDE
        grid[base_r : base_r + guide_len, col0 + bar_gap + 1] = GUIDE


def _gen_attempt() -> np.ndarray | None:
    blue_patch = _gen_blue_patch()
    if blue_patch is None:
        return None

    blue_h, blue_w = map(int, blue_patch.shape)
    orientation = random.choice(("H", "V"))
    location = random.choice(("right", "bottom"))
    guide_len = random.choice(_GEN_GUIDE_LENGTHS)
    bar_gap = random.choice(_GEN_BAR_GAPS)
    gap = random.choice(_GEN_GAPS)
    top_pad = random.choice(_GEN_PAD_VALUES)
    left_pad = random.choice(_GEN_PAD_VALUES)
    bottom_pad = random.choice(_GEN_PAD_VALUES)
    right_pad = random.choice(_GEN_PAD_VALUES)
    guide_h, guide_w = _guide_shape(orientation, guide_len, bar_gap)

    if location == "right":
        inner_h = max(blue_h, guide_h)
        height = top_pad + inner_h + bottom_pad
        width = left_pad + blue_w + gap + guide_w + right_pad
        blue_top = top_pad + random.randint(0, inner_h - blue_h)
        blue_left = left_pad
    else:
        inner_w = max(blue_w, guide_w)
        height = top_pad + blue_h + gap + guide_h + bottom_pad
        width = left_pad + inner_w + right_pad
        blue_top = top_pad
        blue_left = left_pad + random.randint(0, inner_w - blue_w)

    grid = np.zeros((height, width), dtype=int)
    grid[
        blue_top : blue_top + blue_h,
        blue_left : blue_left + blue_w,
    ] = blue_patch
    _draw_guides(
        grid,
        orientation,
        location,
        guide_len,
        bar_gap,
        top_pad,
        left_pad,
        blue_top,
        blue_left,
        (blue_h, blue_w),
        gap,
        inner_h if location == "right" else blue_h,
        blue_w if location == "right" else inner_w,
    )
    if len({mask for mask, _combo in _all_exact_sprite_covers(grid == BLUE)}) != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 9")
