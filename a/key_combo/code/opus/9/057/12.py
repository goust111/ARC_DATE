from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

HALF_HEIGHT_RANGE = (4, 7)
WIDTH_RANGE = (6, 12)
DISTURBANCE_COUNT_RANGE = (2, 4)
GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _is_nonadjacent_4(coords: list[tuple[int, int]]) -> bool:
    seen = set(coords)
    for r, c in coords:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in seen:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Build the mirrored reference half and output the cropped mask of lower-half perturbations.

    Input structure:
    - The grid is a dense non-black canvas with even height.
    - The upper half is a multicolor pattern using 3-4 non-black colors other than grey.
    - The lower half should be the vertical reflection of the upper half except for 2-4 grey perturbation cells.

    Rule:
    - Split the grid into upper half U and lower half D along the horizontal midline.
    - Flip U vertically to obtain the undisturbed reference lower half R.
    - Mark the cells where D differs from R, crop that difference mask to its tight bounding box,
      color marked cells grey, and leave the remaining cells black.

    Key insight:
    - The comparison target is not another object already present in the input; it must be reconstructed
      by mirroring the upper half, then compared against the actual lower half to isolate the moved marks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    h, _w = g.shape
    assert h % 2 == 0, "grid height must be even"

    half = h // 2
    upper = g[:half]
    lower = g[half:]
    reference = np.flipud(upper)

    diff = lower != reference
    box = _bbox(diff)
    assert box is not None, "expected at least one perturbation"

    cropped = _crop_bbox(diff, box)
    out = np.zeros(cropped.shape, dtype=int)
    out[cropped] = GREY
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    bounds = [0] + cuts + [total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _expand_block_matrix(
    row_heights: list[int], col_widths: list[int], colors: np.ndarray
) -> np.ndarray:
    h = sum(row_heights)
    w = sum(col_widths)
    grid = np.zeros((h, w), dtype=int)
    r = 0
    for i, rh in enumerate(row_heights):
        c = 0
        for j, cw in enumerate(col_widths):
            grid[r : r + rh, c : c + cw] = int(colors[i, j])
            c += cw
        r += rh
    return grid


def _generate_block_top(half_h: int, w: int, palette: tuple[int, ...]) -> np.ndarray | None:
    row_bands = random.randint(2, min(4, half_h))
    col_bands = random.randint(2, min(4, w))
    row_heights = _random_partition(half_h, row_bands)
    col_widths = _random_partition(w, col_bands)

    for _ in range(100):
        band_colors = np.zeros((row_bands, col_bands), dtype=int)
        for r in range(row_bands):
            for c in range(col_bands):
                choices = list(palette)
                if r > 0 and band_colors[r - 1, c] in choices:
                    choices.remove(int(band_colors[r - 1, c]))
                if c > 0 and band_colors[r, c - 1] in choices:
                    choices.remove(int(band_colors[r, c - 1]))
                if not choices:
                    break
                band_colors[r, c] = random.choice(choices)
            else:
                continue
            break
        else:
            top = _expand_block_matrix(row_heights, col_widths, band_colors)
            if len(set(map(int, top.flatten()))) >= 3 and not np.array_equal(top, np.flipud(top)):
                return top
    return None


def _generate_segment_top(half_h: int, w: int, palette: tuple[int, ...]) -> np.ndarray | None:
    top = np.zeros((half_h, w), dtype=int)
    for r in range(half_h):
        for _ in range(100):
            segments = random.randint(2, min(5, w, len(palette)))
            widths = _random_partition(w, segments)
            colors = random.sample(palette, segments)
            row = np.empty(w, dtype=int)
            c0 = 0
            for width, color in zip(widths, colors):
                row[c0 : c0 + width] = color
                c0 += width
            if r == 0 or not np.array_equal(row, top[r - 1]):
                top[r] = row
                break
        else:
            return None

    if len(set(map(int, top.flatten()))) < 3:
        return None
    if np.array_equal(top, np.flipud(top)):
        return None
    return top


def _generate_top_half(half_h: int, w: int) -> np.ndarray | None:
    color_count = random.randint(3, 4)
    palette = tuple(random.sample(PATTERN_COLORS, color_count))
    makers = [_generate_block_top, _generate_segment_top]
    random.shuffle(makers)

    for maker in makers:
        top = maker(half_h, w, palette)
        if top is not None:
            if _top_has_enough_structure(top):
                return top
    return None


def _top_has_enough_structure(top: np.ndarray) -> bool:
    if len({tuple(map(int, row)) for row in top}) < 2:
        return False
    half_h = top.shape[0]
    mirrored_pairs = sum(
        1 for r in range(half_h // 2) if not np.array_equal(top[r], top[half_h - 1 - r])
    )
    return mirrored_pairs >= 1


def _random_mask_bbox(max_h: int, max_w: int, count: int) -> tuple[int, int]:
    h = random.randint(1, min(max_h, count + 1))
    w = random.randint(1, min(max_w, max(2, count + 2)))
    return h, w


def _generate_disturbance_mask(max_h: int, max_w: int) -> np.ndarray | None:
    count = random.randint(*DISTURBANCE_COUNT_RANGE)
    for _ in range(300):
        mask_h, mask_w = _random_mask_bbox(max_h, max_w, count)
        cells = [(r, c) for r in range(mask_h) for c in range(mask_w)]
        random.shuffle(cells)

        chosen: list[tuple[int, int]] = []
        for cell in cells:
            if all(abs(cell[0] - r) + abs(cell[1] - c) != 1 for r, c in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    break
        if len(chosen) != count:
            continue

        mask = np.zeros((mask_h, mask_w), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True

        if _bbox(mask) != (0, 0, mask_h - 1, mask_w - 1):
            continue
        if _mask_is_too_regular(mask):
            continue
        return mask
    return None


def _mask_is_too_regular(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    if len(coords) == 2 and (len(rows) == 1 or len(cols) == 1):
        return True
    return False


def _validate_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    h, w = grid.shape
    assert h % 2 == 0 and 1 <= h <= 30 and 1 <= w <= 30
    assert np.all(grid != BLACK), "input must be dense non-black"

    half = h // 2
    upper = grid[:half]
    lower = grid[half:]
    upper_colors = set(map(int, upper.flatten()))
    assert GREY not in upper_colors, "grey must be reserved for perturbations"
    assert 3 <= len(upper_colors) <= 4, "upper half must use 3-4 colors"

    reference = np.flipud(upper)
    diff = lower != reference
    coords = [tuple(map(int, rc)) for rc in np.argwhere(diff)]
    assert DISTURBANCE_COUNT_RANGE[0] <= len(coords) <= DISTURBANCE_COUNT_RANGE[1]
    assert _is_nonadjacent_4(coords), "perturbations must not be 4-neighbor adjacent"
    assert np.all(lower[diff] == GREY), "all differences must be grey in the lower half"
    assert np.array_equal(lower[~diff], reference[~diff]), "non-perturbed cells must match the mirror"

    out = main(grid)
    box = _bbox(diff)
    assert box is not None
    expected = np.zeros((box[2] - box[0] + 1, box[3] - box[1] + 1), dtype=int)
    expected[_crop_bbox(diff, box)] = GREY
    assert np.array_equal(out, expected), "output must be the tight cropped difference mask"
    return diff, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        half_h = random.randint(*HALF_HEIGHT_RANGE)
        w = random.randint(*WIDTH_RANGE)

        top = _generate_top_half(half_h, w)
        if top is None:
            continue

        disturbance = _generate_disturbance_mask(half_h, w)
        if disturbance is None:
            continue
        mask_h, mask_w = disturbance.shape
        start_r = random.randint(0, half_h - mask_h)
        start_c = random.randint(0, w - mask_w)

        lower = np.flipud(top).copy()
        window = lower[start_r : start_r + mask_h, start_c : start_c + mask_w]
        window[disturbance] = GREY
        grid = np.vstack([top, lower])

        try:
            diff, out = _validate_grid(grid)
        except AssertionError:
            continue

        if int(diff.sum()) != int(np.sum(out == GREY)):
            continue
        if np.array_equal(out, np.full(out.shape, GREY, dtype=int)):
            continue
        return grid

    raise RuntimeError("failed to generate a valid mirror-difference ARC input")
