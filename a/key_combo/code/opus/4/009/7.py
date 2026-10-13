from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
PALETTE = [2, 3, 4, 5, 6, 7, 8, 9]
REPEAT_OPTIONS = ((2, 2), (2, 3), (3, 2))
GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    box = _bbox(grid != BLACK)
    assert box is not None, "expected a non-black object"
    return _crop_bbox(grid, box)


def _rotate_cw(grid: np.ndarray, quarter_turns: int) -> np.ndarray:
    k = int(quarter_turns) % 4
    return np.rot90(grid, -k)


def _tile_fill(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    h, w = map(int, shape)
    th, tw = tile.shape
    return np.tile(tile, (h // th, w // tw))


def _split_sections(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == BLUE)]
    assert len(sep_rows) == 2, "expected exactly two full-width blue separator rows"
    top_end, mid_end = sep_rows
    assert 0 < top_end < mid_end - 1 < g.shape[0] - 1, "invalid section layout"

    top = g[:top_end, :]
    middle = g[top_end + 1 : mid_end, :]
    bottom = g[mid_end + 1 :, :]
    assert top.size > 0 and middle.size > 0 and bottom.shape[0] == 3, "unexpected section sizes"
    return top, middle, bottom


def _find_clock_patch(clock_band: np.ndarray) -> np.ndarray:
    h, w = clock_band.shape
    assert h == 3 and w >= 3, "clock band must be 3 rows high"
    c0 = (w - 3) // 2
    patch = clock_band[:, c0 : c0 + 3].copy()
    pts = [tuple(map(int, rc)) for rc in np.argwhere(patch != BLACK)]
    assert int(patch[1, 1]) != BLACK and len(pts) == 2 and (1, 1) in pts, "invalid centered 3x3 clock patch"
    return patch


def _read_clock_rotation(clock_band: np.ndarray) -> int:
    patch = _find_clock_patch(clock_band)
    pts = [tuple(map(int, rc)) for rc in np.argwhere(patch != BLACK)]
    other = pts[0] if pts[1] == (1, 1) else pts[1]
    mapping = {
        (0, 1): 0,  # up
        (1, 2): 1,  # right
        (2, 1): 2,  # down
        (1, 0): 3,  # left
    }
    assert other in mapping, "invalid clock-hand direction"
    return mapping[other]


def _ideal_periodic_region(top_shape: tuple[int, int], reference_band: np.ndarray, clock_band: np.ndarray) -> np.ndarray:
    reference = _crop_nonzero(reference_band)
    turns = _read_clock_rotation(clock_band)
    corrected_tile = _rotate_cw(reference, turns)
    h, w = top_shape
    th, tw = corrected_tile.shape
    assert h % th == 0 and w % tw == 0, "top section must be tiled exactly by the corrected tile"
    return _tile_fill(top_shape, corrected_tile)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a noisy periodic region using a rotated reference tile chosen by a clock-hand code.

    Input structure:
    - Two full-width blue separator rows split the grid into three horizontal sections.
    - The top section is a periodic tiling of one small tile, but a few cells are corrupted.
    - The middle section contains one intact reference tile on black background.
    - The bottom section is exactly 3 rows high and contains one 3x3 clock-hand pattern on black background.

    Rule:
    - Read the clock-hand direction as a clockwise rotation: up=0, right=90, down=180, left=270.
    - Crop the middle-section reference tile to its minimal bounding box.
    - Rotate that cropped tile by the encoded amount, then tile it from the top-left across the full top section.
    - Use that clean periodic tiling as the repaired version of the top section.

    Key insight:
    - The top pattern is intentionally ambiguous on its own; only the middle tile plus the clock rotation uniquely determine the correct repair.
    """
    top, middle, bottom = _split_sections(np.asarray(grid, dtype=int))
    return _ideal_periodic_region(top.shape, middle, bottom)


def _has_smaller_period(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for ph in range(1, h + 1):
        if h % ph != 0:
            continue
        for pw in range(1, w + 1):
            if w % pw != 0:
                continue
            if ph == h and pw == w:
                continue
            seed = tile[:ph, :pw]
            if np.array_equal(np.tile(seed, (h // ph, w // pw)), tile):
                return True
    return False


def _half_turn_pairs(size: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs = []
    for r in range(size):
        for c in range(size):
            rr = size - 1 - r
            cc = size - 1 - c
            if (r, c) > (rr, cc):
                continue
            if (r, c) == (rr, cc):
                continue
            pairs.append(((r, c), (rr, cc)))
    return pairs


def _generate_tile() -> np.ndarray | None:
    size = random.choice((3, 4))
    palette = random.sample(PALETTE, k=random.randint(3, 4))

    tile = np.zeros((size, size), dtype=int)
    for (r0, c0), (r1, c1) in _half_turn_pairs(size):
        color = random.choice(palette)
        tile[r0, c0] = color
        tile[r1, c1] = color
    if size % 2 == 1:
        tile[size // 2, size // 2] = random.choice(palette)

    pair = random.choice(_half_turn_pairs(size))
    mutate_first = random.choice((True, False))
    (mr, mc), (pr, pc) = pair if mutate_first else pair[::-1]
    choices = [color for color in palette if color != int(tile[pr, pc])]
    tile[mr, mc] = random.choice(choices)

    if _has_smaller_period(tile):
        return None
    if any(np.array_equal(tile, _rotate_cw(tile, turns)) for turns in (1, 2, 3)):
        return None
    if int(np.count_nonzero(tile != _rotate_cw(tile, 2))) != 2:
        return None
    if min(int(np.count_nonzero(tile != _rotate_cw(tile, turns))) for turns in (1, 3)) <= 2:
        return None
    if len(set(map(int, tile.flatten()))) < 3:
        return None
    return tile


def _clock_patch(turns: int, color: int) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    patch[1, 1] = int(color)
    direction = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }[int(turns) % 4]
    patch[1 + direction[0], 1 + direction[1]] = int(color)
    return patch


def _rotation_scores(noisy_top: np.ndarray, reference: np.ndarray) -> list[int]:
    scores = []
    for turns in range(4):
        candidate = _tile_fill(noisy_top.shape, _rotate_cw(reference, turns))
        scores.append(int(np.count_nonzero(candidate != noisy_top)))
    return scores


def _build_instance() -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    actual_tile = _generate_tile()
    if actual_tile is None:
        return None

    rep_r, rep_c = random.choice(REPEAT_OPTIONS)
    clean_top = np.tile(actual_tile, (rep_r, rep_c))
    confuser_top = np.tile(_rotate_cw(actual_tile, 2), (rep_r, rep_c))

    diff_coords = [tuple(map(int, rc)) for rc in np.argwhere(clean_top != confuser_top)]
    if len(diff_coords) % 2 != 0:
        return None

    noisy_top = clean_top.copy()
    for r, c in random.sample(diff_coords, len(diff_coords) // 2):
        noisy_top[r, c] = confuser_top[r, c]

    turns = random.choice((1, 2, 3))
    reference = _rotate_cw(actual_tile, -turns)

    width = clean_top.shape[1]
    middle_height = actual_tile.shape[0] + random.randint(0, 2)
    middle = np.zeros((middle_height, width), dtype=int)
    ref_r0 = random.randint(0, middle_height - reference.shape[0])
    ref_c0 = random.randint(0, width - reference.shape[1])
    middle[ref_r0 : ref_r0 + reference.shape[0], ref_c0 : ref_c0 + reference.shape[1]] = reference

    bottom = np.zeros((3, width), dtype=int)
    clock_color = random.choice(PALETTE)
    clock = _clock_patch(turns, clock_color)
    clock_c0 = (width - 3) // 2
    bottom[:, clock_c0 : clock_c0 + 3] = clock

    separator = np.full((1, width), BLUE, dtype=int)
    grid = np.vstack([noisy_top, separator, middle, separator.copy(), bottom])
    return grid, clean_top, reference, turns


def _validate_instance(grid: np.ndarray, clean_top: np.ndarray, reference: np.ndarray, turns: int) -> bool:
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, clean_top):
        return False

    top, middle, bottom = _split_sections(grid)
    if np.array_equal(top, clean_top):
        return False
    if int(np.count_nonzero(top != clean_top)) not in (4, 6):
        return False

    cropped_reference = _crop_nonzero(middle)
    if not np.array_equal(cropped_reference, reference):
        return False
    if _read_clock_rotation(bottom) != int(turns):
        return False

    direct_fill = _tile_fill(top.shape, cropped_reference)
    if np.array_equal(direct_fill, clean_top):
        return False

    scores = _rotation_scores(top, cropped_reference)
    best = min(scores)
    if scores[int(turns)] != best:
        return False
    if scores[(int(turns) + 2) % 4] != best:
        return False
    if sum(score == best for score in scores) != 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        grid, clean_top, reference, turns = built
        if _validate_instance(grid, clean_top, reference, turns):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
