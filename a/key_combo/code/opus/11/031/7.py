from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox


OCCLUSION = int(Color.BLACK)
SEED = int(Color.PINK)
PATTERN_COLORS = [int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW), int(Color.ORANGE)]
GEN_ATTEMPTS = 400
TILE_RANGE = (3, 5)
REPEAT_RANGE = (3, 5)
SEED_COLOR_COUNT_RANGE = (3, 4)


def _detect_axis_period(grid: np.ndarray, axis: int, ignore: tuple[int, ...]) -> int:
    length = int(grid.shape[int(axis)])
    ignored = set(map(int, ignore))
    for period in range(1, length):
        if axis == 0:
            a = grid[:-period, :]
            b = grid[period:, :]
        else:
            a = grid[:, :-period]
            b = grid[:, period:]
        mask = (~np.isin(a, list(ignored))) & (~np.isin(b, list(ignored)))
        if mask.any() and np.array_equal(a[mask], b[mask]):
            return int(period)
    raise AssertionError("expected a detectable translational period")


def _detect_periods(grid: np.ndarray) -> tuple[int, int]:
    return _detect_axis_period(grid, 0, (OCCLUSION, SEED)), _detect_axis_period(grid, 1, (OCCLUSION, SEED))


def _restore_pattern(grid: np.ndarray, row_period: int, col_period: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    H, W = out.shape
    for r, c in np.argwhere(grid == OCCLUSION):
        samples = {
            int(grid[rr, cc])
            for rr in range(int(r) % row_period, H, row_period)
            for cc in range(int(c) % col_period, W, col_period)
            if int(grid[rr, cc]) not in (OCCLUSION, SEED)
        }
        assert len(samples) == 1, "each occluded orbit must expose one pattern color"
        out[int(r), int(c)] = samples.pop()
    return out


def _find_hidden_seed(grid: np.ndarray, row_period: int, col_period: int) -> tuple[int, int]:
    visible = np.argwhere(grid == SEED)
    assert visible.shape == (1, 2), "expected exactly one visible seed in the input"
    vr, vc = map(int, visible[0])
    candidates = [
        (int(r), int(c))
        for r, c in np.argwhere(grid == OCCLUSION)
        if int(r) % row_period == vr % row_period and int(c) % col_period == vc % col_period
    ]
    assert len(candidates) == 1, "the hidden seed must be the unique occluded cell in the visible seed's orbit"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the occluded symmetric pattern, then crop the rectangle whose corners are the visible and hidden seed pixels.

    Input structure:
    - The whole grid is tiled by a multi-color translational pattern with no ordinary background.
    - One black rectangle occludes part of the pattern.
    - One visible magenta seed pixel lies outside the occlusion, and one hidden magenta seed lies inside it.

    Rule:
    - Detect the row and column translation periods while ignoring black occlusion pixels and the visible magenta seed.
    - Restore each black cell from the non-black, non-magenta cells in its translation orbit.
    - The hidden seed is the unique occluded cell that shares the visible seed's row-phase and column-phase under those periods; keep both seeds magenta in the restored grid.
    - Use the two seed pixels as opposite corners of an axis-aligned rectangle and output that crop.

    Key insight:
    The occlusion must be repaired with translational orbits first, and the same orbit structure then reveals which covered cell is the hidden seed that determines the crop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    black_mask = g == OCCLUSION
    occ_bbox = bbox(black_mask)
    assert occ_bbox is not None, "expected one black occlusion rectangle"
    r0, c0, r1, c1 = occ_bbox
    assert black_mask[r0 : r1 + 1, c0 : c1 + 1].all(), "the occlusion must be a solid rectangle"

    row_period, col_period = _detect_periods(g)
    restored = _restore_pattern(g, row_period, col_period)
    hidden_r, hidden_c = _find_hidden_seed(g, row_period, col_period)

    visible = np.argwhere(g == SEED)[0]
    vr, vc = map(int, visible)
    restored[vr, vc] = SEED
    restored[hidden_r, hidden_c] = SEED

    crop = (
        min(vr, hidden_r),
        min(vc, hidden_c),
        max(vr, hidden_r),
        max(vc, hidden_c),
    )
    return crop_bbox(restored, crop)


def _make_tile(tile_h: int, tile_w: int, colors: list[int], rep_r: int, rep_c: int) -> np.ndarray | None:
    for _ in range(150):
        tile = np.array(random.choices(colors, k=tile_h * tile_w), dtype=int).reshape(tile_h, tile_w)
        if len(set(map(int, tile.flat))) != len(colors):
            continue
        if any(len(set(map(int, tile[r, :]))) == 1 for r in range(tile_h)):
            continue
        if any(len(set(map(int, tile[:, c]))) == 1 for c in range(tile_w)):
            continue
        pattern = np.tile(tile, (rep_r, rep_c))
        if _detect_axis_period(pattern, 0, ()) != tile_h:
            continue
        if _detect_axis_period(pattern, 1, ()) != tile_w:
            continue
        return tile
    return None


def _choose_seed_pair(tile_h: int, tile_w: int, rep_r: int, rep_c: int) -> tuple[tuple[int, int], tuple[int, int]]:
    row_sign, col_sign = random.choice(((1, 1), (1, -1), (-1, 1), (-1, -1)))
    residue_r = random.randrange(tile_h)
    residue_c = random.randrange(tile_w)

    if row_sign > 0:
        hidden_block_r = random.randrange(1, rep_r)
        visible_block_r = random.randrange(0, hidden_block_r)
    else:
        hidden_block_r = random.randrange(0, rep_r - 1)
        visible_block_r = random.randrange(hidden_block_r + 1, rep_r)

    if col_sign > 0:
        hidden_block_c = random.randrange(1, rep_c)
        visible_block_c = random.randrange(0, hidden_block_c)
    else:
        hidden_block_c = random.randrange(0, rep_c - 1)
        visible_block_c = random.randrange(hidden_block_c + 1, rep_c)

    hidden = (hidden_block_r * tile_h + residue_r, hidden_block_c * tile_w + residue_c)
    visible = (visible_block_r * tile_h + residue_r, visible_block_c * tile_w + residue_c)
    return visible, hidden


def _choose_occlusion(
    shape: tuple[int, int],
    tile_h: int,
    tile_w: int,
    hidden: tuple[int, int],
) -> tuple[int, int, int, int]:
    H, W = map(int, shape)
    occ_h = random.randint(3, min(5, tile_h))
    occ_w = random.randint(3, min(5, tile_w))
    hr, hc = map(int, hidden)
    r0 = random.randint(max(0, hr - occ_h + 1), min(hr, H - occ_h))
    c0 = random.randint(max(0, hc - occ_w + 1), min(hc, W - occ_w))
    return int(r0), int(c0), int(r0 + occ_h - 1), int(c0 + occ_w - 1)


def _scene_expected(full_grid: np.ndarray, visible: tuple[int, int], hidden: tuple[int, int]) -> np.ndarray:
    r0 = min(int(visible[0]), int(hidden[0]))
    c0 = min(int(visible[1]), int(hidden[1]))
    r1 = max(int(visible[0]), int(hidden[0]))
    c1 = max(int(visible[1]), int(hidden[1]))
    return crop_bbox(full_grid, (r0, c0, r1, c1))


def _validate_scene(
    inp: np.ndarray,
    expected: np.ndarray,
    *,
    tile_h: int,
    tile_w: int,
    visible: tuple[int, int],
    hidden: tuple[int, int],
    occ_bbox: tuple[int, int, int, int],
) -> bool:
    try:
        row_period, col_period = _detect_periods(inp)
        if (row_period, col_period) != (tile_h, tile_w):
            return False

        black_mask = inp == OCCLUSION
        if bbox(black_mask) != occ_bbox:
            return False
        r0, c0, r1, c1 = occ_bbox
        if not black_mask[r0 : r1 + 1, c0 : c1 + 1].all():
            return False

        visible_coords = np.argwhere(inp == SEED)
        if visible_coords.shape != (1, 2):
            return False
        if tuple(map(int, visible_coords[0])) != tuple(map(int, visible)):
            return False

        candidates = [
            (int(r), int(c))
            for r, c in np.argwhere(black_mask)
            if int(r) % row_period == visible[0] % row_period and int(c) % col_period == visible[1] % col_period
        ]
        if candidates != [tuple(map(int, hidden))]:
            return False

        restored = _restore_pattern(inp, row_period, col_period)
        restored[visible] = SEED
        restored[hidden] = SEED
        if not np.array_equal(_scene_expected(restored, visible, hidden), expected):
            return False

        occ_crop = crop_bbox(restored, occ_bbox)
        if occ_crop.shape == expected.shape and np.array_equal(occ_crop, expected):
            return False
        if expected.shape[0] <= (r1 - r0 + 1) or expected.shape[1] <= (c1 - c0 + 1):
            return False
        if not np.array_equal(main(inp), expected):
            return False
    except Exception:
        return False
    return True


def _generate_one() -> np.ndarray | None:
    tile_h = random.randint(*TILE_RANGE)
    tile_w = random.randint(*TILE_RANGE)
    rep_r = random.randint(*REPEAT_RANGE)
    rep_c = random.randint(*REPEAT_RANGE)
    H, W = tile_h * rep_r, tile_w * rep_c
    if max(H, W) > 30:
        return None

    color_count = random.randint(*SEED_COLOR_COUNT_RANGE)
    colors = random.sample(PATTERN_COLORS, color_count)
    tile = _make_tile(tile_h, tile_w, colors, rep_r, rep_c)
    if tile is None:
        return None

    pattern = np.tile(tile, (rep_r, rep_c))
    visible, hidden = _choose_seed_pair(tile_h, tile_w, rep_r, rep_c)
    occ_bbox = _choose_occlusion(pattern.shape, tile_h, tile_w, hidden)
    r0, c0, r1, c1 = occ_bbox

    full = pattern.copy()
    full[visible] = SEED
    full[hidden] = SEED
    inp = full.copy()
    inp[r0 : r1 + 1, c0 : c1 + 1] = OCCLUSION
    if inp[visible] != SEED:
        return None
    if (r0 <= visible[0] <= r1) and (c0 <= visible[1] <= c1):
        return None

    expected = _scene_expected(full, visible, hidden)
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if expected.shape[0] < 4 or expected.shape[1] < 4:
        return None
    if len(set(map(int, expected.flat))) < 3:
        return None

    if not _validate_scene(
        inp,
        expected,
        tile_h=tile_h,
        tile_w=tile_w,
        visible=visible,
        hidden=hidden,
        occ_bbox=occ_bbox,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-seed crop scene")
