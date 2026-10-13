from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, label_mask, tile_fill


GREY = Color.GREY
GREEN = Color.GREEN
YELLOW = Color.YELLOW
ORANGE = Color.ORANGE
BLUE = Color.BLUE
BLACK = Color.BLACK

MASK_W_RANGE = (9, 12)
MASK_H_RANGE = (7, 10)
SEED_DIMS = ((3, 3), (3, 4), (4, 3), (4, 4))
TOP_GAP_RANGE = (1, 3)
BOTTOM_MARGIN_RANGE = (0, 2)
LEFT_RIGHT_AREA_RANGE = (5, 10)
GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 300


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and label_mask(mask, connectivity=4)[1] == 1


def _bbox_border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((h, w), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    return out


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    separator_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(separator_cols) == 1, "expected one full grey separator column"
    sep = int(separator_cols[0])

    frame_mask = g == ORANGE
    frame_bbox = bbox(frame_mask)
    assert frame_bbox is not None, "expected one orange frame"
    fr0, fc0, fr1, fc1 = frame_bbox
    assert fr1 - fr0 >= 2 and fc1 - fc0 >= 2, "frame must enclose a non-empty cavity"

    frame_crop = crop_bbox(frame_mask, frame_bbox)
    border = _bbox_border_mask(frame_crop.shape)
    assert np.array_equal(frame_crop, border), "orange pixels must form a 1-cell rectangular frame"

    cavity = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    assert cavity.size > 0 and np.all(cavity == BLACK), "frame cavity must start black"

    half_w = min(sep, w - sep - 1)
    assert half_w > 0, "separator must leave equal-width areas on both sides"
    left = g[:fr0, sep - half_w : sep]
    right = g[:fr0, sep + 1 : sep + 1 + half_w]
    assert left.shape == right.shape, "aligned mask regions must have the same shape"

    left_fg = left == GREEN
    right_fg = right == YELLOW
    diff = np.logical_xor(left_fg, right_fg)
    diff_bbox = bbox(diff)

    if diff_bbox is None:
        seed_tile = np.zeros((1, 1), dtype=int)
        seed_mask = np.zeros((1, 1), dtype=bool)
    else:
        seed_mask = crop_bbox(diff, diff_bbox)
        seed_tile = np.where(seed_mask, BLUE, BLACK).astype(int)

    return {
        "grid": g,
        "separator_col": sep,
        "frame_bbox": frame_bbox,
        "cavity_bbox": (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1),
        "left_fg": left_fg,
        "right_fg": right_fg,
        "diff_mask": diff,
        "diff_bbox": diff_bbox,
        "seed_mask": seed_mask,
        "seed_tile": seed_tile,
    }


def _fill_from_tile(grid: np.ndarray, cavity_bbox: tuple[int, int, int, int], tile: np.ndarray) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = cavity_bbox
    cavity_h = int(r1 - r0 + 1)
    cavity_w = int(c1 - c0 + 1)
    tile_h, tile_w = map(int, tile.shape)
    assert tile_h > 0 and tile_w > 0
    assert cavity_h % tile_h == 0 and cavity_w % tile_w == 0
    out[r0 : r1 + 1, c0 : c1 + 1] = tile_fill((cavity_h, cavity_w), tile, offset=(0, 0))
    return out


def _tile_from_mask(mask: np.ndarray, color: int = BLUE) -> np.ndarray | None:
    b = bbox(mask)
    if b is None:
        return None
    return np.where(crop_bbox(mask, b), int(color), BLACK).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """Reveal the hidden xor seed from the two upper masks and tile it across the lower framed cavity.

    Input structure:
    - A black canvas contains one full grey separator column and one orange rectangular frame with a black interior.
    - Above the frame, equal-width regions on the two sides of the separator contain a green connected mask on the left and a yellow connected mask on the right.
    - The two masks overlap in some cells and differ in others; the cavity lies below them and does not overlap the separator.

    Rule:
    - Align the two upper side regions cell by cell across the separator and mark exactly the cells where one side has foreground and the other side has black background.
    - Take the bounding box of those xor-difference cells as the seed tile; inside that box paint difference cells blue and leave non-difference cells black.
    - Tile that seed from the cavity's top-left corner across the entire framed interior, preserving black gaps inside the seed and leaving the separator, frame, and original upper masks unchanged.

    Key insight:
    - The fill pattern is not drawn explicitly anywhere in the input; it emerges only after xor-ing the two aligned masks, and the tiled cavity exposes that hidden seed.
    """
    scene = _parse_scene(grid)
    if scene["diff_bbox"] is None:
        return scene["grid"].copy()
    return _fill_from_tile(scene["grid"], scene["cavity_bbox"], scene["seed_tile"])


def _neighbors4(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _random_connected_mask(rng: np.random.Generator, h: int, w: int, size: int) -> np.ndarray:
    cells = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells[start] = True
    while int(cells.sum()) < size:
        frontier = sorted(
            {
                nbr
                for r, c in np.argwhere(cells)
                for nbr in _neighbors4(h, w, int(r), int(c))
                if not cells[nbr]
            }
        )
        assert frontier
        rr, cc = frontier[int(rng.integers(0, len(frontier)))]
        cells[rr, cc] = True
    return cells


def _grow_connected_superset(
    rng: np.random.Generator,
    base: np.ndarray,
    target_size: int,
    forbidden: np.ndarray | None = None,
) -> np.ndarray | None:
    h, w = base.shape
    out = base.copy()
    blocked = np.zeros_like(base, dtype=bool) if forbidden is None else forbidden.copy()
    blocked &= ~base

    while int(out.sum()) < target_size:
        frontier = sorted(
            {
                nbr
                for r, c in np.argwhere(out)
                for nbr in _neighbors4(h, w, int(r), int(c))
                if not out[nbr] and not blocked[nbr]
            }
        )
        if not frontier:
            return None
        rr, cc = frontier[int(rng.integers(0, len(frontier)))]
        out[rr, cc] = True
    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _generate_seed_pair(rng: np.random.Generator, seed_h: int, seed_w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    area = seed_h * seed_w
    for _ in range(MASK_ATTEMPTS):
        shared_size_max = min(3, area - 8)
        if shared_size_max < 1:
            return None
        shared_size = int(rng.integers(1, shared_size_max + 1))
        shared = _random_connected_mask(rng, seed_h, seed_w, shared_size)

        feasible: list[tuple[int, int]] = []
        for left_area in range(max(shared_size + 1, LEFT_RIGHT_AREA_RANGE[0]), min(LEFT_RIGHT_AREA_RANGE[1], area - 1) + 1):
            for right_area in range(max(shared_size + 1, LEFT_RIGHT_AREA_RANGE[0]), min(LEFT_RIGHT_AREA_RANGE[1], area - 1) + 1):
                if left_area + right_area - shared_size <= area:
                    feasible.append((left_area, right_area))
        if not feasible:
            return None

        left_area, right_area = feasible[int(rng.integers(0, len(feasible)))]
        left = _grow_connected_superset(rng, shared, left_area)
        if left is None:
            continue

        forbidden = left & ~shared
        right = _grow_connected_superset(rng, shared, right_area, forbidden=forbidden)
        if right is None:
            continue

        diff = np.logical_xor(left, right)
        if not diff.any() or diff.all():
            continue
        if not _touches_all_sides(diff):
            continue
        if not (_is_connected(left) and _is_connected(right)):
            continue
        if np.array_equal(left, diff) or np.array_equal(right, diff):
            continue
        return left, right, diff
    return None


def _render_input(
    rng: np.random.Generator,
    mask_h: int,
    mask_w: int,
    top_gap: int,
    bottom_margin: int,
    seed_h: int,
    seed_w: int,
    rep_h: int,
    rep_w: int,
    seed_row: int,
    seed_col: int,
    left_seed: np.ndarray,
    right_seed: np.ndarray,
    frame_side: str,
    frame_outer_margin: int,
) -> np.ndarray:
    sep = mask_w
    cavity_h = seed_h * rep_h
    cavity_w = seed_w * rep_w
    frame_h = cavity_h + 2
    frame_w = cavity_w + 2
    height = mask_h + top_gap + frame_h + bottom_margin
    width = mask_w * 2 + 1
    grid = np.zeros((height, width), dtype=int)
    grid[:, sep] = GREY

    grid[seed_row : seed_row + seed_h, seed_col : seed_col + seed_w] = np.where(left_seed, GREEN, BLACK)
    right_c0 = sep + 1 + seed_col
    grid[seed_row : seed_row + seed_h, right_c0 : right_c0 + seed_w] = np.where(right_seed, YELLOW, BLACK)

    frame_r0 = mask_h + top_gap
    frame_r1 = frame_r0 + frame_h - 1
    if frame_side == "left":
        c0_min = frame_outer_margin
        c0_max = sep - frame_w
    else:
        c0_min = sep + 1
        c0_max = width - frame_w - frame_outer_margin
    assert c0_min <= c0_max
    frame_c0 = int(c0_min)
    if c0_min != c0_max:
        frame_c0 = int(rng.integers(c0_min, c0_max + 1))
    frame_c1 = frame_c0 + frame_w - 1

    grid[frame_r0 : frame_r1 + 1, frame_c0] = ORANGE
    grid[frame_r0 : frame_r1 + 1, frame_c1] = ORANGE
    grid[frame_r0, frame_c0 : frame_c1 + 1] = ORANGE
    grid[frame_r1, frame_c0 : frame_c1 + 1] = ORANGE
    return grid


def _validate_semantics(grid: np.ndarray) -> None:
    scene = _parse_scene(grid)
    diff_bbox = scene["diff_bbox"]
    assert diff_bbox is not None, "generator must keep xor non-empty"

    left_fg = scene["left_fg"]
    right_fg = scene["right_fg"]
    assert _is_connected(left_fg), "left mask must be 4-connected"
    assert _is_connected(right_fg), "right mask must be 4-connected"
    assert LEFT_RIGHT_AREA_RANGE[0] <= int(left_fg.sum()) <= LEFT_RIGHT_AREA_RANGE[1]
    assert LEFT_RIGHT_AREA_RANGE[0] <= int(right_fg.sum()) <= LEFT_RIGHT_AREA_RANGE[1]
    assert (left_fg & right_fg).any(), "masks must overlap"
    assert np.logical_xor(left_fg, right_fg).any(), "masks must differ"

    seed_mask = scene["seed_mask"]
    assert not seed_mask.all(), "xor seed must keep black holes inside its bbox"
    assert _touches_all_sides(seed_mask), "xor bbox must be tight"

    cr0, cc0, cr1, cc1 = scene["cavity_bbox"]
    cavity_h = cr1 - cr0 + 1
    cavity_w = cc1 - cc0 + 1
    tile_h, tile_w = seed_mask.shape
    assert cavity_h % tile_h == 0 and cavity_w % tile_w == 0
    assert cavity_h > tile_h or cavity_w > tile_w, "tiling must repeat along at least one axis"

    actual = main(grid)
    expected = _fill_from_tile(scene["grid"], scene["cavity_bbox"], scene["seed_tile"])
    assert np.array_equal(actual, expected), "main() must implement the intended rule"
    assert not np.array_equal(actual, grid), "output must change on generated examples"

    alt_masks = [
        _tile_from_mask(left_fg),
        _tile_from_mask(right_fg),
        _tile_from_mask(left_fg | right_fg),
    ]
    for alt in alt_masks:
        if alt is None:
            continue
        ah, aw = alt.shape
        if cavity_h % ah == 0 and cavity_w % aw == 0:
            wrong = _fill_from_tile(scene["grid"], scene["cavity_bbox"], alt)
            assert not np.array_equal(actual, wrong), "generator must reject shortcut-equivalent examples"


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    mask_w = int(rng.integers(MASK_W_RANGE[0], MASK_W_RANGE[1] + 1))
    mask_h = int(rng.integers(MASK_H_RANGE[0], MASK_H_RANGE[1] + 1))
    seed_h, seed_w = SEED_DIMS[int(rng.integers(0, len(SEED_DIMS)))]

    pair = _generate_seed_pair(rng, seed_h, seed_w)
    if pair is None:
        return None
    left_seed, right_seed, _diff = pair

    max_rep_w = (mask_w - 3) // seed_w
    max_rep_h = 12 // seed_h
    if max_rep_w < 1 or max_rep_h < 1:
        return None

    rep_w_choices = list(range(1, max_rep_w + 1))
    rep_h_choices = list(range(1, max_rep_h + 1))
    rep_pairs = [(rh, rw) for rh in rep_h_choices for rw in rep_w_choices if rh > 1 or rw > 1]
    if not rep_pairs:
        return None
    rep_h, rep_w = rep_pairs[int(rng.integers(0, len(rep_pairs)))]

    seed_row = int(rng.integers(0, mask_h - seed_h + 1))
    seed_col = int(rng.integers(0, mask_w - seed_w + 1))

    top_gap = int(rng.integers(TOP_GAP_RANGE[0], TOP_GAP_RANGE[1] + 1))
    bottom_margin = int(rng.integers(BOTTOM_MARGIN_RANGE[0], BOTTOM_MARGIN_RANGE[1] + 1))
    frame_side = "left" if int(rng.integers(0, 2)) == 0 else "right"
    frame_outer_margin = int(rng.integers(0, 2))

    cavity_w = seed_w * rep_w
    frame_w = cavity_w + 2
    if frame_side == "left":
        if frame_w > mask_w:
            return None
    else:
        if frame_w + frame_outer_margin > mask_w:
            return None

    grid = _render_input(
        rng=rng,
        mask_h=mask_h,
        mask_w=mask_w,
        top_gap=top_gap,
        bottom_margin=bottom_margin,
        seed_h=seed_h,
        seed_w=seed_w,
        rep_h=rep_h,
        rep_w=rep_w,
        seed_row=seed_row,
        seed_col=seed_col,
        left_seed=left_seed,
        right_seed=right_seed,
        frame_side=frame_side,
        frame_outer_margin=frame_outer_margin,
    )

    if max(grid.shape) > 30:
        return None

    try:
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
