from __future__ import annotations

import numpy as np


BG = 0
RED = 2
YELLOW = 4
TEAL = 8
TILE_COLOR_POOL = (1, 3, 6, 7, 9)
GEN_MAX_ATTEMPTS = 400


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _path_mask(red: tuple[int, int], teal: tuple[int, int]) -> np.ndarray:
    rr, rc = red
    tr, tc = teal
    r0, r1 = sorted((rr, tr))
    c0, c1 = sorted((rc, tc))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)

    rr -= r0
    rc -= c0
    tr -= r0
    tc -= c0

    c_step = 1 if rc <= tc else -1
    for c in range(rc, tc + c_step, c_step):
        mask[rr, c] = True

    r_step = 1 if rr <= tr else -1
    for r in range(rr, tr + r_step, r_step):
        mask[r, tc] = True

    return mask


def _extract_tile(grid: np.ndarray) -> np.ndarray:
    tile_mask = (grid != BG) & (grid != RED) & (grid != TEAL)
    bbox = _bbox_from_mask(tile_mask)
    tile = _crop_bbox(grid, bbox)
    colors = sorted(set(int(x) for x in tile.flatten()) - {BG})
    assert tile.shape[0] == tile.shape[1], "tile must be square"
    assert len(colors) >= 2, "tile must be polychrome"
    assert np.all(tile != BG), "tile should be a solid square patch"
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw an L-shaped path between the red and teal endpoints, use that path as a control sprite, and upscale it by tiling.

    Input structure:
    - The grid is a black background with exactly one red pixel and one teal pixel, not sharing a row or column.
    - A separate multicolor square tile appears elsewhere on the grid and is disconnected from both endpoints.
    - The endpoints define a right-angle path whose bounding box becomes the control region.

    Rule:
    - Draw yellow horizontally from the red pixel to the teal column, then vertically along the teal column to the teal row; keep the endpoints as red and teal while forming the path mask.
    - Crop the path's bounding box and treat every path/end-point cell inside it as filled and every black cell as empty.
    - Replace each filled mask cell with a full copy of the multicolor tile, and each empty mask cell with a same-sized black block.

    Key insight:
    - The visible output is not the path itself; the path only defines a binary L-shaped control sprite that gets expanded tile-by-tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    reds = np.argwhere(g == RED)
    teals = np.argwhere(g == TEAL)
    assert reds.shape == (1, 2), "expected exactly one red endpoint"
    assert teals.shape == (1, 2), "expected exactly one teal endpoint"

    red = tuple(map(int, reds[0]))
    teal = tuple(map(int, teals[0]))
    assert red[0] != teal[0] and red[1] != teal[1], "endpoints must form an L"

    tile = _extract_tile(g)
    mask = _path_mask(red, teal)

    th, tw = map(int, tile.shape)
    mh, mw = map(int, mask.shape)
    out = np.full((mh * th, mw * tw), BG, dtype=int)

    for r, c in np.argwhere(mask):
        _paste(out, tile, int(r) * th, int(c) * tw)

    return out


def _make_tile(rng: np.random.Generator) -> np.ndarray:
    size = int(rng.choice([2, 3, 4]))
    color_count = int(rng.choice([2, 3]))
    colors = np.array(rng.choice(TILE_COLOR_POOL, size=color_count, replace=False), dtype=int)
    rr, cc = np.indices((size, size))

    for _ in range(50):
        a = int(rng.integers(1, 4))
        b = int(rng.integers(1, 4))
        bias = int(rng.integers(0, color_count))
        idx = (a * rr + b * cc + bias + (rr == cc) + 2 * (rr + cc >= size)) % color_count
        tile = colors[idx]
        if len(np.unique(tile)) == color_count:
            return tile

    return colors[(rr + cc) % color_count]


def _split_extra(total: int, rng: np.random.Generator) -> tuple[int, int]:
    left = int(rng.integers(0, total + 1))
    return left, total - left


def _choose_layout(
    box_h: int, box_w: int, tile_size: int, rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    relation = str(rng.choice(["left", "right", "above", "below"]))
    gap = int(rng.integers(2, 5))

    if relation in {"left", "right"}:
        min_h = max(box_h, tile_size)
        extra_h = int(rng.integers(4, 9))
        top, bottom = _split_extra(extra_h, rng)
        usable_h = min_h
        height = min_h + extra_h

        tile_r = top + int(rng.integers(0, usable_h - tile_size + 1))
        box_r = top + int(rng.integers(0, usable_h - box_h + 1))

        min_w = tile_size + gap + box_w
        extra_w = int(rng.integers(4, 9))
        left, right = _split_extra(extra_w, rng)
        width = min_w + extra_w

        if relation == "left":
            tile_c = left
            box_c = left + tile_size + gap
        else:
            box_c = left
            tile_c = left + box_w + gap

    else:
        min_w = max(box_w, tile_size)
        extra_w = int(rng.integers(4, 9))
        left, right = _split_extra(extra_w, rng)
        usable_w = min_w
        width = min_w + extra_w

        tile_c = left + int(rng.integers(0, usable_w - tile_size + 1))
        box_c = left + int(rng.integers(0, usable_w - box_w + 1))

        min_h = tile_size + gap + box_h
        extra_h = int(rng.integers(4, 9))
        top, bottom = _split_extra(extra_h, rng)
        height = min_h + extra_h

        if relation == "above":
            tile_r = top
            box_r = top + tile_size + gap
        else:
            box_r = top
            tile_r = top + box_h + gap

    assert height <= 30 and width <= 30
    return (height, width), (box_r, box_c), (tile_r, tile_c)


def _place_endpoints(
    box_r: int, box_c: int, box_h: int, box_w: int, rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int]]:
    row_sign = int(rng.choice([-1, 1]))
    col_sign = int(rng.choice([-1, 1]))

    red_r = box_r if row_sign > 0 else box_r + box_h - 1
    teal_r = box_r + box_h - 1 if row_sign > 0 else box_r
    red_c = box_c if col_sign > 0 else box_c + box_w - 1
    teal_c = box_c + box_w - 1 if col_sign > 0 else box_c

    return (red_r, red_c), (teal_r, teal_c)


def _oracle_ok(grid: np.ndarray, box_h: int, box_w: int, tile_size: int) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    return out.shape == (box_h * tile_size, box_w * tile_size) and max(out.shape) <= 30


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    tile = _make_tile(rng)
    tile_size = int(tile.shape[0])
    max_box = min(8, 30 // tile_size)
    if max_box < 3:
        return None

    box_h = int(rng.integers(3, max_box + 1))
    box_w = int(rng.integers(3, max_box + 1))
    canvas_shape, (box_r, box_c), (tile_r, tile_c) = _choose_layout(box_h, box_w, tile_size, rng)
    red, teal = _place_endpoints(box_r, box_c, box_h, box_w, rng)

    grid = np.full(canvas_shape, BG, dtype=int)
    _paste(grid, tile, tile_r, tile_c)
    grid[red] = RED
    grid[teal] = TEAL

    if abs(red[0] - teal[0]) < 2 or abs(red[1] - teal[1]) < 2:
        return None

    tile_bbox = (tile_r, tile_c, tile_r + tile_size - 1, tile_c + tile_size - 1)
    path_bbox = (box_r, box_c, box_r + box_h - 1, box_c + box_w - 1)
    if not (
        tile_bbox[2] + 1 < path_bbox[0]
        or path_bbox[2] + 1 < tile_bbox[0]
        or tile_bbox[3] + 1 < path_bbox[1]
        or path_bbox[3] + 1 < tile_bbox[1]
    ):
        return None

    tile_colors = sorted(set(int(x) for x in tile.flatten()))
    if len(tile_colors) < 2 or any(color in {BG, RED, YELLOW, TEAL} for color in tile_colors):
        return None

    if not _oracle_ok(grid, box_h, box_w, tile_size):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
