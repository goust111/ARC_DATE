from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import colors_present, crop_bbox, find_components, get_bg, is_contiguous, paste


BG = 0
GEN_MAX_ATTEMPTS = 500
GEN_CTRL_FILL_MIN = 0.35
GEN_CTRL_FILL_MAX = 0.72


def _component_info(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> dict:
    patch = crop_bbox(grid, bbox)
    fg_colors = colors_present(patch, exclude={bg})
    h, w = patch.shape
    filled = int(np.count_nonzero(patch != bg))
    return {
        "bbox": bbox,
        "patch": patch,
        "fg_colors": fg_colors,
        "h": int(h),
        "w": int(w),
        "filled": filled,
        "square": bool(h == w),
    }


def _identify_objects(grid: np.ndarray) -> tuple[int, dict, dict]:
    bg = int(get_bg(grid))
    comps = find_components(
        grid,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert len(comps) == 2, "expected exactly two foreground components"

    infos = [_component_info(grid, comp["bbox"], bg) for comp in comps]
    tile_candidates = [info for info in infos if info["square"] and len(info["fg_colors"]) >= 2]
    ctrl_candidates = [info for info in infos if len(info["fg_colors"]) == 1]

    assert len(tile_candidates) == 1, "expected one multicolor square tile"
    assert len(ctrl_candidates) == 1, "expected one monochrome control sprite"

    tile = tile_candidates[0]
    ctrl = ctrl_candidates[0]
    n = tile["h"]
    assert int(tile["w"]) == int(n)
    assert int(n) <= int(ctrl["h"]) and int(n) <= int(ctrl["w"])
    assert ctrl["filled"] >= 2
    return bg, ctrl, tile


def _stamped_output(ctrl_patch: np.ndarray, tile_patch: np.ndarray, bg: int) -> np.ndarray:
    ctrl_color = int(colors_present(ctrl_patch, exclude={bg})[0])
    ctrl_mask = ctrl_patch == ctrl_color
    n = int(tile_patch.shape[0])
    rows, cols = ctrl_patch.shape

    out = np.full((rows * n, cols * n), int(bg), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if not bool(ctrl_mask[r, c]):
                continue
            rr = r * n
            cc = c * n
            paste(out, tile_patch, rr, cc)
            out[rr + (r % n), cc + (c % n)] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a square tile through a monochrome control mask, then punch one position-coded hole in each stamped copy.

    Input structure:
    - A black background contains exactly two disconnected 8-connected foreground objects.
    - One object is a monochrome rectangular control sprite that may contain black holes inside its bbox.
    - The other object is a fully colored square tile that uses multiple non-black colors.

    Rule:
    - Identify the monochrome control sprite and the multicolor square tile.
    - Build an output grid of size (control_rows * tile_size) by (control_cols * tile_size).
    - For each cell (r, c) of the control sprite bbox: if that cell is filled, stamp one copy of the tile into the corresponding output block; if it is black, leave that whole block black.
    - In every stamped tile copy, replace the local pixel at (r mod tile_size, c mod tile_size) with black.

    Key insight:
    - The control sprite decides where tile copies exist, while each copy's single black hole is not intrinsic to the tile: it encodes that copy's array coordinate.
    """
    g = np.asarray(grid, dtype=int)
    bg, ctrl, tile = _identify_objects(g)
    return _stamped_output(ctrl["patch"], tile["patch"], bg)


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], rng: np.random.Generator) -> None:
    r, c = start
    tr, tc = end
    mask[r, c] = True
    while (r, c) != (tr, tc):
        dr = 0 if r == tr else (1 if tr > r else -1)
        dc = 0 if c == tc else (1 if tc > c else -1)
        options: list[tuple[int, int]] = []
        if dr != 0:
            options.append((dr, 0))
        if dc != 0:
            options.append((0, dc))
        if dr != 0 and dc != 0:
            options.append((dr, dc))
        step = options[int(rng.integers(0, len(options)))]
        r += step[0]
        c += step[1]
        mask[r, c] = True


def _frontier8(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    frontier: set[tuple[int, int]] = set()
    for r_raw, c_raw in np.argwhere(mask):
        r = int(r_raw)
        c = int(c_raw)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    frontier.add((nr, nc))
    return list(frontier)


def _make_control_mask(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray | None:
    area = rows * cols
    mask = np.zeros((rows, cols), dtype=bool)

    anchor = (int(rng.integers(0, rows)), int(rng.integers(0, cols)))
    edge_cells = [
        (0, int(rng.integers(0, cols))),
        (rows - 1, int(rng.integers(0, cols))),
        (int(rng.integers(0, rows)), 0),
        (int(rng.integers(0, rows)), cols - 1),
    ]
    rng.shuffle(edge_cells)
    for cell in edge_cells:
        _draw_path(mask, cell, anchor, rng)

    base = int(mask.sum())
    lo = max(2, base, int(np.ceil(GEN_CTRL_FILL_MIN * area)))
    hi = min(area - 1, int(np.floor(GEN_CTRL_FILL_MAX * area)))
    if lo > hi:
        return None
    target = int(rng.integers(lo, hi + 1))

    while int(mask.sum()) < target:
        frontier = _frontier8(mask)
        if not frontier:
            return None
        pick = frontier[int(rng.integers(0, len(frontier)))]
        mask[pick] = True

    if not is_contiguous(mask.astype(int), background=0, connectivity=8):
        return None
    if rows > 2 and cols > 2 and bool(np.all(mask[1:-1, 1:-1])):
        return None
    return mask


def _ensure_all_colors(tile: np.ndarray, palette: list[int], rng: np.random.Generator) -> None:
    missing = [color for color in palette if not np.any(tile == color)]
    if not missing:
        return
    slots = [(r, c) for r in range(tile.shape[0]) for c in range(tile.shape[1])]
    rng.shuffle(slots)
    for color, (r, c) in zip(missing, slots):
        tile[r, c] = int(color)


def _make_tile(size: int, palette: list[int], rng: np.random.Generator) -> np.ndarray | None:
    tile = np.zeros((size, size), dtype=int)
    mode = int(rng.integers(0, 3))
    offset = int(rng.integers(0, len(palette)))

    if mode == 0:
        weights = rng.random(len(palette))
        weights /= weights.sum()
        tile[:, :] = rng.choice(np.array(palette, dtype=int), size=(size, size), p=weights)
    elif mode == 1:
        for r in range(size):
            for c in range(size):
                tile[r, c] = int(palette[(r + c + offset) % len(palette)])
    else:
        a = int(rng.choice([1, 2]))
        b = int(rng.choice([1, 2]))
        for r in range(size):
            for c in range(size):
                tile[r, c] = int(palette[(a * r + b * c + offset) % len(palette)])

    flip_count = int(rng.integers(0, size + 1))
    for _ in range(flip_count):
        r = int(rng.integers(0, size))
        c = int(rng.integers(0, size))
        tile[r, c] = int(rng.choice(palette))

    _ensure_all_colors(tile, palette, rng)
    if len(set(map(int, tile.flatten()))) < 2:
        return None
    if len({tuple(map(int, row)) for row in tile}) == 1 and len({tuple(map(int, col)) for col in tile.T}) == 1:
        return None
    return tile


def _sample_control_dims(size: int, rng: np.random.Generator) -> tuple[int, int] | None:
    lower = max(3, int(size))
    upper = min(7, 30 // int(size))
    if lower > upper:
        return None

    choices = list(range(lower, upper + 1))
    larger = [dim for dim in choices if dim > size]
    if larger and rng.random() < 0.6:
        rows = int(rng.choice(larger))
        cols = int(rng.choice(choices))
        if cols == size and rng.random() < 0.5:
            cols = int(rng.choice(larger))
    else:
        rows = int(rng.choice(choices))
        cols = int(rng.choice(choices))
    return rows, cols


def _place_objects(ctrl: np.ndarray, tile: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    ctrl_h, ctrl_w = ctrl.shape
    tile_h, tile_w = tile.shape
    border = int(rng.integers(1, 4))
    gap = int(rng.integers(2, 5))
    horizontal = bool(rng.integers(0, 2))

    if horizontal:
        min_h = max(ctrl_h, tile_h) + 2 * border
        min_w = ctrl_w + tile_w + gap + 2 * border
        max_h = min(30, max(min_h, 18) + 6)
        max_w = min(30, max(min_w, 18) + 6)
        if min_h > max_h or min_w > max_w:
            return None
        H = int(rng.integers(min_h, max_h + 1))
        W = int(rng.integers(min_w, max_w + 1))

        left_is_ctrl = bool(rng.integers(0, 2))
        left_w = ctrl_w if left_is_ctrl else tile_w
        right_w = tile_w if left_is_ctrl else ctrl_w
        slack = W - 2 * border - left_w - right_w - gap
        left_c = border + int(rng.integers(0, slack + 1))
        right_c = left_c + left_w + gap
        left_r = border + int(rng.integers(0, H - 2 * border - (ctrl_h if left_is_ctrl else tile_h) + 1))
        right_r = border + int(rng.integers(0, H - 2 * border - (tile_h if left_is_ctrl else ctrl_h) + 1))

        if left_is_ctrl:
            ctrl_pos = (left_r, left_c)
            tile_pos = (right_r, right_c)
        else:
            tile_pos = (left_r, left_c)
            ctrl_pos = (right_r, right_c)
    else:
        min_h = ctrl_h + tile_h + gap + 2 * border
        min_w = max(ctrl_w, tile_w) + 2 * border
        max_h = min(30, max(min_h, 18) + 6)
        max_w = min(30, max(min_w, 18) + 6)
        if min_h > max_h or min_w > max_w:
            return None
        H = int(rng.integers(min_h, max_h + 1))
        W = int(rng.integers(min_w, max_w + 1))

        top_is_ctrl = bool(rng.integers(0, 2))
        top_h = ctrl_h if top_is_ctrl else tile_h
        bottom_h = tile_h if top_is_ctrl else ctrl_h
        slack = H - 2 * border - top_h - bottom_h - gap
        top_r = border + int(rng.integers(0, slack + 1))
        bottom_r = top_r + top_h + gap
        top_c = border + int(rng.integers(0, W - 2 * border - (ctrl_w if top_is_ctrl else tile_w) + 1))
        bottom_c = border + int(rng.integers(0, W - 2 * border - (tile_w if top_is_ctrl else ctrl_w) + 1))

        if top_is_ctrl:
            ctrl_pos = (top_r, top_c)
            tile_pos = (bottom_r, bottom_c)
        else:
            tile_pos = (top_r, top_c)
            ctrl_pos = (bottom_r, bottom_c)

    grid = np.full((H, W), BG, dtype=int)
    paste(grid, tile, tile_pos[0], tile_pos[1])
    paste(grid, ctrl, ctrl_pos[0], ctrl_pos[1], transparent=BG)
    return grid


def _local_hole_positions(mask: np.ndarray, size: int) -> set[tuple[int, int]]:
    positions = set()
    for r_raw, c_raw in np.argwhere(mask):
        positions.add((int(r_raw) % size, int(c_raw) % size))
    return positions


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    size = int(rng.integers(2, 6))
    dims = _sample_control_dims(size, rng)
    if dims is None:
        return None
    rows, cols = dims

    palette = list(rng.choice(np.arange(1, 10, dtype=int), size=4, replace=False))
    ctrl_color = int(palette[0])
    tile_color_count = int(rng.integers(2, 4))
    tile_colors = [int(color) for color in palette[1 : 1 + tile_color_count]]

    ctrl_mask = _make_control_mask(rows, cols, rng)
    if ctrl_mask is None:
        return None
    if len(_local_hole_positions(ctrl_mask, size)) < min(3, int(ctrl_mask.sum())):
        return None

    tile = _make_tile(size, tile_colors, rng)
    if tile is None:
        return None

    ctrl = np.full((rows, cols), BG, dtype=int)
    ctrl[ctrl_mask] = ctrl_color
    out = _stamped_output(ctrl, tile, BG)
    if max(out.shape) > 30:
        return None

    inp = _place_objects(ctrl, tile, rng)
    if inp is None or max(inp.shape) > 30:
        return None

    try:
        bg, ctrl_info, tile_info = _identify_objects(inp)
    except AssertionError:
        return None

    if bg != BG:
        return None
    if ctrl_info["patch"].shape != ctrl.shape:
        return None
    if tile_info["patch"].shape != tile.shape:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid masked-stamping input")
