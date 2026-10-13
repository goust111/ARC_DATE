from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, crop_bbox, get_bg, paste


BG = 0
CONTROL_COLOR = 2
SPRITE_COLORS = [1, 3, 4, 6, 7, 8, 9]


def _dense_tile_from_sparse(coords: np.ndarray, color: int, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bbox = bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    rel = coords - np.array([r0, c0], dtype=int)
    if np.any(rel % 2 != 0):
        raise ValueError("sparse sprite must lie on a step-2 lattice")

    tile_h = (r1 - r0) // 2 + 1
    tile_w = (c1 - c0) // 2 + 1
    if tile_h != tile_w:
        raise ValueError("densified sparse sprites must become square tiles")

    tile = np.full((tile_h, tile_w), int(bg), dtype=int)
    for dr, dc in rel:
        tile[int(dr) // 2, int(dc) // 2] = int(color)
    return tile, bbox


def _extract_instance(grid: np.ndarray) -> tuple[int, np.ndarray, list[np.ndarray], list[tuple[int, int, int, int]], list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    control_coords = np.argwhere(g == CONTROL_COLOR)
    if control_coords.size == 0:
        raise ValueError("missing red control sprite")
    control_bbox = bbox_coords(control_coords)
    control = crop_bbox(g, control_bbox)

    sprite_colors = [c for c in colors_present(g, exclude={bg, CONTROL_COLOR}) if c in SPRITE_COLORS]
    if not (2 <= len(sprite_colors) <= 4):
        raise ValueError("expected 2-4 sparse sprites in distinct non-red colors")

    tiles: list[np.ndarray] = []
    bboxes: list[tuple[int, int, int, int]] = []
    for color in sprite_colors:
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        tile, bbox = _dense_tile_from_sparse(coords, color, bg)
        tiles.append(tile)
        bboxes.append(bbox)

    if len(tiles) != len(sprite_colors):
        raise ValueError("failed to recover all sparse sprites")

    tile_shapes = {tuple(tile.shape) for tile in tiles}
    if len(tile_shapes) != 1:
        raise ValueError("all densified tiles must share one size")

    order = np.argsort([(b[0], b[1]) for b in bboxes], axis=0)
    # np.argsort on tuples is awkward; use Python sort for stable row-major order.
    order = sorted(range(len(bboxes)), key=lambda i: (bboxes[i][0], bboxes[i][1]))
    ordered_tiles = [tiles[i] for i in order]
    ordered_bboxes = [bboxes[i] for i in order]
    ordered_colors = [sprite_colors[i] for i in order]
    return bg, control, ordered_tiles, ordered_bboxes, ordered_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Densify several sparse sprites into square tiles and place them into the control mask in row-major order.

    Input structure:
    - Black background with one small red control sprite near the upper-right; its bounding box contains both red cells and background holes.
    - Two to four larger sparse monochrome sprites elsewhere, each in a different non-red color.
    - Every sparse sprite lies on a step-2 lattice, so its foreground cells are separated by one background cell horizontally and vertically.

    Rule:
    - Convert each sparse sprite into a dense tile by collapsing its step-2 lattice to step-1 while preserving the sprite's foreground pattern and color.
    - Sort those tiles by the top-left position of the original sparse sprites, first by row then by column.
    - Scan the red control sprite's bounding box in row-major order; each red cell receives the next tile, and each background cell expands to an equally sized blank tile region.
    - Return only the enlarged tiled canvas whose size is `(control_h * tile_h, control_w * tile_w)`.

    Key insight:
    - The visible tile content is hidden inside the sparse sprites, so solving requires densifying them first and then using source position order, not color order or tile statistics, to map them onto the control mask slots.
    """
    bg, control, tiles, _bboxes, _colors = _extract_instance(grid)
    tile_h, tile_w = map(int, tiles[0].shape)
    ctrl_h, ctrl_w = map(int, control.shape)

    if int(np.sum(control == CONTROL_COLOR)) != len(tiles):
        raise ValueError("control fill count must equal sparse sprite count")

    out = np.full((ctrl_h * tile_h, ctrl_w * tile_w), int(bg), dtype=int)
    tile_idx = 0
    for r in range(ctrl_h):
        for c in range(ctrl_w):
            if int(control[r, c]) == CONTROL_COLOR:
                paste(out, tiles[tile_idx], r * tile_h, c * tile_w)
                tile_idx += 1
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    return bbox_coords(coords)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _make_logical_mask(size: int, area: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(200):
        mask = np.zeros((size, size), dtype=bool)
        start = (int(rng.integers(size)), int(rng.integers(size)))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                frontier.extend((rr, cc) for rr, cc in _neighbors4(int(r), int(c), size, size) if not mask[rr, cc])
            if not frontier:
                break
            rr, cc = frontier[int(rng.integers(len(frontier)))]
            mask[rr, cc] = True
        if int(mask.sum()) != area:
            continue
        r0, c0, r1, c1 = _bbox(mask)
        if r0 == 0 and c0 == 0 and r1 == size - 1 and c1 == size - 1 and area < size * size:
            return mask
    return None


def _make_sparse_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    size = int(mask.shape[0])
    patch = np.full((2 * size - 1, 2 * size - 1), int(bg), dtype=int)
    for r, c in np.argwhere(mask):
        patch[2 * int(r), 2 * int(c)] = int(color)
    return patch


def _row_major_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = [tuple(map(int, p)) for p in np.argwhere(mask)]
    return sorted(pts)


def _col_major_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = [tuple(map(int, p)) for p in np.argwhere(mask)]
    return sorted(pts, key=lambda rc: (rc[1], rc[0]))


def _make_control_patch(count: int, rng: np.random.Generator) -> np.ndarray | None:
    shapes = [(3, 4), (4, 4), (4, 5), (5, 5)]
    rng.shuffle(shapes)
    for h, w in shapes:
        cells = [(r, c) for r in range(h) for c in range(w)]
        for _ in range(200):
            chosen = rng.choice(len(cells), size=count, replace=False)
            mask = np.zeros((h, w), dtype=bool)
            for idx in chosen:
                r, c = cells[int(idx)]
                mask[r, c] = True
            r0, c0, r1, c1 = _bbox(mask)
            if (r0, c0, r1, c1) != (0, 0, h - 1, w - 1):
                continue
            if _row_major_positions(mask) == _col_major_positions(mask):
                continue
            patch = np.full((h, w), BG, dtype=int)
            patch[mask] = CONTROL_COLOR
            return patch
    return None


def _choose_layout_slots(n: int, d: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    gap_r = int(rng.integers(2, 5))
    gap_c = int(rng.integers(2, 5))
    r0 = int(rng.integers(6, 9))
    c0 = int(rng.integers(1, 4))
    slots = [
        (r0, c0),
        (r0, c0 + d + gap_c),
        (r0 + d + gap_r, c0),
        (r0 + d + gap_r, c0 + d + gap_c),
    ]
    patterns = {
        2: [[0, 1], [0, 2], [1, 2], [0, 3]],
        3: [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        4: [[0, 1, 2, 3]],
    }
    return [slots[i] for i in patterns[n][int(rng.integers(len(patterns[n])))]]


def _reject_order_shortcut(values: list[int]) -> bool:
    if values == sorted(values):
        return False
    if len(values) >= 3 and values == sorted(values, reverse=True):
        return False
    return True


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 - margin <= br1 and br0 - margin <= ar1 and ac0 - margin <= bc1 and bc0 - margin <= ac1


def _render_scene(
    sparse_patches: list[np.ndarray],
    control_patch: np.ndarray,
    slots: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    d = int(sparse_patches[0].shape[0])
    ctrl_h, ctrl_w = map(int, control_patch.shape)
    max_sparse_r = max(r + d for r, _ in slots)
    max_sparse_c = max(c + d for _, c in slots)

    right_margin = int(rng.integers(1, 3))
    bottom_margin = int(rng.integers(1, 3))
    ctrl_r = int(rng.integers(1, 4))
    ctrl_c_min = max_sparse_c + int(rng.integers(2, 5))
    width_min = max(ctrl_c_min + ctrl_w + right_margin, max_sparse_c + right_margin)
    height_min = max(max_sparse_r + bottom_margin, ctrl_r + ctrl_h + bottom_margin, 18)
    width = min(30, width_min + int(rng.integers(0, max(1, 30 - width_min + 1))))
    height = min(30, height_min + int(rng.integers(0, max(1, 30 - height_min + 1))))
    ctrl_c = width - ctrl_w - right_margin

    grid = np.full((height, width), BG, dtype=int)
    for patch, (r, c) in zip(sparse_patches, slots):
        paste(grid, patch, r, c)
    paste(grid, control_patch, ctrl_r, ctrl_c)
    return grid


def _validate_semantics(inp: np.ndarray) -> bool:
    try:
        bg, control, tiles, bboxes, colors = _extract_instance(inp)
    except Exception:
        return False

    if bg != BG:
        return False
    if int(np.sum(control == CONTROL_COLOR)) != len(tiles):
        return False
    if len({tuple(t.shape) for t in tiles}) != 1:
        return False

    color_ids = colors
    areas = [int(np.count_nonzero(t != BG)) for t in tiles]
    if not _reject_order_shortcut(color_ids):
        return False
    if not _reject_order_shortcut(areas):
        return False

    control_mask = control == CONTROL_COLOR
    if _row_major_positions(control_mask) == _col_major_positions(control_mask):
        return False

    out = main(inp)
    tile_h, tile_w = map(int, tiles[0].shape)
    if out.shape != (control.shape[0] * tile_h, control.shape[1] * tile_w):
        return False

    seen_tiles = []
    for r, c in _row_major_positions(control_mask):
        patch = out[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w]
        seen_tiles.append(patch)
    for a, b in zip(seen_tiles, tiles):
        if not np.array_equal(a, b):
            return False

    blank_cells = np.argwhere(~control_mask)
    for r, c in blank_cells:
        patch = out[int(r) * tile_h : (int(r) + 1) * tile_h, int(c) * tile_w : (int(c) + 1) * tile_w]
        if np.any(patch != bg):
            return False

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _boxes_overlap(bboxes[i], bboxes[j], margin=1):
                return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(2, 5))
    feasible_sizes = [s for s in (3, 4, 5) if max(4, 5) * s <= 30 and 2 * s - 1 <= 9]
    size = int(rng.choice(feasible_sizes))

    possible_areas = list(range(size + 1, size * size))
    if len(possible_areas) < n:
        return None
    area_values = sorted(rng.choice(possible_areas, size=n, replace=False).tolist())

    colors = rng.choice(np.array(SPRITE_COLORS, dtype=int), size=n, replace=False).tolist()
    masks: list[np.ndarray] = []
    for area in area_values:
        mask = _make_logical_mask(size, int(area), rng)
        if mask is None:
            return None
        masks.append(mask)

    if len({mask.tobytes() for mask in masks}) != len(masks):
        return None

    control_patch = _make_control_patch(n, rng)
    if control_patch is None:
        return None

    order = rng.permutation(n).tolist()
    ordered_areas = [area_values[i] for i in order]
    ordered_colors = [int(colors[i]) for i in order]
    if not _reject_order_shortcut(ordered_areas):
        return None
    if not _reject_order_shortcut(ordered_colors):
        return None

    sparse_patches = [_make_sparse_patch(masks[i], int(colors[i]), BG) for i in order]
    slots = _choose_layout_slots(n, int(sparse_patches[0].shape[0]), rng)
    inp = _render_scene(sparse_patches, control_patch, slots, rng)
    if not _validate_semantics(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid sparse-sprite tiling input")
