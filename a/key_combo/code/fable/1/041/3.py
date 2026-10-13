from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    h, w = map(int, patch.shape)
    dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = patch
    return dst


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean mask as coord arrays."""
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def random_free_location(
    canvas: np.ndarray,
    patch: np.ndarray,
    *,
    transparent: int,
    border: int = 1,
    padding: int = 1,
) -> Coord:
    """
    Return a random top-left location where the patch bbox fits without nearby foreground.

    For clarity we reserve the full rectangular footprint of the patch, not only its
    non-background pixels.
    """
    h, w = map(int, patch.shape)
    H, W = map(int, canvas.shape)
    bg = int(transparent)
    occ = canvas != bg
    candidates: list[Coord] = []

    for r0 in range(border, H - border - h + 1):
        for c0 in range(border, W - border - w + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(H, r0 + h + padding)
            cc1 = min(W, c0 + w + padding)
            if not occ[rr0:rr1, cc0:cc1].any():
                candidates.append((r0, c0))

    if not candidates:
        raise ValueError("no free location")
    return random.choice(candidates)


def _tile_from_component(grid: np.ndarray, bg: int, coords: np.ndarray) -> dict:
    box = bbox_coords(coords)
    patch = crop_bbox(grid, box)
    colors = Counter(map(int, patch[patch != bg]))
    assert 1 <= len(colors) <= 2, "each tile must contain one main color and at most one pointer color"

    if len(colors) == 1:
        main_color = next(iter(colors))
        next_color = None
    else:
        (main_color, main_count), (next_color, next_count) = colors.most_common(2)
        assert next_count == 1 and main_count > next_count, "dirty tile must have exactly one foreign pixel"
        next_color = int(next_color)

    repaired = patch.copy()
    repaired[(repaired != bg) & (repaired != int(main_color))] = int(main_color)
    return {
        "bbox": box,
        "patch": patch,
        "repaired": repaired,
        "main_color": int(main_color),
        "next_color": next_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the hidden color-pointer chain, repair each tile, and concatenate the clean tiles left-to-right.

    Input structure:
    - The background contains several separated tiles with identical bounding-box size and identical foreground mask.
    - Each tile has its own main color; all but one tile contain exactly one foreign-colored pixel inside the mask.
    - A foreign pixel's color is the main color of the next tile in the hidden chain, while the final tile is clean.

    Rule:
    - Treat each separated non-background component as one tile and find its main color by majority among non-background cells.
    - Read the unique foreign pixel, if present, as a pointer to the tile whose main color matches that color.
    - The chain head is the tile whose main color is never referenced; follow pointers through every tile, repair each foreign pixel back to its own main color, then concatenate the repaired tiles from left to right.

    Key insight:
    - Spatial placement is a decoy: the only ordering signal is the one-pixel color pointer hidden inside each otherwise monochrome tile.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg)
    assert len(comps) >= 3, "expected multiple scattered tiles"

    tiles = [_tile_from_component(g, bg, coords) for coords in comps]
    tile_shapes = {tile["repaired"].shape for tile in tiles}
    assert len(tile_shapes) == 1, "all tiles must share one bounding-box size"
    tile_h, tile_w = next(iter(tile_shapes))

    main_colors = [tile["main_color"] for tile in tiles]
    assert len(set(main_colors)) == len(tiles), "tile identities must be unique colors"

    referenced = {tile["next_color"] for tile in tiles if tile["next_color"] is not None}
    assert referenced <= set(main_colors), "pointers must target tile main colors"
    assert sum(tile["next_color"] is None for tile in tiles) == 1, "expected one clean chain tail"

    head_candidates = [color for color in main_colors if color not in referenced]
    assert len(head_candidates) == 1, "expected one unreferenced chain head"
    head = head_candidates[0]

    by_color = {tile["main_color"]: tile for tile in tiles}
    ordered_tiles: list[np.ndarray] = []
    seen: set[int] = set()
    cur = head

    while True:
        assert cur not in seen, "pointer chain must not cycle"
        seen.add(cur)
        tile = by_color[cur]
        ordered_tiles.append(tile["repaired"])
        if tile["next_color"] is None:
            break
        assert tile["next_color"] in by_color, "pointer target must exist"
        cur = tile["next_color"]

    assert len(ordered_tiles) == len(tiles), "pointer chain must visit every tile exactly once"

    out = np.full((tile_h, tile_w * len(ordered_tiles)), bg, dtype=int)
    for i, patch in enumerate(ordered_tiles):
        paste(out, patch, 0, i * tile_w)
    return out


PALETTE = tuple(range(10))
TILE_H_RANGE = (3, 5)
TILE_W_RANGE = (3, 5)
TILE_COUNT_RANGE = (3, 6)
MAX_ATTEMPTS = 400


def _mask_is_valid(mask: np.ndarray) -> bool:
    return (
        bool(mask[0].any())
        and bool(mask[-1].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
        and len(components_mask(mask)) == 1
    )


def _generate_mask(h: int, w: int) -> np.ndarray | None:
    total = h * w
    min_area = max(5, (total * 2 + 2) // 3)
    max_area = total - 1 if total >= 8 else total
    if min_area > max_area:
        return None

    target_area = random.randint(min_area, max_area)
    mask = np.ones((h, w), dtype=bool)
    attempts = 0
    while int(mask.sum()) > target_area and attempts < 500:
        attempts += 1
        rr, cc = random.choice(np.argwhere(mask).tolist())
        candidate = mask.copy()
        candidate[rr, cc] = False
        if _mask_is_valid(candidate):
            mask = candidate

    if int(mask.sum()) != target_area or not _mask_is_valid(mask):
        return None
    return mask


def _build_clean_tile(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _tile_chain_expected(mask: np.ndarray, chain: list[int], stain_pos: Coord, bg: int) -> tuple[np.ndarray, list[np.ndarray]]:
    clean_tiles = [_build_clean_tile(mask, color, bg) for color in chain]
    h, w = mask.shape
    out = np.full((h, w * len(chain)), bg, dtype=int)
    for i, patch in enumerate(clean_tiles):
        paste(out, patch, 0, i * w)
    dirty_tiles: list[np.ndarray] = []
    sr, sc = stain_pos
    for i, patch in enumerate(clean_tiles):
        dirty = patch.copy()
        if i + 1 < len(chain):
            dirty[sr, sc] = int(chain[i + 1])
        dirty_tiles.append(dirty)
    return out, dirty_tiles


def _chain_is_spatially_obvious(positions: list[Coord]) -> bool:
    chain_order = list(range(len(positions)))
    row_major = sorted(chain_order, key=lambda i: (positions[i][0], positions[i][1]))
    col_major = sorted(chain_order, key=lambda i: (positions[i][1], positions[i][0]))
    return row_major == chain_order or col_major == chain_order


def _gen_attempt() -> np.ndarray | None:
    tile_h = random.randint(*TILE_H_RANGE)
    tile_w = random.randint(*TILE_W_RANGE)
    tile_count = random.randint(TILE_COUNT_RANGE[0], min(TILE_COUNT_RANGE[1], 30 // tile_w))

    mask = _generate_mask(tile_h, tile_w)
    if mask is None:
        return None
    stain_pos = random.choice(np.argwhere(mask).tolist())
    stain_pos = (int(stain_pos[0]), int(stain_pos[1]))

    bg = random.choice(PALETTE)
    chain = random.sample([c for c in PALETTE if c != bg], tile_count)
    expected, dirty_tiles = _tile_chain_expected(mask, chain, stain_pos, bg)

    padded_area = tile_count * (tile_h + 2) * (tile_w + 2)
    min_side = max(tile_h + 4, tile_w + 4, int(np.ceil(np.sqrt(padded_area))))
    if min_side > 30:
        return None

    H = random.randint(min_side, 30)
    W = random.randint(min_side, 30)
    grid = np.full((H, W), bg, dtype=int)

    placement_order = list(range(tile_count))
    random.shuffle(placement_order)
    positions: list[Coord | None] = [None] * tile_count
    for idx in placement_order:
        try:
            r0, c0 = random_free_location(grid, dirty_tiles[idx], transparent=bg, border=1, padding=1)
        except ValueError:
            return None
        paste(grid, dirty_tiles[idx], r0, c0)
        positions[idx] = (r0, c0)

    top_lefts = [pos for pos in positions if pos is not None]
    assert len(top_lefts) == tile_count
    if _chain_is_spatially_obvious(top_lefts):
        return None

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid color-pointer-chain input")
