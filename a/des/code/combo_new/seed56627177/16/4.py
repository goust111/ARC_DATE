from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
BORDER_COLORS = (1, 2, 3, 6, 7, 8, 9)

MAX_GRID_SIZE = 30
GENERATION_ATTEMPTS = 500
SHAPE_ATTEMPTS = 120
REGION_PADDING = 2
SPRITE_PADDING = 1


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    norm = _normalize_mask(mask)
    return (int(norm.shape[0]), int(norm.shape[1])), norm.tobytes()


def _rotation_keys(mask: np.ndarray) -> list[tuple[tuple[int, int], bytes]]:
    cur = _normalize_mask(mask)
    keys: list[tuple[tuple[int, int], bytes]] = []
    seen = set()
    for _ in range(4):
        key = _shape_key(cur)
        if key not in seen:
            seen.add(key)
            keys.append(key)
        cur = np.rot90(cur)
    return keys


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_rotation_keys(mask), key=lambda item: (item[0][0], item[0][1], item[1]))


def _extract_regions_and_sprites(
    grid: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    regions: list[dict[str, object]] = []
    sprites: list[dict[str, object]] = []

    for comp in _components(grid != BG, connectivity=4):
        colors = {int(grid[r, c]) for r, c in comp}
        if colors == {YELLOW}:
            sprites.append({"coords": comp, "mask": _mask_from_coords(comp)})
            continue

        if GREY not in colors or YELLOW in colors:
            continue

        border_colors = colors - {GREY}
        if len(border_colors) != 1:
            continue

        border_color = int(next(iter(border_colors)))
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        hole_candidates = []
        for hole in _components(sub == BG, connectivity=4):
            if any(
                rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1
                for rr, cc in hole
            ):
                continue
            hole_abs = [(r0 + rr, c0 + cc) for rr, cc in hole]
            hole_candidates.append((hole_abs, _mask_from_coords(hole)))

        if len(hole_candidates) != 1:
            continue

        hole_abs, hole_mask = hole_candidates[0]
        regions.append(
            {
                "border_color": border_color,
                "hole_abs": hole_abs,
                "hole_mask": hole_mask,
            }
        )

    return regions, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each framed hole with its shape-matched outside yellow sprite, recolored to the frame border.

    Input structure:
    - The background is black.
    - Several large closed regions appear as one-pixel-thick colored borders around grey-filled interiors.
    - Each grey interior contains exactly one black hole shape.
    - Outside those regions, the background contains several scattered yellow sprite components.

    Rule:
    - Extract every hole shape from the framed regions and every outside yellow sprite shape.
    - Match sprites to holes by exact shape up to rotation only; flips are not allowed.
    - Clear the original yellow sprites from their scattered locations.
    - Fill each hole in place using the matching sprite's occupied cells, recolored to that region's border color.

    Key insight:
    - The yellow sprites are movable shape tokens: their geometry chooses which hole they fill, while the destination frame color determines the final recoloring.
    """
    g = np.asarray(grid, dtype=int)
    regions, sprites = _extract_regions_and_sprites(g)
    assert regions and sprites

    hole_counts = Counter(_canonical_rotation_key(region["hole_mask"]) for region in regions)
    sprite_counts = Counter(_canonical_rotation_key(sprite["mask"]) for sprite in sprites)
    assert hole_counts == sprite_counts

    out = g.copy()
    out[out == YELLOW] = BG
    for region in regions:
        color = int(region["border_color"])
        for r, c in region["hole_abs"]:
            out[r, c] = color
    return out


def _random_shape() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        target = random.randint(3, 7)
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}

        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.add(nxt)

        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0 = min(rs)
        c0 = min(cs)
        norm = {(r - r0, c - c0) for r, c in cells}
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if min(h, w) < 2 or max(h, w) > 4:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        if len(_rotation_keys(mask)) != 4:
            continue
        return mask
    return None


def _sample_distinct_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen = set()
    for _ in range(SHAPE_ATTEMPTS * 4):
        shape = _random_shape()
        if shape is None:
            continue
        key = _canonical_rotation_key(shape)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(shape)
        if len(shapes) == count:
            return shapes
    return None


def _make_region_patch(hole_mask: np.ndarray, border_color: int) -> np.ndarray:
    hole_h, hole_w = hole_mask.shape
    interior_h = hole_h + random.randint(2, 5)
    interior_w = hole_w + random.randint(2, 5)
    hole_r = random.randint(1, interior_h - hole_h - 1)
    hole_c = random.randint(1, interior_w - hole_w - 1)

    patch = np.full((interior_h + 2, interior_w + 2), GREY, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    window = patch[1 + hole_r : 1 + hole_r + hole_h, 1 + hole_c : 1 + hole_c + hole_w]
    window[hole_mask] = BG
    return patch


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, padding: int) -> bool:
    h_grid, w_grid = occupied.shape
    if top < 0 or left < 0 or top + h > h_grid or left + w > w_grid:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h_grid, top + h + padding)
    c1 = min(w_grid, left + w + padding)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    padding: int,
) -> tuple[int, int] | None:
    h_grid, w_grid = grid.shape
    h, w = patch.shape
    candidates = [
        (r, c)
        for r in range(h_grid - h + 1)
        for c in range(w_grid - w + 1)
        if _can_place(occupied, r, c, h, w, padding)
    ]
    if not candidates:
        return None

    top, left = random.choice(candidates)
    view = grid[top : top + h, left : left + w]
    mask = patch != BG
    view[mask] = patch[mask]
    occupied[top : top + h, left : left + w] = True
    return int(top), int(left)


def _build_case() -> np.ndarray | None:
    h = random.randint(24, MAX_GRID_SIZE)
    w = random.randint(24, MAX_GRID_SIZE)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    region_count = random.choice([2, 3])
    shapes = _sample_distinct_shapes(region_count)
    if shapes is None:
        return None
    border_colors = random.sample(list(BORDER_COLORS), region_count)

    regions = []
    sprites = []
    for shape, border_color in zip(shapes, border_colors, strict=False):
        hole_rot = random.randrange(4)
        sprite_rot = random.randrange(4)
        hole_mask = np.rot90(shape, hole_rot)
        sprite_mask = np.rot90(shape, sprite_rot)
        region_patch = _make_region_patch(hole_mask, int(border_color))
        sprite_patch = np.where(sprite_mask, YELLOW, BG).astype(int)
        regions.append(region_patch)
        sprites.append(sprite_patch)

    regions.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)
    for patch in regions:
        if _place_patch(grid, occupied, patch, REGION_PADDING) is None:
            return None

    random.shuffle(sprites)
    for patch in sprites:
        if _place_patch(grid, occupied, patch, SPRITE_PADDING) is None:
            return None

    regions_found, sprites_found = _extract_regions_and_sprites(grid)
    if len(regions_found) != region_count or len(sprites_found) != region_count:
        return None

    if Counter(_canonical_rotation_key(region["hole_mask"]) for region in regions_found) != Counter(
        _canonical_rotation_key(sprite["mask"]) for sprite in sprites_found
    ):
        return None

    if np.count_nonzero(grid == BG) <= grid.size // 3:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.any(out == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
