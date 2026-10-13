from __future__ import annotations

from collections import deque
import math

import numpy as np


BLACK = 0
GREEN = 3
TOP_TEMPLATE_HEIGHT = 2
TEMPLATE_GAP = 2
REGION_TOP_MIN = 4
GRID_ROW_RANGE = (18, 25)
GRID_COL_RANGE = (18, 25)
MAX_ATTEMPTS = 400
COLOR_POOL = [1, 2, 4, 5, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _extract_templates(top: np.ndarray) -> dict[int, np.ndarray]:
    records = []
    for color in sorted(int(v) for v in np.unique(top)):
        if color in (BLACK, GREEN):
            continue
        coords = np.argwhere(top == color)
        assert coords.size > 0, "template color must appear in the top band"
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = top[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch == color
        records.append((c0, color, mask))
    records.sort()
    assert len(records) == 3, "expected exactly three template colors"
    return {color: mask.copy() for _c0, color, mask in records}


def _find_enclosed_regions(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    green_rows = np.argwhere(grid == GREEN)[:, 0]
    assert green_rows.size > 0, "expected green boundaries"
    region_row0 = int(green_rows.min())
    scene = grid[region_row0:]
    open_mask = scene != GREEN
    regions = [coords for coords in _components(open_mask) if not _touches_border(coords, scene.shape)]
    assert len(regions) == 3, "expected exactly three enclosed regions"
    return region_row0, regions


def _seed_color(scene: np.ndarray, coords: np.ndarray) -> int:
    colors = {
        int(scene[int(r), int(c)])
        for r, c in coords
        if int(scene[int(r), int(c)]) != BLACK
    }
    assert len(colors) == 1, "each region must contain exactly one seed color"
    return next(iter(colors))


def _coords_to_local_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return (r0, c0, r1, c1), mask


def _tile_boolean(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mh, mw = mask.shape
    reps = (math.ceil(h / mh), math.ceil(w / mw))
    return np.tile(mask, reps)[:h, :w]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each irregular enclosed region with the template selected by its seed color, tiled then clipped to the region.

    Input structure:
    - The top band shows three small binary templates, each using one unique color plus black.
    - Lower in the grid, green one-cell-wide closed boundaries surround three black interior regions of irregular shape.
    - Each enclosed region contains exactly one seed pixel whose color matches one of the top templates.

    Rule:
    - Identify the three top templates by color and read the seed color inside each enclosed region.
    - For each region, choose the template with the same color, tile that binary template across the region's minimal bounding box, and use the seed color as the tile color.
    - Keep only the tiled pixels that lie inside the enclosed region, overwrite the seed, and preserve the top templates plus all green boundaries.

    Key insight:
    - The seed color is only a selector: it chooses which upper template to tile, while the green boundary determines the irregular clipping mask applied to that tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    region_row0, regions = _find_enclosed_regions(g)
    templates = _extract_templates(g[:region_row0])
    scene = g[region_row0:]
    out = g.copy()

    for coords in regions:
        color = _seed_color(scene, coords)
        assert color in templates, "seed color must match a top template"
        template = templates[color]
        (r0, c0, r1, c1), region_mask = _coords_to_local_mask(coords)
        tiled = _tile_boolean(template, region_mask.shape)
        patch = np.zeros(region_mask.shape, dtype=int)
        patch[tiled] = color
        out_slice = out[region_row0 + r0 : region_row0 + r1 + 1, c0 : c1 + 1]
        out_slice[region_mask] = patch[region_mask]

    return out


def _template_pool(width: int) -> list[np.ndarray]:
    pool = []
    n = TOP_TEMPLATE_HEIGHT * width
    for bits in range(1 << n):
        mask = np.array([(bits >> k) & 1 for k in range(n)], dtype=bool).reshape(TOP_TEMPLATE_HEIGHT, width)
        count = int(mask.sum())
        if count < 2 or count >= n:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        pool.append(mask)
    return pool


_TEMPLATE_POOLS = {2: _template_pool(2), 3: _template_pool(3)}


def _bbox_choices(template_shape: tuple[int, int]) -> list[tuple[int, int]]:
    _h, w = template_shape
    if w == 2:
        return [(4, 4), (4, 6), (6, 4)]
    return [(4, 3), (6, 3), (4, 6)]


def _apply_notch(mask: np.ndarray, side: str, start: int, length: int, depth: int) -> None:
    h, w = mask.shape
    if side == "top":
        mask[:depth, start : start + length] = False
    elif side == "bottom":
        mask[h - depth :, start : start + length] = False
    elif side == "left":
        mask[start : start + length, :depth] = False
    else:
        mask[start : start + length, w - depth :] = False


def _valid_region_mask(mask: np.ndarray) -> bool:
    if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return False
    if np.all(mask):
        return False
    area = int(mask.sum())
    if not (8 <= area <= 25):
        return False
    comps = _components(mask)
    return len(comps) == 1


def _random_region_mask(rng: np.random.Generator, shape: tuple[int, int], template: np.ndarray) -> np.ndarray | None:
    h, w = shape
    for _ in range(120):
        mask = np.ones((h, w), dtype=bool)
        sides = ["top", "bottom", "left", "right"]
        notch_count = 1 if rng.random() < 0.45 else 2
        rng.shuffle(sides)
        for side in sides[:notch_count]:
            if side in ("top", "bottom"):
                if w < 3:
                    continue
                start = int(rng.integers(1, w - 1))
                max_len = w - start - 1
                if max_len <= 0:
                    continue
                length = int(rng.integers(1, max_len + 1))
                max_depth = max(1, min(h - 1, h // 2))
            else:
                if h < 3:
                    continue
                start = int(rng.integers(1, h - 1))
                max_len = h - start - 1
                if max_len <= 0:
                    continue
                length = int(rng.integers(1, max_len + 1))
                max_depth = max(1, min(w - 1, w // 2))
            depth = int(rng.integers(1, max_depth + 1))
            _apply_notch(mask, side, start, length, depth)

        if not _valid_region_mask(mask):
            continue

        tiled = _tile_boolean(template, mask.shape)
        if not np.any(tiled & mask):
            continue
        if not np.any(mask & ~tiled):
            continue
        if not np.any(tiled & ~mask):
            continue
        return mask
    return None


def _boundary_patch(region_mask: np.ndarray, seed_color: int, seed_pos: tuple[int, int]) -> np.ndarray:
    h, w = region_mask.shape
    patch = np.zeros((h + 2, w + 2), dtype=int)
    interior = np.zeros_like(patch, dtype=bool)
    interior[1 : h + 1, 1 : w + 1] = region_mask

    boundary = np.zeros_like(interior, dtype=bool)
    for r, c in np.argwhere(interior):
        for nr, nc in _neighbors4(int(r), int(c)):
            if not interior[nr, nc]:
                boundary[nr, nc] = True
    patch[boundary] = GREEN
    sr, sc = seed_pos
    patch[1 + sr, 1 + sc] = seed_color
    return patch


def _rectangles_separated(
    placed: list[tuple[int, int, int, int]], rect: tuple[int, int, int, int], pad: int = 1
) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + pad < pr0 or pr1 + pad < r0 or c1 + pad < pc0 or pc1 + pad < c0):
            return False
    return True


def _place_patches(
    rng: np.random.Generator, grid_shape: tuple[int, int], patches: list[np.ndarray]
) -> list[tuple[int, int]] | None:
    gh, gw = grid_shape
    order = list(range(len(patches)))
    rng.shuffle(order)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    placed_rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if ph >= gh - REGION_TOP_MIN or pw >= gw:
            return None
        candidates = []
        for r0 in range(REGION_TOP_MIN, gh - ph + 1):
            for c0 in range(0, gw - pw + 1):
                rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if _rectangles_separated(placed_rects, rect, pad=1):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        choice = candidates[int(rng.integers(0, len(candidates)))]
        placements[idx] = choice
        r0, c0 = choice
        placed_rects.append((r0, c0, r0 + ph - 1, c0 + pw - 1))

    return [pos for pos in placements if pos is not None]


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        region_row0, regions = _find_enclosed_regions(grid)
        templates = _extract_templates(grid[:region_row0])
        if len(templates) != 3 or len(set(templates)) != 3:
            return False

        scene = grid[region_row0:]
        seed_colors = []
        areas = []
        for coords in regions:
            color = _seed_color(scene, coords)
            if color not in templates:
                return False
            seed_colors.append(color)
            (_r0, _c0, _r1, _c1), region_mask = _coords_to_local_mask(coords)
            template = templates[color]
            th, tw = template.shape
            h, w = region_mask.shape
            if h % th != 0 or w % tw != 0:
                return False
            if np.all(region_mask):
                return False
            tiled = _tile_boolean(template, region_mask.shape)
            if not np.any(tiled & region_mask):
                return False
            if not np.any(region_mask & ~tiled):
                return False
            if not np.any(tiled & ~region_mask):
                return False
            areas.append(int(region_mask.sum()))

        if sorted(seed_colors) != sorted(templates):
            return False
        if len(set(seed_colors)) != 3 or len(set(areas)) != 3:
            return False
        return True
    except Exception:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(*GRID_ROW_RANGE))
    grid_w = int(rng.integers(*GRID_COL_RANGE))
    grid = np.zeros((grid_h, grid_w), dtype=int)

    widths = [2 if rng.random() < 0.5 else 3 for _ in range(3)]
    masks = []
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    for width in widths:
        pool = _TEMPLATE_POOLS[width]
        rng.shuffle(pool)
        chosen = None
        for mask in pool:
            key = (mask.shape, mask.tobytes())
            if key not in used_shapes:
                used_shapes.add(key)
                chosen = mask.copy()
                break
        if chosen is None:
            return None
        masks.append(chosen)

    colors = rng.choice(np.array(COLOR_POOL, dtype=int), size=3, replace=False).tolist()
    template_records = list(zip(colors, masks))
    rng.shuffle(template_records)

    total_template_w = sum(mask.shape[1] for _color, mask in template_records) + TEMPLATE_GAP * 2
    if total_template_w > grid_w:
        return None
    left = int(rng.integers(0, grid_w - total_template_w + 1))
    cursor = left
    for color, mask in template_records:
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = color
        grid[: mask.shape[0], cursor : cursor + mask.shape[1]] = patch
        cursor += mask.shape[1] + TEMPLATE_GAP

    patches = []
    region_areas = []
    for color, template in template_records:
        bbox_shape = _bbox_choices(template.shape)[int(rng.integers(0, 3))]
        region_mask = _random_region_mask(rng, bbox_shape, template)
        if region_mask is None:
            return None
        seed_cells = np.argwhere(region_mask)
        seed_index = int(rng.integers(0, len(seed_cells)))
        seed_pos = tuple(int(v) for v in seed_cells[seed_index])
        patches.append(_boundary_patch(region_mask, color, seed_pos))
        region_areas.append(int(region_mask.sum()))

    if len(set(region_areas)) != 3:
        return None

    placements = _place_patches(rng, grid.shape, patches)
    if placements is None:
        return None

    for patch, (r0, c0) in zip(patches, placements):
        ph, pw = patch.shape
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any((sub != BLACK) & (patch != BLACK)):
            return None
        sub[patch != BLACK] = patch[patch != BLACK]

    if not _validate_instance(grid):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
