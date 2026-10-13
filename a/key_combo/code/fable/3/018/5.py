from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]  # (r0, c0, r1, c1), inclusive


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    monochromatic: bool,
    colors: list[int] | None = None,
) -> list[dict]:
    if monochromatic:
        color_list = colors
        if color_list is None:
            color_list = [int(v) for v in np.unique(grid) if int(v) != int(bg)]
        comps: list[dict] = []
        for color in color_list:
            for coords in _components_from_mask(grid == int(color)):
                comps.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": _bbox_from_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return comps

    comps = []
    for coords in _components_from_mask(grid != int(bg)):
        vals = {int(grid[r, c]) for r, c in coords}
        comps.append(
            {
                "color": None if len(vals) != 1 else next(iter(vals)),
                "coords": coords,
                "bbox": _bbox_from_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _bbox_shape(bbox: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _touches_corner(bbox: BBox, shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _is_solid_rect_of_fg(grid: np.ndarray, bbox: BBox, bg: int) -> bool:
    patch = _crop_bbox(grid, bbox)
    return bool(np.all(patch != int(bg)))


def _detect_mosaic_bbox(grid: np.ndarray, bg: int) -> BBox:
    candidates: list[BBox] = []
    for comp in _find_components(grid, bg=bg, monochromatic=False):
        bbox = comp["bbox"]
        if not _touches_corner(bbox, grid.shape):
            continue
        if not _is_solid_rect_of_fg(grid, bbox, bg):
            continue
        patch = _crop_bbox(grid, bbox)
        if len(np.unique(patch)) < 2:
            continue
        candidates.append(bbox)
    assert len(candidates) == 1, "expected exactly one corner-anchored mosaic block"
    return candidates[0]


def _build_skeleton(mosaic: np.ndarray, bg: int) -> np.ndarray:
    comps = _find_components(mosaic, bg=bg, monochromatic=True)
    assert comps, "mosaic must contain rectangles"

    row_starts = sorted({int(comp["bbox"][0]) for comp in comps})
    col_starts = sorted({int(comp["bbox"][1]) for comp in comps})
    skeleton = np.full((len(row_starts), len(col_starts)), bg, dtype=int)

    for comp in comps:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        patch = mosaic[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(patch == color)), "each mosaic region must be a solid rectangle"

    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            skeleton[ri, ci] = int(mosaic[r, c])
    return skeleton


def _extract_sprite_masks(grid: np.ndarray, bg: int, mosaic_bbox: BBox, colors: list[int]) -> dict[int, np.ndarray]:
    sprite_masks: dict[int, np.ndarray] = {}
    comps = _find_components(grid, bg=bg, monochromatic=True, colors=colors)
    for comp in comps:
        color = int(comp["color"])
        bbox = comp["bbox"]
        if _boxes_overlap(bbox, mosaic_bbox):
            continue
        if _bbox_shape(bbox) != (3, 3):
            continue
        patch = _crop_bbox(grid, bbox)
        if not bool(np.all(np.isin(patch, [bg, color]))):
            continue
        mask = patch == color
        assert bool(mask.any()), "legend sprite cannot be empty"
        assert color not in sprite_masks, "each color must have exactly one legend sprite"
        sprite_masks[color] = mask

    assert set(sprite_masks) == set(map(int, colors)), "legend must cover every mosaic color"
    return sprite_masks


def _render_output(skeleton: np.ndarray, sprite_masks: dict[int, np.ndarray], bg: int) -> np.ndarray:
    out = np.full((skeleton.shape[0] * 3, skeleton.shape[1] * 3), bg, dtype=int)
    for r in range(skeleton.shape[0]):
        for c in range(skeleton.shape[1]):
            color = int(skeleton[r, c])
            patch = np.full((3, 3), bg, dtype=int)
            patch[sprite_masks[color]] = color
            rr = 3 * r
            cc = 3 * c
            out[rr : rr + 3, cc : cc + 3] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the unequal rectangular mosaic to its partition skeleton, then re-expand each skeleton cell with its color's 3x3 legend sprite.

    Input structure:
    - One corner contains a solid rectangular mosaic made of gapless monochrome rectangles with varying sizes.
    - The rest of the grid is background plus one 3x3-bbox monochrome legend sprite for each color used in the mosaic.
    - The legend sprites are separate from the mosaic and reuse the same colors as the mosaic cells.

    Rule:
    - Detect the corner-anchored mosaic block and treat each monochrome rectangle inside it as one partition cell.
    - Collect the unique upper-left row and column positions of those rectangles; sampling the mosaic at those coordinates yields the compact color skeleton.
    - Replace every skeleton color cell by its color's 3x3 legend sprite, painting the sprite cells in that color and leaving the other cells as background.

    Key insight:
    - The mosaic must first be normalized by partition coordinates rather than resized proportionally, and only then expanded a second time using the separate color-indexed legend sprites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    mosaic_bbox = _detect_mosaic_bbox(g, bg)
    mosaic = _crop_bbox(g, mosaic_bbox)
    skeleton = _build_skeleton(mosaic, bg)
    colors = sorted(int(v) for v in np.unique(skeleton) if int(v) != int(bg))
    sprites = _extract_sprite_masks(g, bg, mosaic_bbox, colors)
    return _render_output(skeleton, sprites, bg)


_SPRITE_SHAPES = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
)


def _sample_lengths(n_parts: int) -> list[int]:
    for _ in range(200):
        lengths = [random.randint(1, 5) for _ in range(n_parts)]
        if sum(lengths) <= 18:
            return lengths
    raise RuntimeError("failed to sample partition lengths")


def _sample_color_grid(rows: int, cols: int, palette: list[int]) -> np.ndarray:
    n_cells = rows * cols
    k = len(palette)
    max_per_color = (n_cells + 1) // 2

    for _ in range(300):
        counts = {color: 1 for color in palette}
        remaining = n_cells - k
        while remaining:
            color = random.choice(palette)
            if counts[color] < max_per_color:
                counts[color] += 1
                remaining -= 1

        grid = np.full((rows, cols), -1, dtype=int)

        def backtrack(idx: int) -> bool:
            if idx == n_cells:
                return True
            r = idx // cols
            c = idx % cols

            used_up = int(grid[r - 1, c]) if r > 0 else None
            used_left = int(grid[r, c - 1]) if c > 0 else None
            candidates = [
                color
                for color in palette
                if counts[color] > 0 and color != used_up and color != used_left
            ]
            random.shuffle(candidates)
            candidates.sort(key=lambda color: counts[color], reverse=True)

            cells_left = n_cells - idx - 1
            for color in candidates:
                counts[color] -= 1
                needed = sum(1 for value in counts.values() if value > 0)
                if needed <= cells_left:
                    grid[r, c] = color
                    if backtrack(idx + 1):
                        return True
                grid[r, c] = -1
                counts[color] += 1
            return False

        if backtrack(0):
            return grid

    raise RuntimeError("failed to assign mosaic colors")


def _render_mosaic(color_grid: np.ndarray, row_heights: list[int], col_widths: list[int], bg: int) -> np.ndarray:
    mosaic = np.full((sum(row_heights), sum(col_widths)), bg, dtype=int)
    row_edges = np.cumsum([0] + row_heights)
    col_edges = np.cumsum([0] + col_widths)
    for r in range(color_grid.shape[0]):
        for c in range(color_grid.shape[1]):
            mosaic[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]] = int(color_grid[r, c])
    return mosaic


def _corner_origin(corner: str, grid_shape: tuple[int, int], patch_shape: tuple[int, int]) -> Coord:
    h, w = grid_shape
    ph, pw = patch_shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - pw
    if corner == "bl":
        return h - ph, 0
    return h - ph, w - pw


def _dilate_box(mask: np.ndarray, bbox: BBox, pad: int) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r1 + pad + 1)
    cc1 = min(w, c1 + pad + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _place_legends(
    grid: np.ndarray,
    mosaic_bbox: BBox,
    legend_items: list[tuple[int, np.ndarray]],
    bg: int,
) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    _dilate_box(blocked, mosaic_bbox, 1)

    items = legend_items[:]
    random.shuffle(items)
    for color, mask in items:
        candidates: list[Coord] = []
        for r in range(1, h - 3):
            for c in range(1, w - 3):
                if blocked[r - 1 : r + 4, c - 1 : c + 4].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        patch = np.full((3, 3), bg, dtype=int)
        patch[mask] = int(color)
        grid[r0 : r0 + 3, c0 : c0 + 3] = patch
        _dilate_box(blocked, (r0, c0, r0 + 2, c0 + 2), 1)
    return True


def _spec_output(color_grid: np.ndarray, sprite_masks: dict[int, np.ndarray], bg: int) -> np.ndarray:
    return _render_output(color_grid, sprite_masks, bg)


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    color_grid: np.ndarray,
    row_heights: list[int],
    col_widths: list[int],
    sprite_masks: dict[int, np.ndarray],
) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(set(row_heights)) == 1 and len(set(col_widths)) == 1:
        return False
    if any(color_grid[r, c] == color_grid[r + 1, c] for r in range(color_grid.shape[0] - 1) for c in range(color_grid.shape[1])):
        return False
    if any(color_grid[r, c] == color_grid[r, c + 1] for r in range(color_grid.shape[0]) for c in range(color_grid.shape[1] - 1)):
        return False
    if len({mask.tobytes() for mask in sprite_masks.values()}) != len(sprite_masks):
        return False

    bg = _get_bg(grid)
    try:
        mosaic_bbox = _detect_mosaic_bbox(grid, bg)
        mosaic = _crop_bbox(grid, mosaic_bbox)
        parsed_skeleton = _build_skeleton(mosaic, bg)
        parsed_sprites = _extract_sprite_masks(
            grid,
            bg,
            mosaic_bbox,
            sorted(int(v) for v in np.unique(parsed_skeleton) if int(v) != int(bg)),
        )
        solved = main(grid)
    except Exception:
        return False

    if not np.array_equal(parsed_skeleton, color_grid):
        return False
    if set(parsed_sprites) != set(sprite_masks):
        return False
    if any(not np.array_equal(parsed_sprites[color], sprite_masks[color]) for color in sprite_masks):
        return False
    if not np.array_equal(solved, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(2, 5)
    cols = random.randint(2, 5)
    row_heights = _sample_lengths(rows)
    col_widths = _sample_lengths(cols)
    if len(set(row_heights)) == 1 and len(set(col_widths)) == 1:
        return None

    n_cells = rows * cols
    n_colors = random.randint(2, min(5, n_cells, len(_SPRITE_SHAPES)))
    all_colors = list(range(10))
    palette = random.sample(all_colors, n_colors + 1)
    bg = palette[0]
    colors = palette[1:]

    color_grid = _sample_color_grid(rows, cols, colors)
    mosaic = _render_mosaic(color_grid, row_heights, col_widths, bg)
    sprite_shape_choices = random.sample(_SPRITE_SHAPES, len(colors))
    sprite_masks = {color: mask.copy() for color, mask in zip(colors, sprite_shape_choices)}

    mosaic_h, mosaic_w = mosaic.shape
    out_h = rows * 3
    out_w = cols * 3
    if out_h > 30 or out_w > 30:
        return None

    max_extra_h = 30 - mosaic_h
    max_extra_w = 30 - mosaic_w
    if max_extra_h < 5 or max_extra_w < 5:
        return None
    grid_h = mosaic_h + random.randint(5, min(12, max_extra_h))
    grid_w = mosaic_w + random.randint(5, min(12, max_extra_w))

    corner = random.choice(("tl", "tr", "bl", "br"))
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    mr0, mc0 = _corner_origin(corner, grid.shape, mosaic.shape)
    grid[mr0 : mr0 + mosaic_h, mc0 : mc0 + mosaic_w] = mosaic
    mosaic_bbox = (mr0, mc0, mr0 + mosaic_h - 1, mc0 + mosaic_w - 1)

    legend_items = [(color, sprite_masks[color]) for color in colors]
    if not _place_legends(grid, mosaic_bbox, legend_items, bg):
        return None

    expected = _spec_output(color_grid, sprite_masks, bg)
    if not _validate_generated(grid, expected, color_grid, row_heights, col_widths, sprite_masks):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
