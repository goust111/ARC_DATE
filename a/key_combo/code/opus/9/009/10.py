from __future__ import annotations

import math
import random

import numpy as np


BACKGROUND = 0
DIVIDER_COLOR = 5
PALETTE = tuple(color for color in range(1, 10) if color != DIVIDER_COLOR)

GRID_SIDES = (13, 15, 17)
OUTPUT_SIDES = (4, 6)
GENERATION_RETRIES = 400

_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _quadrants(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    row_hits = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER_COLOR)]
    col_hits = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER_COLOR)]
    assert len(row_hits) == 1 and len(col_hits) == 1, "expected one full divider row and one full divider col"
    split_r = int(row_hits[0])
    split_c = int(col_hits[0])
    quadrants = [
        grid[:split_r, :split_c],
        grid[:split_r, split_c + 1 :],
        grid[split_r + 1 :, :split_c],
        grid[split_r + 1 :, split_c + 1 :],
    ]
    sizes = {(q.shape[0], q.shape[1]) for q in quadrants}
    assert len(sizes) == 1, "quadrants must be equal-sized once the divider row/col are removed"
    return quadrants, split_r, split_c


def _quadrant_foreground_color(quadrant: np.ndarray) -> int:
    colors = sorted(set(map(int, quadrant.ravel())) - {BACKGROUND, DIVIDER_COLOR})
    assert len(colors) == 1, "each quadrant must contain exactly one foreground color"
    return int(colors[0])


def _tile_to_size(tile: np.ndarray, side: int) -> np.ndarray:
    th, tw = tile.shape
    out = np.zeros((side, side), dtype=int)
    for r in range(side):
        for c in range(side):
            out[r, c] = int(tile[r % th, c % tw])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the uniquely colored quadrant sprite as a tile and repeat it to the size encoded by the other three quadrants.

    Input structure:
    - The grid is an odd-sized square with black background and one full grey row and one full grey column crossing at the center.
    - Those dividers split the grid into four equal quadrants.
    - Three quadrants contain sparse pixels of one shared foreground color, while the fourth quadrant contains a small 4-connected shape in a different color.

    Rule:
    - Split the grid at the grey divider row and column.
    - Identify the unique outlier-color quadrant, crop that quadrant to the bounding box of its foreground shape, and keep both the outlier color and any black cells inside that box as the template tile.
    - Count the shared-color pixels across the other three quadrants; that total must be a perfect square N = s^2.
    - Produce an s-by-s output by tiling the cropped outlier template from the top-left, wrapping both rows and columns cyclically.

    Key insight:
    - The outlier quadrant says what to tile, but only the three matching-color quadrants say how large the tiled result must be.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    quadrants, _split_r, _split_c = _quadrants(g)

    colors = [_quadrant_foreground_color(q) for q in quadrants]
    outlier_indices = [i for i, color in enumerate(colors) if colors.count(color) == 1]
    assert len(outlier_indices) == 1, "expected exactly one outlier-colored quadrant"
    outlier_index = int(outlier_indices[0])

    common_indices = [i for i in range(4) if i != outlier_index]
    common_colors = {colors[i] for i in common_indices}
    assert len(common_colors) == 1, "the other three quadrants must share one color"
    common_color = next(iter(common_colors))

    template_quadrant = quadrants[outlier_index]
    template_bbox = _bbox(template_quadrant == colors[outlier_index])
    assert template_bbox is not None, "outlier quadrant must contain foreground"
    template = _crop_bbox(template_quadrant, template_bbox)

    total_common = sum(int(np.count_nonzero(quadrants[i] == common_color)) for i in common_indices)
    side = int(math.isqrt(total_common))
    assert side * side == total_common and side > 0, "shared-color count must be a positive perfect square"
    assert side % template.shape[0] == 0 and side % template.shape[1] == 0, "template bbox must tile the output exactly"

    return _tile_to_size(template, side)


def _shape_masks_for_side(side: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    dims: list[tuple[int, int]] = []
    for h in range(2, min(side, 4) + 1):
        if side % h != 0:
            continue
        for w in range(2, min(side, 4) + 1):
            if side % w != 0:
                continue
            if h == side and w == side:
                continue
            dims.append((h, w))

    for h, w in dims:
        area = h * w
        for bits in range(1 << area):
            pop = bits.bit_count()
            if not (3 <= pop <= 5):
                continue
            mask = np.zeros((h, w), dtype=bool)
            for k in range(area):
                if bits & (1 << k):
                    mask[k // w, k % w] = True
            if not mask[0, :].any() or not mask[-1, :].any():
                continue
            if not mask[:, 0].any() or not mask[:, -1].any():
                continue
            if np.all(mask):
                continue
            if len(_components4(mask)) != 1:
                continue
            masks.append(mask)

    assert masks, f"expected at least one legal template mask for side {side}"
    return masks


def _sample_common_counts(side: int, quad_area: int) -> tuple[int, int, int]:
    target = side * side
    candidates: list[tuple[int, int, int]] = []
    for a in range(2, min(14, quad_area) + 1):
        for b in range(2, min(14, quad_area) + 1):
            c = target - a - b
            if not (2 <= c <= min(14, quad_area)):
                continue
            counts = (a, b, c)
            if len(set(counts)) < 2:
                continue
            candidates.append(counts)
    assert candidates, "expected at least one valid count split"
    return random.choice(candidates)


def _scatter_points(h: int, w: int, count: int, color: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(all_cells)
    for r, c in all_cells[:count]:
        grid[r, c] = int(color)
    return grid


def _place_template(quadrant: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray:
    h, w = quadrant.shape
    th, tw = mask.shape
    max_r = h - th
    max_c = w - tw
    assert max_r >= 0 and max_c >= 0

    anchors = [
        (r, c)
        for r in range(max_r + 1)
        for c in range(max_c + 1)
        if r > 0 or c > 0 or r + th < h or c + tw < w
    ]
    random.shuffle(anchors)
    for r0, c0 in anchors:
        placed = np.zeros_like(quadrant)
        placed[r0 : r0 + th, c0 : c0 + tw][mask] = int(color)
        local_bbox = _bbox(placed == color)
        assert local_bbox is not None
        cropped = _crop_bbox(placed, local_bbox)
        if cropped.shape != mask.shape:
            continue
        if np.count_nonzero(cropped == BACKGROUND) == 0:
            continue
        if len(_components4(cropped == color)) != 1:
            continue
        return placed
    raise AssertionError("failed to place template with visible padding")


def _build_case() -> np.ndarray:
    side = random.choice(GRID_SIDES)
    quad = side // 2
    out_side = random.choice(OUTPUT_SIDES)
    template_mask = random.choice(_shape_masks_for_side(out_side))
    assert out_side % template_mask.shape[0] == 0 and out_side % template_mask.shape[1] == 0
    assert template_mask.shape[0] <= quad and template_mask.shape[1] <= quad

    counts = list(_sample_common_counts(out_side, quad * quad))
    random.shuffle(counts)
    common_color, outlier_color = random.sample(PALETTE, 2)

    grid = np.zeros((side, side), dtype=int)
    split = quad
    grid[split, :] = DIVIDER_COLOR
    grid[:, split] = DIVIDER_COLOR

    quadrant_slices = [
        (slice(0, split), slice(0, split)),
        (slice(0, split), slice(split + 1, side)),
        (slice(split + 1, side), slice(0, split)),
        (slice(split + 1, side), slice(split + 1, side)),
    ]
    outlier_index = random.randrange(4)
    count_index = 0

    for idx, (rs, cs) in enumerate(quadrant_slices):
        if idx == outlier_index:
            tile = _place_template(np.zeros((quad, quad), dtype=int), template_mask, outlier_color)
            grid[rs, cs] = tile
        else:
            tile = _scatter_points(quad, quad, counts[count_index], common_color)
            count_index += 1
            grid[rs, cs] = tile

    _validate_generated_case(grid)
    return grid


def _expected_from_rules(grid: np.ndarray) -> tuple[np.ndarray, int, tuple[int, int], tuple[int, int, int]]:
    quadrants, _split_r, _split_c = _quadrants(grid)
    colors = [_quadrant_foreground_color(q) for q in quadrants]
    outlier_index = next(i for i, color in enumerate(colors) if colors.count(color) == 1)
    template_bbox = _bbox(quadrants[outlier_index] == colors[outlier_index])
    assert template_bbox is not None
    template = _crop_bbox(quadrants[outlier_index], template_bbox)
    common_color = next(colors[i] for i in range(4) if i != outlier_index)
    common_counts = tuple(int(np.count_nonzero(quadrants[i] == common_color)) for i in range(4) if i != outlier_index)
    total = sum(common_counts)
    side = int(math.isqrt(total))
    return _tile_to_size(template, side), side, template.shape, common_counts


def _validate_generated_case(grid: np.ndarray) -> None:
    quadrants, split_r, split_c = _quadrants(grid)
    assert split_r == grid.shape[0] // 2 and split_c == grid.shape[1] // 2
    colors = [_quadrant_foreground_color(q) for q in quadrants]
    outlier_indices = [i for i, color in enumerate(colors) if colors.count(color) == 1]
    assert len(outlier_indices) == 1
    outlier_index = outlier_indices[0]

    common_indices = [i for i in range(4) if i != outlier_index]
    common_color = colors[common_indices[0]]
    common_counts = tuple(int(np.count_nonzero(quadrants[i] == common_color)) for i in common_indices)
    assert min(common_counts) >= 2
    assert len(set(common_counts)) >= 2, "avoid equal-per-quadrant shortcuts"

    total = sum(common_counts)
    side = int(math.isqrt(total))
    assert side in OUTPUT_SIDES and side * side == total

    outlier_quadrant = quadrants[outlier_index]
    outlier_color = colors[outlier_index]
    mask = outlier_quadrant == outlier_color
    comps = _components4(mask)
    assert len(comps) == 1, "outlier template must be one 4-connected component"
    bbox = _bbox(mask)
    assert bbox is not None
    cropped = _crop_bbox(outlier_quadrant, bbox)
    assert np.count_nonzero(cropped == outlier_color) in (3, 4, 5)
    assert np.count_nonzero(cropped == BACKGROUND) > 0, "cropping must preserve black holes in the tile"
    assert side % cropped.shape[0] == 0 and side % cropped.shape[1] == 0
    assert cropped.shape[0] < side or cropped.shape[1] < side, "tiling must repeat on at least one axis"

    expected = _tile_to_size(cropped, side)
    actual = main(grid)
    assert np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_case()
        expected, side, template_shape, common_counts = _expected_from_rules(grid)
        signature = (expected.shape, expected.tobytes())
        if signature in _SEEN_OUTPUT_SIGNATURES:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(signature)
        assert side == int(math.isqrt(sum(common_counts)))
        assert side % template_shape[0] == 0 and side % template_shape[1] == 0
        return grid
    raise RuntimeError(f"failed to generate a unique valid instance after {GENERATION_RETRIES} attempts")
