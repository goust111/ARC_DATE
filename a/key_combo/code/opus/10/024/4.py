from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np


BACKGROUND = 0
SPRITE_COLOR = 5
FRAME_COLOR_POOL = (1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_from_mask(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    q = deque([start])
    seen = {start}
    coords: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return np.array(coords, dtype=int)


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp = _component_from_mask(mask, (r, c))
            seen[comp[:, 0], comp[:, 1]] = True
            comps.append(comp)
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_nonzero(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr != 0)
    if coords.size == 0:
        raise AssertionError("expected a non-empty sprite")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_unit_hollow_frame(grid: np.ndarray, color: int, comp: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    region = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    region[comp[:, 0] - r0, comp[:, 1] - c0] = True
    border = np.zeros_like(region, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(region[border]):
        return False
    if np.any(region[~border]):
        return False
    return int(comp.shape[0]) == int(border.sum())


def _rotate_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, -int(quarter_turns))


def _render_output_row(tiles: list[np.ndarray]) -> np.ndarray:
    assert tiles, "expected at least one tile"
    height = max(tile.shape[0] for tile in tiles)
    width = sum(tile.shape[1] for tile in tiles) + (len(tiles) - 1)
    out = np.zeros((height, width), dtype=int)
    cursor = 0
    for i, tile in enumerate(tiles):
        out[: tile.shape[0], cursor : cursor + tile.shape[1]] = tile
        cursor += tile.shape[1]
        if i + 1 != len(tiles):
            cursor += 1
    return out


def _analyze_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_colors = [int(c) for c in np.unique(g) if int(c) not in {BACKGROUND, SPRITE_COLOR}]
    assert len(frame_colors) == 3, "expected exactly three frame colors"

    gray_components = _find_components_of_color(g, SPRITE_COLOR)
    assert len(gray_components) == 3, "expected exactly three gray sprites"

    groups: list[dict] = []
    used_gray: set[int] = set()
    for color in frame_colors:
        frame_components = _find_components_of_color(g, color)
        assert frame_components, "each frame color must appear"
        for comp in frame_components:
            assert _is_unit_hollow_frame(g, color, comp), "expected disconnected 1px hollow rectangles"

        bboxes = sorted((_bbox(comp) for comp in frame_components), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        centers = {(b[0] + b[2], b[1] + b[3]) for b in bboxes}
        assert len(centers) == 1, "frames of one group must be concentric"
        for inner, outer in zip(bboxes, bboxes[1:]):
            assert outer[0] < inner[0] and outer[1] < inner[1], "outer frame must strictly enclose inner frame"
            assert outer[2] > inner[2] and outer[3] > inner[3], "outer frame must strictly enclose inner frame"

        inner = bboxes[0]
        hole_bbox = (inner[0] + 1, inner[1] + 1, inner[2] - 1, inner[3] - 1)
        matches = []
        for idx, comp in enumerate(gray_components):
            if idx in used_gray:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if hole_bbox[0] <= r0 and hole_bbox[1] <= c0 and r1 <= hole_bbox[2] and c1 <= hole_bbox[3]:
                matches.append((idx, comp))
        assert len(matches) == 1, "each inner frame must contain exactly one gray sprite"
        gray_idx, gray_comp = matches[0]
        used_gray.add(gray_idx)

        sprite_mask = np.zeros_like(g, dtype=int)
        sprite_mask[gray_comp[:, 0], gray_comp[:, 1]] = 1
        sprite_crop = _crop_nonzero(sprite_mask)

        groups.append(
            {
                "color": int(color),
                "depth": len(frame_components),
                "outer_bbox": bboxes[-1],
                "sprite": sprite_crop,
            }
        )

    assert len(used_gray) == 3, "all sprites must be assigned"
    groups.sort(key=lambda item: int(item["outer_bbox"][1]))
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode nested frame depth into sprite rotation, recolor each sprite by its frame, and order results left-to-right.

    Input structure:
    - The grid has black background and exactly three disconnected nested-frame groups.
    - Each group uses one non-black, non-grey color for one or more concentric 1px hollow rectangles, with a grey 4-connected sprite inside the innermost frame.
    - The three groups have depths 1, 2, and 3 in random left-to-right order, and the sprites share the same base shape.

    Rule:
    - Group rectangles by color and count how many concentric frames each color has.
    - Extract the grey sprite inside the innermost frame, rotate it clockwise by 90 degrees times the frame depth, and recolor it with that group's frame color.
    - Crop each rotated sprite tightly, then arrange the three cropped sprites from left to right using the groups' outer-frame positions, with one black column between neighbors.

    Key insight:
    - The nested-frame topology is the rotation code: frame depth sets the turn amount, while the outer frame color and horizontal position supply the recolor and output ordering.
    """
    groups = _analyze_grid(np.asarray(grid, dtype=int))
    tiles: list[np.ndarray] = []
    for group in groups:
        rotated = _rotate_cw(group["sprite"], group["depth"])
        tile = np.where(rotated != 0, int(group["color"]), 0).astype(int)
        tiles.append(_crop_nonzero(tile))
    return _render_output_row(tiles)


_SPRITE_LIBRARY = (
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
)


def _all_distinct_rotations(sprite: np.ndarray) -> bool:
    rotations = [_crop_nonzero(_rotate_cw(sprite, k)) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            if rotations[i].shape == rotations[j].shape and np.array_equal(rotations[i], rotations[j]):
                return False
    return True


def _sample_base_sprite() -> np.ndarray:
    while True:
        sprite = random.choice(_SPRITE_LIBRARY).copy()
        sprite = _rotate_cw(sprite, random.randrange(4))
        if _all_distinct_rotations(sprite):
            return sprite


def _paint_frame_border(patch: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    patch[top, left : left + width] = int(color)
    patch[top + height - 1, left : left + width] = int(color)
    patch[top : top + height, left] = int(color)
    patch[top : top + height, left + width - 1] = int(color)


def _build_group_patch(
    depth: int,
    color: int,
    sprite: np.ndarray,
    hole_height: int,
    hole_width: int,
) -> tuple[np.ndarray, dict]:
    inner_height = hole_height + 2
    inner_width = hole_width + 2
    outer_height = inner_height + 4 * (depth - 1)
    outer_width = inner_width + 4 * (depth - 1)
    patch = np.zeros((outer_height, outer_width), dtype=int)

    for layer in range(depth):
        top = 2 * (depth - 1 - layer)
        left = 2 * (depth - 1 - layer)
        height = inner_height + 4 * layer
        width = inner_width + 4 * layer
        _paint_frame_border(patch, top, left, height, width, color)

    inner_top = 2 * (depth - 1)
    inner_left = 2 * (depth - 1)
    hole_top = inner_top + 1
    hole_left = inner_left + 1
    max_row_offset = hole_height - sprite.shape[0]
    max_col_offset = hole_width - sprite.shape[1]
    sprite_row = hole_top + random.randint(0, max_row_offset)
    sprite_col = hole_left + random.randint(0, max_col_offset)
    patch[sprite_row : sprite_row + sprite.shape[0], sprite_col : sprite_col + sprite.shape[1]][sprite != 0] = SPRITE_COLOR

    meta = {
        "color": int(color),
        "depth": int(depth),
        "sprite_row": int(sprite_row),
        "sprite_col": int(sprite_col),
        "outer_shape": patch.shape,
    }
    return patch, meta


def _construct_expected_output(base_sprite: np.ndarray, ordered_groups: list[dict]) -> np.ndarray:
    tiles = []
    for group in ordered_groups:
        rotated = _rotate_cw(base_sprite, int(group["depth"]))
        tile = np.where(rotated != 0, int(group["color"]), 0).astype(int)
        tiles.append(_crop_nonzero(tile))
    return _render_output_row(tiles)


def _distribute_extra(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _validate_semantics(
    grid: np.ndarray,
    base_sprite: np.ndarray,
    ordered_groups: list[dict],
    expected_output: np.ndarray,
) -> bool:
    try:
        actual = main(grid)
        analyzed = _analyze_grid(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected_output):
        return False

    if [group["depth"] for group in analyzed] != [group["depth"] for group in ordered_groups]:
        return False
    if [group["color"] for group in analyzed] != [group["color"] for group in ordered_groups]:
        return False

    rotated = [_crop_nonzero(_rotate_cw(base_sprite, d)) for d in (1, 2, 3)]
    for i in range(3):
        for j in range(i + 1, 3):
            if rotated[i].shape == rotated[j].shape and np.array_equal(rotated[i], rotated[j]):
                return False

    if [group["depth"] for group in ordered_groups] == [1, 2, 3]:
        return False

    unrotated_output = _render_output_row(
        [np.where(base_sprite != 0, int(group["color"]), 0).astype(int) for group in ordered_groups]
    )
    if np.array_equal(unrotated_output, expected_output):
        return False

    color_sorted = sorted(ordered_groups, key=lambda item: int(item["color"]))
    if color_sorted != ordered_groups:
        color_sorted_output = _construct_expected_output(base_sprite, color_sorted)
        if np.array_equal(color_sorted_output, expected_output):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    base_sprite = _sample_base_sprite()
    depths_in_order = random.sample([1, 2, 3], 3)
    colors_in_order = random.sample(FRAME_COLOR_POOL, 3)

    hole_widths = {1: 3, 2: 3, 3: 3}
    if random.random() < 0.5:
        widened_depth = random.choice([1, 2])
        hole_widths[widened_depth] = 4

    hole_heights = {}
    for depth in (1, 2, 3):
        min_h = max(3, base_sprite.shape[0])
        max_h = min(5, base_sprite.shape[0] + 2)
        hole_heights[depth] = random.randint(min_h, max_h)

    group_specs = []
    patches = []
    widths = []
    heights = []
    for depth, color in zip(depths_in_order, colors_in_order):
        patch, meta = _build_group_patch(
            depth=depth,
            color=color,
            sprite=base_sprite,
            hole_height=hole_heights[depth],
            hole_width=hole_widths[depth],
        )
        group_specs.append(meta)
        patches.append(patch)
        widths.append(patch.shape[1])
        heights.append(patch.shape[0])

    min_width = sum(widths) + 2
    if min_width > 30:
        return None
    grid_width = random.randint(min_width, 30)
    extra_width = grid_width - min_width
    left_extra, gap1_extra, gap2_extra, right_extra = _distribute_extra(extra_width, 4)
    left_margin = left_extra
    gaps = [1 + gap1_extra, 1 + gap2_extra]
    _right_margin = right_extra

    grid_height = random.randint(max(max(heights) + 1, 18), min(max(heights) + 6, 24))
    if grid_height > 30:
        return None

    grid = np.zeros((grid_height, grid_width), dtype=int)
    col = left_margin
    ordered_groups: list[dict] = []
    for idx, (patch, meta) in enumerate(zip(patches, group_specs)):
        max_top = grid_height - patch.shape[0]
        if max_top < 0:
            return None
        top = random.randint(0, max_top)
        grid[top : top + patch.shape[0], col : col + patch.shape[1]] = patch
        ordered_groups.append(
            {
                "color": int(meta["color"]),
                "depth": int(meta["depth"]),
                "left": int(col),
                "top": int(top),
            }
        )
        col += patch.shape[1]
        if idx < 2:
            col += gaps[idx]

    expected_output = _construct_expected_output(base_sprite, ordered_groups)
    if expected_output.shape[0] > 30 or expected_output.shape[1] > 30:
        return None
    if not _validate_semantics(grid, base_sprite, ordered_groups, expected_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame rotation puzzle")
