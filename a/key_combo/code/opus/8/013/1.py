import random
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path("/home/lixinyu/snap/new")
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from solvers.arc_utils import (  # noqa: E402
    Color,
    bbox_coords,
    bbox_h,
    bbox_union,
    bbox_w,
    crop_bbox,
    find_components,
    paste,
)


BG = Color.BLACK
OBSTACLE = Color.GREY
SPRITE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GENERATION_RETRIES = 600


@dataclass(frozen=True)
class Sprite:
    color: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    patch: np.ndarray


@dataclass(frozen=True)
class LandedSprite:
    color: int
    original_bbox: tuple[int, int, int, int]
    bbox: tuple[int, int, int, int]
    patch: np.ndarray
    coords: np.ndarray
    drop: int


def _build_patch(coords: np.ndarray, color: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    box = bbox_coords(coords)
    r0, c0, _r1, _c1 = box
    patch = np.full((bbox_h(box), bbox_w(box)), BG, dtype=int)
    local = coords - np.array([r0, c0], dtype=int)
    patch[local[:, 0], local[:, 1]] = int(color)
    return box, patch


def _extract_sprites(grid: np.ndarray) -> list[Sprite]:
    sprites: list[Sprite] = []
    seen_colors: set[int] = set()
    for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in {BG, OBSTACLE}:
            continue
        assert color not in seen_colors, "each sprite color must be unique"
        coords = np.asarray(comp["coords"], dtype=int)
        box, patch = _build_patch(coords, color)
        sprites.append(Sprite(color=color, coords=coords, bbox=box, patch=patch))
        seen_colors.add(color)
    assert 3 <= len(sprites) <= 5, "expected 3-5 sprites"
    return sprites


def _obstacle_rows_by_col(grid: np.ndarray) -> dict[int, list[int]]:
    rows_by_col: dict[int, list[int]] = {}
    for r, c in np.argwhere(grid == OBSTACLE):
        rows_by_col.setdefault(int(c), []).append(int(r))
    for rows in rows_by_col.values():
        rows.sort()
    return rows_by_col


def _drop_distance(coords: np.ndarray, height: int, rows_by_col: dict[int, list[int]]) -> int:
    best = height
    for r0, c0 in coords:
        r = int(r0)
        c = int(c0)
        rows = rows_by_col.get(c, [])
        idx = bisect_right(rows, r)
        floor_r = rows[idx] if idx < len(rows) else height
        best = min(best, int(floor_r - r - 1))
    assert best >= 0
    return int(best)


def _land_sprite(sprite: Sprite, height: int, rows_by_col: dict[int, list[int]]) -> LandedSprite:
    drop = _drop_distance(sprite.coords, height, rows_by_col)
    landed_coords = sprite.coords + np.array([drop, 0], dtype=int)
    landed_bbox = bbox_coords(landed_coords)
    return LandedSprite(
        color=sprite.color,
        original_bbox=sprite.bbox,
        bbox=landed_bbox,
        patch=sprite.patch,
        coords=landed_coords,
        drop=drop,
    )


def _render_layers(layers: list[LandedSprite], *, use_original_positions: bool = False) -> np.ndarray:
    bboxes = [layer.original_bbox if use_original_positions else layer.bbox for layer in layers]
    canvas_box = bbox_union(bboxes)
    assert canvas_box is not None
    out = np.full((bbox_h(canvas_box), bbox_w(canvas_box)), BG, dtype=int)
    base_r, base_c = int(canvas_box[0]), int(canvas_box[1])
    for layer in layers:
        box = layer.original_bbox if use_original_positions else layer.bbox
        paste(out, layer.patch, int(box[0] - base_r), int(box[1] - base_c), transparent=BG)
    return out


def _landing_order(layer: LandedSprite) -> tuple[int, int]:
    return (-int(layer.bbox[2]), int(layer.bbox[1]))


def _simulate(grid: np.ndarray) -> list[LandedSprite]:
    sprites = _extract_sprites(grid)
    rows_by_col = _obstacle_rows_by_col(grid)
    landed = [_land_sprite(sprite, grid.shape[0], rows_by_col) for sprite in sprites]
    for layer in landed:
        assert layer.drop > 0, "sprites should start suspended above their landing sites"
    return landed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each colored sprite onto the grey shelves, then overlay the landed sprites in height-determined z-order.

    Input structure:
    - The grid has black background, several grey obstacle pixels, and 3-5 single-color sprite components.
    - Each sprite is a 4-connected irregular shape in the upper half, and different sprites use different colors.
    - Sprite column ranges partially overlap, but the sprites do not overlap each other in the input.

    Rule:
    - Ignore the original sprite pixels and keep only each sprite's shape and columns.
    - Drop every sprite straight downward until the next step would hit a grey obstacle in one of its occupied columns or leave the bottom edge; all sprites fall independently.
    - Record each landed sprite's bbox bottom row, crop an output canvas to the union of all landed bboxes, and paste the sprites at their landed positions.
    - Draw sprites from larger landed bottom row to smaller landed bottom row; if two landed bottom rows tie, the left sprite is drawn first so the right sprite appears on top.
    - Grey obstacle pixels never appear in the output.

    Key insight:
    - The obstacle heights do double duty: they stop the fall and also encode the overlay order, because higher landings are painted later and therefore win overlaps.
    """
    g = np.asarray(grid, dtype=int)
    landed = sorted(_simulate(g), key=_landing_order)
    return _render_layers(landed)


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = [0, total]
    for _ in range(parts - 1):
        cuts.append(random.randint(0, total))
    cuts.sort()
    return [cuts[i + 1] - cuts[i] for i in range(parts)]


def _grow_mask(height: int, width: int, area: int) -> np.ndarray | None:
    for _ in range(80):
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        touch = sum((rr, cc) in cells for rr, cc in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)))
                        frontier.extend([(nr, nc)] * max(1, touch))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        box = bbox_coords(coords)
        cropped = mask[box[0] : box[2] + 1, box[1] : box[3] + 1]
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        if int(cropped.sum()) != area:
            continue
        if area == cropped.size:
            continue
        return cropped
    return None


def _random_sprite_mask() -> np.ndarray | None:
    for _ in range(100):
        height = random.randint(2, 3)
        width = random.randint(3, 4)
        max_area = min(7, height * width - 1)
        if max_area < 4:
            continue
        area = random.randint(4, max_area)
        mask = _grow_mask(height, width, area)
        if mask is not None:
            return mask
    return None


def _occupied_columns(mask: np.ndarray, left: int) -> set[int]:
    return {int(left + c) for c in np.unique(np.argwhere(mask)[:, 1])}


def _lowest_row_in_column(mask: np.ndarray, col: int) -> int:
    rows = np.flatnonzero(mask[:, col])
    return int(rows.max())


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[rr + int(top), cc + int(left)] = int(color)


def _sample_column_positions(width: int, sprite_widths: list[int]) -> list[int] | None:
    count = len(sprite_widths)
    for _ in range(160):
        order = list(range(count))
        random.shuffle(order)
        widths = [int(sprite_widths[idx]) for idx in order]
        lefts_sorted = [0] * count
        lefts_sorted[0] = random.randint(0, width - widths[0])
        ok = True
        for i in range(1, count):
            prev_left = lefts_sorted[i - 1]
            prev_right = prev_left + widths[i - 1] - 1
            if widths[i - 1] < 2 or widths[i] < 2:
                ok = False
                break
            left = prev_right
            if left + widths[i] > width:
                ok = False
                break
            lefts_sorted[i] = left
        if not ok:
            continue
        lefts = [0] * count
        for idx, left in zip(order, lefts_sorted):
            lefts[idx] = int(left)
        return lefts
    return None


def _sample_row_positions(height: int, sprite_heights: list[int]) -> list[int] | None:
    upper_limit = (height + 1) // 2
    total_h = sum(sprite_heights)
    slack = upper_limit - total_h
    if slack < 0:
        return None
    gaps = _random_composition(slack, len(sprite_heights) + 1)
    tops: list[int] = []
    row = gaps[0]
    for h, gap in zip(sprite_heights, gaps[1:]):
        tops.append(row)
        row += h + gap
    if any(top + h > upper_limit for top, h in zip(tops, sprite_heights)):
        return None
    return tops


def _count_overlap_pairs(layers: list[LandedSprite]) -> int:
    as_sets = [set(map(tuple, layer.coords.tolist())) for layer in layers]
    pairs = 0
    for i in range(len(as_sets)):
        for j in range(i + 1, len(as_sets)):
            if as_sets[i] & as_sets[j]:
                pairs += 1
    return pairs


def _same_order(a: list[LandedSprite], b: list[LandedSprite]) -> bool:
    return [layer.color for layer in a] == [layer.color for layer in b]


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        landed = _simulate(grid)
    except Exception:
        return False

    if len({int(layer.bbox[2]) for layer in landed}) < 2:
        return False
    if _count_overlap_pairs(landed) < 2:
        return False

    actual_layers = sorted(landed, key=_landing_order)
    actual = _render_layers(actual_layers)

    if any(layer.drop < 2 for layer in landed):
        return False

    reverse_layers = sorted(landed, key=lambda layer: (int(layer.bbox[2]), int(layer.bbox[1])))
    if np.array_equal(actual, _render_layers(reverse_layers)):
        return False

    original_order_layers = sorted(landed, key=lambda layer: (int(layer.original_bbox[0]), int(layer.original_bbox[1])))
    if _same_order(actual_layers, original_order_layers):
        return False
    if np.array_equal(actual, _render_layers(original_order_layers)):
        return False

    if np.array_equal(actual, _render_layers(actual_layers, use_original_positions=True)):
        return False

    tied = {}
    for layer in landed:
        tied.setdefault(int(layer.bbox[2]), []).append(layer)
    has_meaningful_tie = False
    for group in tied.values():
        if len(group) < 2:
            continue
        has_meaningful_tie = True
        tie_reversed = sorted(landed, key=lambda layer: (-int(layer.bbox[2]), -int(layer.bbox[1])))
        if np.array_equal(actual, _render_layers(tie_reversed)):
            return False
        break

    obstacle_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == OBSTACLE)]
    if not (3 <= len(obstacle_cells) <= 5):
        return False
    if len({r for r, _c in obstacle_cells}) < 2 or len({c for _r, c in obstacle_cells}) < 2:
        return False

    sprites = _extract_sprites(grid)
    intervals = [(int(sprite.bbox[1]), int(sprite.bbox[3])) for sprite in sprites]
    overlap_intervals = 0
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            if max(intervals[i][0], intervals[j][0]) <= min(intervals[i][1], intervals[j][1]):
                overlap_intervals += 1
    if overlap_intervals < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.randint(3, 5)
    height = random.randint(max(18, sprite_count * 3 + 6), 22)
    width = random.randint(12, 16)

    masks: list[np.ndarray] = []
    for _ in range(sprite_count):
        mask = _random_sprite_mask()
        if mask is None:
            return None
        masks.append(mask)

    lefts = _sample_column_positions(width, [mask.shape[1] for mask in masks])
    if lefts is None:
        return None
    tops = _sample_row_positions(height, [mask.shape[0] for mask in masks])
    if tops is None:
        return None

    color_order = random.sample(SPRITE_COLORS, sprite_count)
    grid = np.full((height, width), BG, dtype=int)

    sprite_columns = [_occupied_columns(mask, left) for mask, left in zip(masks, lefts)]
    unique_columns = []
    for i, cols in enumerate(sprite_columns):
        other_cols = set().union(*(sprite_columns[j] for j in range(sprite_count) if j != i))
        unique = cols - other_cols
        if not unique:
            return None
        unique_columns.append(sorted(unique))

    for mask, top, left, color in zip(masks, tops, lefts, color_order):
        _paint_mask(grid, mask, top, left, color)

    orig_bottoms = [int(top + mask.shape[0] - 1) for mask, top in zip(masks, tops)]
    bottom_sprite = random.randrange(sprite_count) if random.random() < 0.35 else None
    tie_pair = random.sample(range(sprite_count), 2) if sprite_count >= 3 and random.random() < 0.5 else []
    shared_landing = random.randint(height - 6, height - 2) if tie_pair else None

    target_bottoms: list[int] = []
    for i, bottom in enumerate(orig_bottoms):
        if i == bottom_sprite:
            target = height - 1
        elif i in tie_pair:
            target = int(shared_landing)
        else:
            lo = max(bottom + 2, height - 6)
            hi = height - 2
            if lo > hi:
                return None
            target = random.randint(lo, hi)
        if target <= bottom:
            return None
        target_bottoms.append(int(target))

    obstacles: set[tuple[int, int]] = set()
    all_sprite_columns = set().union(*sprite_columns)
    for i, (mask, top, left, target_bottom) in enumerate(zip(masks, tops, lefts, target_bottoms)):
        if i == bottom_sprite:
            continue
        control_col = random.choice(unique_columns[i])
        local_col = int(control_col - left)
        local_low = _lowest_row_in_column(mask, local_col)
        drop = target_bottom - (top + mask.shape[0] - 1)
        obstacle_row = int(top + local_low + drop + 1)
        if not (0 <= obstacle_row < height):
            return None
        if grid[obstacle_row, control_col] != BG:
            return None
        obstacles.add((obstacle_row, control_col))

    if len(obstacles) > 5:
        return None

    desired_obstacle_count = random.randint(max(3, len(obstacles)), 5)
    free_columns = [c for c in range(width) if c not in all_sprite_columns]
    if len(free_columns) < desired_obstacle_count - len(obstacles):
        return None

    decoy_attempts = 0
    while len(obstacles) < desired_obstacle_count and decoy_attempts < 80:
        decoy_attempts += 1
        c = random.choice(free_columns)
        r = random.randint(height // 2, height - 2)
        if grid[r, c] != BG or (r, c) in obstacles:
            continue
        obstacles.add((r, c))
    if len(obstacles) != desired_obstacle_count:
        return None

    for r, c in obstacles:
        grid[r, c] = OBSTACLE

    if not _semantic_ok(grid):
        return None

    actual = main(grid)
    if actual.ndim != 2 or max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_RETRIES} attempts")
