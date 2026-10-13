from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


for parent in Path(__file__).resolve().parents:
    for candidate in (parent / "new" / "solvers", parent / "new"):
        if (candidate / "arc_utils.py").exists():
            sys.path.append(str(candidate))
            break
    else:
        continue
    break
else:
    raise ImportError("could not locate arc_utils.py")

from arc_utils import Color, bbox_coords, find_components, get_bg, neighbors4  # type: ignore


Coord = tuple[int, int]

BG = Color.BLACK
INDICATOR = Color.MAROON
OBJECT_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
DIRECTIONS = ("up", "right", "down", "left")

GRID_SIDE_RANGE = (20, 28)
OBJECT_COUNT_RANGE = (3, 4)
AREA_VALUES = tuple(range(4, 11))
MAX_SHAPE_SIDE = 5
SHAPE_RETRIES = 200
GEN_RETRIES = 500


@dataclass(frozen=True)
class Sprite:
    color: int
    area: int
    direction: str
    cells: frozenset[Coord]
    indicator: Coord


def _normalize(cells: set[Coord] | list[Coord] | tuple[Coord, ...]) -> tuple[Coord, ...]:
    arr = np.array(sorted(cells), dtype=int)
    r0, c0, _r1, _c1 = bbox_coords(arr)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in arr))


def _translated(cells: tuple[Coord, ...] | frozenset[Coord], top: int, left: int) -> frozenset[Coord]:
    return frozenset((int(top + r), int(left + c)) for r, c in cells)


def _shape_dims(cells: tuple[Coord, ...]) -> tuple[int, int]:
    arr = np.array(cells, dtype=int)
    _r0, _c0, r1, c1 = bbox_coords(arr)
    return int(r1 + 1), int(c1 + 1)


def _is_connected(cells: set[Coord]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _generate_shape(area: int) -> tuple[Coord, ...] | None:
    for _ in range(SHAPE_RETRIES):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: list[Coord] = []
            seen_frontier: set[Coord] = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) in cells or (nr, nc) in seen_frontier:
                        continue
                    trial = cells | {(int(nr), int(nc))}
                    offsets = _normalize(trial)
                    h, w = _shape_dims(offsets)
                    if h > MAX_SHAPE_SIDE or w > MAX_SHAPE_SIDE:
                        continue
                    seen_frontier.add((int(nr), int(nc)))
                    frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue
        offsets = _normalize(cells)
        h, w = _shape_dims(offsets)
        if h < 2 or w < 2:
            continue
        if len(offsets) == h * w:
            continue
        if not any(r == 0 for r, _c in offsets):
            continue
        if not any(c == 0 for _r, c in offsets):
            continue
        if not any(r == h - 1 for r, _c in offsets):
            continue
        if not any(c == w - 1 for _r, c in offsets):
            continue
        return offsets
    return None


def _band_cells(cells: frozenset[Coord], direction: str, shape: tuple[int, int]) -> frozenset[Coord]:
    h, w = map(int, shape)
    if direction == "up":
        top_by_col: dict[int, int] = {}
        for r, c in cells:
            top_by_col[c] = min(top_by_col.get(c, r), r)
        return frozenset((rr, c) for c, top in top_by_col.items() for rr in range(0, top))
    if direction == "down":
        bot_by_col: dict[int, int] = {}
        for r, c in cells:
            bot_by_col[c] = max(bot_by_col.get(c, r), r)
        return frozenset((rr, c) for c, bot in bot_by_col.items() for rr in range(bot + 1, h))
    if direction == "left":
        left_by_row: dict[int, int] = {}
        for r, c in cells:
            left_by_row[r] = min(left_by_row.get(r, c), c)
        return frozenset((r, cc) for r, left in left_by_row.items() for cc in range(0, left))
    if direction == "right":
        right_by_row: dict[int, int] = {}
        for r, c in cells:
            right_by_row[r] = max(right_by_row.get(r, c), c)
        return frozenset((r, cc) for r, right in right_by_row.items() for cc in range(right + 1, w))
    raise AssertionError(f"unknown direction {direction}")


def _indicator_side(bbox_vals: tuple[int, int, int, int], indicator: Coord) -> str:
    r0, c0, r1, c1 = map(int, bbox_vals)
    ir, ic = map(int, indicator)
    if ir == r0 - 1 and c0 <= ic <= c1:
        return "up"
    if ic == c1 + 1 and r0 <= ir <= r1:
        return "right"
    if ir == r1 + 1 and c0 <= ic <= c1:
        return "down"
    if ic == c0 - 1 and r0 <= ir <= r1:
        return "left"
    raise AssertionError("indicator must sit just outside one bbox side")


def _indicator_options(cells: frozenset[Coord], direction: str) -> list[Coord]:
    arr = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    cell_set = set(cells)
    options: list[Coord] = []
    if direction == "up":
        for c in range(c0, c1 + 1):
            if (r0, c) in cell_set:
                options.append((r0 - 1, c))
    elif direction == "down":
        for c in range(c0, c1 + 1):
            if (r1, c) in cell_set:
                options.append((r1 + 1, c))
    elif direction == "left":
        for r in range(r0, r1 + 1):
            if (r, c0) in cell_set:
                options.append((r, c0 - 1))
    elif direction == "right":
        for r in range(r0, r1 + 1):
            if (r, c1) in cell_set:
                options.append((r, c1 + 1))
    else:
        raise AssertionError(f"unknown direction {direction}")
    return options


def _expand_with_padding(cells: frozenset[Coord]) -> set[Coord]:
    expanded: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((int(r + dr), int(c + dc)))
    return expanded


def _make_sprite_variants(
    shape: tuple[Coord, ...],
    top: int,
    left: int,
    direction: str,
    color: int,
    area: int,
) -> list[Sprite]:
    cells = _translated(shape, top, left)
    return [
        Sprite(color=color, area=area, direction=direction, cells=cells, indicator=indicator)
        for indicator in _indicator_options(cells, direction)
    ]


def _guided_candidates(
    shape: tuple[Coord, ...],
    direction: str,
    color: int,
    area: int,
    grid_shape: tuple[int, int],
    cross: Coord,
) -> list[Sprite]:
    h, w = map(int, grid_shape)
    shape_h, shape_w = _shape_dims(shape)
    cross_r, cross_c = map(int, cross)
    out: list[Sprite] = []

    if direction in ("up", "down"):
        cols = sorted({c for _r, c in shape})
        for col_off in cols:
            left = cross_c - col_off
            if not (0 <= left and left + shape_w <= w):
                continue
            col_rows = [r for r, c in shape if c == col_off]
            if direction == "up":
                top_min = min(col_rows)
                row_start = max(1, cross_r - top_min + 1)
                row_stop = h - shape_h
                for top in range(row_start, row_stop + 1):
                    out.extend(_make_sprite_variants(shape, top, left, direction, color, area))
            else:
                bot_max = max(col_rows)
                row_stop = min(h - shape_h - 1, cross_r - bot_max - 1)
                for top in range(0, row_stop + 1):
                    out.extend(_make_sprite_variants(shape, top, left, direction, color, area))
    else:
        rows = sorted({r for r, _c in shape})
        for row_off in rows:
            top = cross_r - row_off
            if not (0 <= top and top + shape_h <= h):
                continue
            row_cols = [c for r, c in shape if r == row_off]
            if direction == "left":
                left_min = min(row_cols)
                col_start = max(1, cross_c - left_min + 1)
                col_stop = w - shape_w
                for left in range(col_start, col_stop + 1):
                    out.extend(_make_sprite_variants(shape, top, left, direction, color, area))
            else:
                right_max = max(row_cols)
                col_stop = min(w - shape_w - 1, cross_c - right_max - 1)
                for left in range(0, col_stop + 1):
                    out.extend(_make_sprite_variants(shape, top, left, direction, color, area))
    random.shuffle(out)
    return out


def _all_candidates(
    shape: tuple[Coord, ...],
    direction: str,
    color: int,
    area: int,
    grid_shape: tuple[int, int],
) -> list[Sprite]:
    h, w = map(int, grid_shape)
    shape_h, shape_w = _shape_dims(shape)
    top_start, top_stop = 0, h - shape_h
    left_start, left_stop = 0, w - shape_w
    if direction == "up":
        top_start = 1
    elif direction == "down":
        top_stop = h - shape_h - 1
    elif direction == "left":
        left_start = 1
    elif direction == "right":
        left_stop = w - shape_w - 1

    if top_start > top_stop or left_start > left_stop:
        return []

    out: list[Sprite] = []
    for top in range(top_start, top_stop + 1):
        for left in range(left_start, left_stop + 1):
            out.extend(_make_sprite_variants(shape, top, left, direction, color, area))
    random.shuffle(out)
    return out


def _patch_cells(sprite: Sprite) -> frozenset[Coord]:
    return sprite.cells | {sprite.indicator}


def _conflicts(sprite: Sprite, blocked: set[Coord]) -> bool:
    return any(cell in blocked for cell in _patch_cells(sprite))


def _render(shape: tuple[int, int], sprites: list[Sprite], smallest_on_top: bool = True) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    ordered = sorted(sprites, key=lambda sprite: -sprite.area)
    if not smallest_on_top:
        ordered = list(reversed(ordered))
    for sprite in ordered:
        paint = sprite.cells | _band_cells(sprite.cells, sprite.direction, shape) | {sprite.indicator}
        for r, c in paint:
            out[r, c] = int(sprite.color)
    return out


def _extract_sprites(grid: np.ndarray) -> list[Sprite]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    sprites: list[Sprite] = []
    used_indicators: set[Coord] = set()

    for comp in comps:
        color = int(comp["color"])
        if color == INDICATOR:
            continue
        cells = frozenset((int(r), int(c)) for r, c in comp["coords"])
        indicator_neighbors: set[Coord] = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == INDICATOR:
                    indicator_neighbors.add((int(nr), int(nc)))
        assert len(indicator_neighbors) == 1, "each sprite must have exactly one adjacent indicator"
        indicator = next(iter(indicator_neighbors))
        side = _indicator_side(comp["bbox"], indicator)
        used_indicators.add(indicator)
        sprites.append(
            Sprite(
                color=color,
                area=int(comp["area"]),
                direction=side,
                cells=cells,
                indicator=indicator,
            )
        )

    assert sprites, "expected at least one non-indicator sprite"
    indicator_pixels = {tuple(map(int, rc)) for rc in np.argwhere(g == INDICATOR)}
    assert indicator_pixels == used_indicators, "every indicator pixel must belong to exactly one sprite"
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each irregular sprite toward the grid edge on its indicated side, then resolve overlaps with smaller sprites on top.

    Input structure:
    - A black background contains 3-4 monochromatic 4-connected irregular sprites in distinct non-black colors.
    - Each sprite has exactly one maroon indicator pixel just outside one side of its bounding box and 4-neighbor-adjacent to that side.
    - Sprite areas are distinct, and different sprites may later produce overlapping extension bands.

    Rule:
    - Identify each colored sprite and its unique adjacent maroon indicator.
    - The indicator side relative to the sprite bbox gives the extension direction: top/right/bottom/left means extend up/right/down/left.
    - Build a directional band projection by filling, for every occupied column or row, all cells from the sprite's outermost cell on that side to the canvas edge.
    - Paint every sprite together with its band in descending original-area order, so smaller sprites are painted later and therefore cover larger ones in overlaps.
    - Remove every maroon indicator by repainting that cell as part of its sprite layer.

    Key insight:
    - The indicator determines which silhouette edge projects to the border, and the original pixel count determines the reverse layering priority in the overlap region.
    """
    g = np.asarray(grid, dtype=int)
    sprites = _extract_sprites(g)
    areas = [sprite.area for sprite in sprites]
    assert len(set(areas)) == len(areas), "sprite areas must be distinct"
    return _render(g.shape, sprites, smallest_on_top=True)


def _choose_shapes(areas: list[int]) -> list[tuple[Coord, ...]] | None:
    shapes: list[tuple[Coord, ...]] = []
    seen: set[tuple[Coord, ...]] = set()
    for area in areas:
        choices: list[tuple[Coord, ...]] = []
        for _ in range(25):
            shape = _generate_shape(area)
            if shape is None:
                continue
            if shape in seen:
                continue
            choices.append(shape)
        if not choices:
            return None
        shape = random.choice(choices)
        seen.add(shape)
        shapes.append(shape)
    return shapes


def _pair_score(a: Sprite, b: Sprite, shape: tuple[int, int]) -> tuple[int, int, int]:
    band_a = _band_cells(a.cells, a.direction, shape)
    band_b = _band_cells(b.cells, b.direction, shape)
    inter = band_a & band_b
    if not inter:
        return (10_000, 10_000, 10_000)
    body_hits = len(band_a & b.cells) + len(band_b & a.cells)
    size_penalty = abs(len(inter) - 3)
    return (body_hits, size_penalty, len(inter))


def _pick_anchor_pair(
    shapes: list[tuple[Coord, ...]],
    colors: list[int],
    areas: list[int],
    grid_shape: tuple[int, int],
    cross: Coord,
) -> list[Sprite] | None:
    h, w = map(int, grid_shape)
    vertical_idx = 0
    horizontal_idx = 1
    vertical_direction = random.choice(("up", "down"))
    horizontal_direction = random.choice(("left", "right"))

    vertical_candidates = _guided_candidates(
        shapes[vertical_idx], vertical_direction, colors[vertical_idx], areas[vertical_idx], (h, w), cross
    )
    horizontal_candidates = _guided_candidates(
        shapes[horizontal_idx], horizontal_direction, colors[horizontal_idx], areas[horizontal_idx], (h, w), cross
    )
    if not vertical_candidates or not horizontal_candidates:
        return None

    best_pair: tuple[tuple[int, int, int], Sprite, Sprite] | None = None
    for vert in vertical_candidates:
        vert_patch = _patch_cells(vert)
        vert_blocked = _expand_with_padding(vert_patch)
        for horiz in horizontal_candidates:
            if _conflicts(horiz, vert_blocked):
                continue
            score = _pair_score(vert, horiz, grid_shape)
            if score[0] >= 10_000:
                continue
            if score[2] > 8:
                continue
            if best_pair is None or score < best_pair[0]:
                best_pair = (score, vert, horiz)
                if score[0] == 0 and score[1] == 0:
                    break
        if best_pair is not None and best_pair[0][0] == 0 and best_pair[0][1] == 0:
            break
    if best_pair is None:
        return None
    return [best_pair[1], best_pair[2]]


def _choose_extra_sprite(
    shape: tuple[Coord, ...],
    color: int,
    area: int,
    grid_shape: tuple[int, int],
    placed: list[Sprite],
) -> Sprite | None:
    blocked = set().union(*(_expand_with_padding(_patch_cells(sprite)) for sprite in placed))
    existing_bands = [_band_cells(sprite.cells, sprite.direction, grid_shape) for sprite in placed]
    existing_band_union = set().union(*existing_bands) if existing_bands else set()
    existing_body_union = set().union(*(sprite.cells | {sprite.indicator} for sprite in placed))

    scored: list[tuple[int, int, Sprite]] = []
    for direction in random.sample(list(DIRECTIONS), k=len(DIRECTIONS)):
        for cand in _all_candidates(shape, direction, color, area, grid_shape):
            if _conflicts(cand, blocked):
                continue
            band = _band_cells(cand.cells, cand.direction, grid_shape)
            band_overlap = len(band & existing_band_union)
            body_hits = len(band & existing_body_union)
            if band_overlap > 8 or body_hits > 2:
                continue
            scored.append((body_hits, band_overlap, cand))
    if not scored:
        return None

    scored.sort(key=lambda item: (item[0], item[1], -item[2].area, random.random()))
    return scored[0][2]


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        sprites = _extract_sprites(grid)
        out = main(grid)
    except Exception:
        return False

    if not (3 <= len(sprites) <= 4):
        return False
    if sorted(sprite.color for sprite in sprites) != sorted(set(sprite.color for sprite in sprites)):
        return False
    areas = [sprite.area for sprite in sprites]
    if len(set(areas)) != len(areas):
        return False
    if any(not (4 <= sprite.area <= 10) for sprite in sprites):
        return False

    band_sets = [_band_cells(sprite.cells, sprite.direction, grid.shape) for sprite in sprites]
    overlap_cells = set()
    for i in range(len(band_sets)):
        for j in range(i + 1, len(band_sets)):
            overlap_cells |= band_sets[i] & band_sets[j]
    if not overlap_cells:
        return False

    cover_count: dict[Coord, int] = {}
    for sprite, band in zip(sprites, band_sets):
        for cell in sprite.cells | band | {sprite.indicator}:
            cover_count[cell] = cover_count.get(cell, 0) + 1
    multi = sum(1 for count in cover_count.values() if count >= 2)
    if multi > 18:
        return False

    reverse = _render(grid.shape, sprites, smallest_on_top=False)
    if np.array_equal(out, reverse):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(list(OBJECT_COLORS), n_objects)
    areas = random.sample(list(AREA_VALUES), n_objects)
    shapes = _choose_shapes(areas)
    if shapes is None:
        return None

    cross = (random.randint(5, h - 6), random.randint(5, w - 6))
    placed = _pick_anchor_pair(shapes, colors, areas, (h, w), cross)
    if placed is None:
        return None

    for idx in range(2, n_objects):
        sprite = _choose_extra_sprite(shapes[idx], colors[idx], areas[idx], (h, w), placed)
        if sprite is None:
            return None
        placed.append(sprite)

    grid = np.full((h, w), BG, dtype=int)
    for sprite in placed:
        for r, c in sprite.cells:
            if grid[r, c] != BG:
                return None
            grid[r, c] = int(sprite.color)
        ir, ic = sprite.indicator
        if grid[ir, ic] != BG:
            return None
        grid[ir, ic] = INDICATOR

    if not _validate_semantics(grid):
        return None

    expected = _render(grid.shape, placed, smallest_on_top=True)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid extend-and-overlay ARC input")
