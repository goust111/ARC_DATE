from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
BLOCKER = 8

SPRITE_PALETTE = (RED, YELLOW, GREY, PINK, ORANGE)
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component(coords: list[tuple[int, int]], color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        out[r - r0, c - c0] = color
    return out


def _concat_top_aligned(crops: list[np.ndarray]) -> np.ndarray:
    assert crops, "expected at least one crop"
    height = max(int(c.shape[0]) for c in crops)
    width = sum(int(c.shape[1]) for c in crops)
    out = np.zeros((height, width), dtype=int)
    col = 0
    for crop in crops:
        h, w = map(int, crop.shape)
        out[:h, col : col + w] = crop
        col += w
    return out


def _components_by_color(grid: np.ndarray, colors: list[int]) -> dict[int, list[list[tuple[int, int]]]]:
    h, w = grid.shape
    out: dict[int, list[list[tuple[int, int]]]] = {}
    for color in colors:
        seen: set[tuple[int, int]] = set()
        comps: list[list[tuple[int, int]]] = []
        for r in range(h):
            for c in range(w):
                if int(grid[r, c]) != color or (r, c) in seen:
                    continue
                stack = [(r, c)]
                seen.add((r, c))
                comp: list[tuple[int, int]] = []
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for nr, nc in _neighbors4(cr, cc):
                        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == color and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
                comps.append(comp)
        out[color] = comps
    return out


def _sprite_index(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    colors = sorted(set(map(int, np.unique(grid))) - {BLACK, GREEN, BLOCKER})
    comp_map = _components_by_color(grid, colors)
    sprites: list[dict[str, object]] = []
    pixel_to_sprite: dict[tuple[int, int], int] = {}
    for color in colors:
        comps = comp_map[color]
        assert len(comps) == 1, "each sprite color must form exactly one component"
        coords = comps[0]
        assert 3 <= len(coords) <= 6, "sprite area must be 3..6"
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        assert r1 - r0 + 1 <= 3 and c1 - c0 + 1 <= 3, "sprite bbox must fit in 3x3"
        sprite_id = len(sprites)
        sprites.append({"color": color, "coords": coords, "bbox": (r0, c0, r1, c1)})
        for rc in coords:
            pixel_to_sprite[rc] = sprite_id
    assert 3 <= len(sprites) <= 5, "expected 3..5 sprites"
    return sprites, pixel_to_sprite


def _trace_collection_order(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    green_pixels = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)]
    assert len(green_pixels) == 2, "expected a 2-pixel start bar"
    (r0, c0), (r1, c1) = green_pixels
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "start bar must be 4-connected"

    def _is_border(rc: tuple[int, int]) -> bool:
        r, c = rc
        h, w = g.shape
        return r == 0 or c == 0 or r == h - 1 or c == w - 1

    border = [rc for rc in green_pixels if _is_border(rc)]
    assert len(border) == 1, "exactly one start pixel must lie on the border"
    border_pixel = border[0]
    inner_pixel = green_pixels[0] if green_pixels[1] == border_pixel else green_pixels[1]
    dr = inner_pixel[0] - border_pixel[0]
    dc = inner_pixel[1] - border_pixel[1]
    assert abs(dr) + abs(dc) == 1

    sprites, pixel_to_sprite = _sprite_index(g)
    assert sprites

    r, c = inner_pixel
    seen_states: set[tuple[int, int, int, int]] = set()
    painted: set[tuple[int, int]] = set()
    collected: list[int] = []
    recorded: set[int] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "path must not cycle"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            break

        nxt = int(g[nr, nc])
        if nxt in (BLACK, GREEN):
            r, c = nr, nc
            if (r, c) not in painted:
                painted.add((r, c))
                adjacent = {
                    pixel_to_sprite[(ar, ac)]
                    for ar, ac in _neighbors4(r, c)
                    if (ar, ac) in pixel_to_sprite and pixel_to_sprite[(ar, ac)] not in recorded
                }
                assert len(adjacent) <= 1, "a path cell must not discover multiple new sprites at once"
                if adjacent:
                    sprite_id = adjacent.pop()
                    recorded.add(sprite_id)
                    collected.append(sprite_id)
            continue

        assert nxt == BLOCKER, "path may only collide with blocker pixels"
        choices = []
        for tdr, tdc in ((-dc, dr), (dc, -dr)):
            tr, tc = r + tdr, c + tdc
            if _in_bounds(g.shape, tr, tc) and int(g[tr, tc]) in (BLACK, GREEN):
                choices.append((tdr, tdc))
        assert len(choices) == 1, "each blocker collision must force a unique 90-degree turn"
        dr, dc = choices[0]

    assert len(collected) == len(sprites), "the walk must collect every sprite exactly once"
    return collected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk the blocker-defined corridor, collect each adjacent sprite on first encounter, and concatenate the cropped sprites in collection order.

    Input structure:
    - Black background contains a teal blocker layout that forces a single one-cell-wide walk with right-angle turns.
    - A green 2-pixel start bar touches the grid border and fixes the initial direction into the corridor.
    - Three to five distinct single-color sprites sit in small alcoves beside the corridor; each sprite is 4-connected and fits in a 3x3 bounding box.

    Rule:
    - Start from the border pixel of the green bar, pass through the inner green pixel, and keep moving straight across black cells.
    - Whenever the next cell is teal, turn to the unique perpendicular black/green cell and continue; stop when the next step would leave the grid.
    - On the first visit to any path cell that is 4-adjacent to an uncollected sprite, record that sprite.
    - Crop every recorded sprite to its tight bounding box and horizontally concatenate the crops in recording order, top-aligned with black padding below shorter crops.

    Key insight:
    - The corridor geometry is the ordering code: tracing the forced path determines when each alcove sprite is first seen, and that discovery order becomes the atlas order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites, _ = _sprite_index(g)
    order = _trace_collection_order(g)
    crops = [_crop_component(sprites[sprite_id]["coords"], int(sprites[sprite_id]["color"])) for sprite_id in order]
    return _concat_top_aligned(crops)


SPRITE_TEMPLATES = [
    {(1, 0), (1, 1), (1, 2)},
    {(1, 0), (1, 1), (0, 1)},
    {(1, 0), (1, 1), (2, 1), (1, 2)},
    {(1, 0), (0, 1), (1, 1), (2, 1)},
    {(1, 0), (1, 1), (0, 1), (0, 2)},
    {(1, 0), (1, 1), (2, 1), (1, 2), (2, 2)},
    {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)},
    {(1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2)},
]
CHAMBER_WALLS = {(0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)}
CHAMBER_CELLS = {(r, c) for r in range(3) for c in range(3)}

MIN_SPRITES = 3
MAX_SPRITES = 5
VERTICAL_GROUP_CHOICES = (2, 3)  # number of horizontal traversals
VERTICAL_LEN_RANGE = (3, 7)
WIDTH_RANGE = (18, 26)
MAX_RETRIES = 500


def _rotated_crop(crop: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(crop, k).copy()


def _sample_vertical_lengths(rng: np.random.Generator, groups: int) -> list[int] | None:
    for _ in range(200):
        lengths = [int(rng.integers(*VERTICAL_LEN_RANGE)) for _ in range(groups + 1)]
        height = sum(lengths) + 2
        if 18 <= height <= 25:
            return lengths
    return None


def _sample_columns(rng: np.random.Generator, width: int, groups: int) -> list[int] | None:
    valid_cols = list(range(3, width - 3))
    if not valid_cols:
        return None
    for _ in range(300):
        cols = [int(rng.choice(valid_cols))]
        ok = True
        for _ in range(groups):
            options = [c for c in valid_cols if abs(c - cols[-1]) >= 4]
            if not options:
                ok = False
                break
            cols.append(int(rng.choice(options)))
        if not ok:
            continue
        diffs = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
        if any(d > 0 for d in diffs) and any(d < 0 for d in diffs):
            return cols
    return None


def _build_base_path(vertical_lengths: list[int], cols: list[int]) -> tuple[list[tuple[int, int]], list[dict[str, object]], int, int]:
    height = sum(vertical_lengths) + 2
    width = max(cols) + 4

    path = [(0, cols[0]), (1, cols[0])]
    segments: list[dict[str, object]] = []
    current = path[-1]

    for i, vlen in enumerate(vertical_lengths):
        seg_cells = [current]
        for _ in range(vlen):
            current = (current[0] + 1, current[1])
            path.append(current)
            seg_cells.append(current)
        segments.append({"orientation": "v", "cells": seg_cells})
        if i == len(vertical_lengths) - 1:
            break

        target_col = cols[i + 1]
        step = 1 if target_col > current[1] else -1
        seg_cells = [current]
        for _ in range(abs(target_col - current[1])):
            current = (current[0], current[1] + step)
            path.append(current)
            seg_cells.append(current)
        segments.append({"orientation": "h", "cells": seg_cells})

    assert current[0] == height - 1
    return path, segments, height, width


def _turn_blockers_from_path(path: list[tuple[int, int]]) -> set[tuple[int, int]]:
    path_set = set(path)
    blockers: set[tuple[int, int]] = set()
    for i in range(1, len(path) - 1):
        pr, pc = path[i - 1]
        cr, cc = path[i]
        nr, nc = path[i + 1]
        d0 = (cr - pr, cc - pc)
        d1 = (nr - cr, nc - cc)
        if d0 == d1:
            continue
        ahead = (cr + d0[0], cc + d0[1])
        wrong = (cr - d1[0], cc - d1[1])
        assert ahead not in path_set and wrong not in path_set
        blockers.add(ahead)
        blockers.add(wrong)
    return blockers


def _segment_anchor_candidates(segment: dict[str, object]) -> list[tuple[int, int]]:
    cells = list(segment["cells"])
    if len(cells) >= 6:
        core = cells[2:-2]
    else:
        core = cells[1:-1]
    return list(core)


def _choose_anchor_plan(
    rng: np.random.Generator,
    segments: list[dict[str, object]],
    sprite_count: int,
) -> list[tuple[int, tuple[int, int]]]:
    candidates = {i: _segment_anchor_candidates(seg) for i, seg in enumerate(segments)}
    eligible = [i for i, pts in candidates.items() if pts]
    if len(eligible) < sprite_count:
        raise ValueError("not enough eligible segments")

    plan: list[tuple[int, tuple[int, int]]] = []
    used: set[int] = set()
    duplicate_seg = None
    duplicate_count = 0
    dup_options = [i for i in eligible if segments[i]["orientation"] == "h" and len(candidates[i]) >= 2]
    if sprite_count >= 4 and dup_options and rng.random() < 0.8:
        duplicate_seg = int(rng.choice(dup_options))
        pts = candidates[duplicate_seg]
        plan.append((duplicate_seg, pts[0]))
        plan.append((duplicate_seg, pts[-1]))
        used.add(duplicate_seg)
        duplicate_count = 2

    need = sprite_count - duplicate_count
    remaining = [i for i in eligible if i not in used]
    rng.shuffle(remaining)
    chosen = remaining[:need]
    for seg_idx in chosen:
        pts = candidates[seg_idx]
        plan.append((seg_idx, pts[len(pts) // 2]))

    by_orientation = defaultdict(int)
    for seg_idx, _anchor in plan:
        by_orientation[str(segments[seg_idx]["orientation"])] += 1
    if not (by_orientation["h"] and by_orientation["v"]):
        raise ValueError("need both horizontal and vertical discovery contexts")

    def plan_key(item: tuple[int, tuple[int, int]]) -> tuple[int, int, int]:
        seg_idx, anchor = item
        seg_cells = list(segments[seg_idx]["cells"])
        return seg_idx, seg_cells.index(anchor), anchor[0] * 100 + anchor[1]

    return sorted(plan, key=plan_key)


def _transform_local(anchor: tuple[int, int], side: str, locals_: set[tuple[int, int]]) -> set[tuple[int, int]]:
    ar, ac = anchor
    out = set()
    for lr, lc in locals_:
        if side == "right":
            out.add((ar + lr - 1, ac + lc + 1))
        elif side == "left":
            out.add((ar + lr - 1, ac - lc - 1))
        elif side == "down":
            out.add((ar + lc + 1, ac + lr - 1))
        else:
            assert side == "up"
            out.add((ar - lc - 1, ac + lr - 1))
    return out


def _side_options(orientation: str) -> list[str]:
    return ["left", "right"] if orientation == "v" else ["up", "down"]


def _place_sprite_in_chamber(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    path_set: set[tuple[int, int]],
    anchor: tuple[int, int],
    orientation: str,
    color: int,
    template: set[tuple[int, int]],
    side: str,
) -> dict[str, object] | None:
    chamber_cells = _transform_local(anchor, side, CHAMBER_CELLS)
    sprite_cells = _transform_local(anchor, side, template)
    wall_cells = _transform_local(anchor, side, CHAMBER_WALLS) - sprite_cells

    for cell in chamber_cells:
        if not _in_bounds(grid.shape, *cell):
            return None
        if cell in reserved or cell in path_set:
            return None

    if not any(abs(anchor[0] - r) + abs(anchor[1] - c) == 1 for r, c in sprite_cells):
        return None

    for r, c in wall_cells:
        grid[r, c] = BLOCKER
    for r, c in sprite_cells:
        grid[r, c] = color

    reserved.update(chamber_cells)
    crop = _crop_component(sorted(sprite_cells), color)
    return {"color": color, "cells": sorted(sprite_cells), "crop": crop}


def _build_expected_output(crops: list[np.ndarray]) -> np.ndarray:
    return _concat_top_aligned(crops)


def _rotate_grid_and_crops(grid: np.ndarray, crops: list[np.ndarray], k: int) -> tuple[np.ndarray, list[np.ndarray]]:
    return np.rot90(grid, k).copy(), [_rotated_crop(crop, k) for crop in crops]


def _collection_colors_from_trace(grid: np.ndarray) -> list[int]:
    sprites, _ = _sprite_index(grid)
    order = _trace_collection_order(grid)
    return [int(sprites[idx]["color"]) for idx in order]


def _validate_generated(grid: np.ndarray, expected: np.ndarray, expected_colors: list[int]) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    try:
        traced_colors = _collection_colors_from_trace(grid)
        sprites, _ = _sprite_index(grid)
    except Exception:
        return False

    if traced_colors != expected_colors:
        return False

    bboxes = {int(s["color"]): tuple(map(int, s["bbox"])) for s in sprites}
    row_order = [color for color, _bbox in sorted(bboxes.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))]
    col_order = [color for color, _bbox in sorted(bboxes.items(), key=lambda kv: (kv[1][1], kv[1][0], kv[0]))]
    if traced_colors == row_order or traced_colors == col_order:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    sprite_count = int(rng.integers(MIN_SPRITES, MAX_SPRITES + 1))
    groups = int(rng.choice(VERTICAL_GROUP_CHOICES))
    vertical_lengths = _sample_vertical_lengths(rng, groups)
    if vertical_lengths is None:
        return None

    height = sum(vertical_lengths) + 2
    width = int(rng.integers(max(WIDTH_RANGE[0], 18), WIDTH_RANGE[1]))
    cols = _sample_columns(rng, width, groups)
    if cols is None:
        return None

    path, segments, _height, width_needed = _build_base_path(vertical_lengths, cols)
    height = _height
    width = max(width, width_needed)

    if width > 25 or height > 25:
        return None

    grid = np.zeros((height, width), dtype=int)
    path_set = set(path)

    grid[path[0]] = GREEN
    grid[path[1]] = GREEN

    blockers = _turn_blockers_from_path(path)
    for r, c in blockers:
        if not _in_bounds(grid.shape, r, c):
            return None
        grid[r, c] = BLOCKER

    try:
        anchor_plan = _choose_anchor_plan(rng, segments, sprite_count)
    except ValueError:
        return None

    colors = list(SPRITE_PALETTE)
    rng.shuffle(colors)
    colors = colors[:sprite_count]

    reserved = set(blockers) | {path[0], path[1]}
    crops: list[np.ndarray] = []
    placed_infos: list[dict[str, object]] = []

    for (seg_idx, anchor), color in zip(anchor_plan, colors, strict=False):
        orientation = str(segments[seg_idx]["orientation"])
        sides = _side_options(orientation)
        random.shuffle(sides)
        templates = list(SPRITE_TEMPLATES)
        random.shuffle(templates)
        placed = None
        for side in sides:
            for template in templates:
                placed = _place_sprite_in_chamber(
                    grid,
                    reserved,
                    path_set,
                    anchor,
                    orientation,
                    color,
                    template,
                    side,
                )
                if placed is not None:
                    break
            if placed is not None:
                break
        if placed is None:
            return None
        crops.append(placed["crop"])
        placed_infos.append(placed)

    # Reserve raw corridor cells so later validations cannot be confused by chamber overlap.
    for r, c in path[2:]:
        assert grid[r, c] == BLACK

    rotation_k = int(rng.integers(0, 4))
    rotated_grid, rotated_crops = _rotate_grid_and_crops(grid, crops, rotation_k)
    expected = _build_expected_output(rotated_crops)
    expected_colors = [int(info["color"]) for info in placed_infos]

    if not _validate_generated(rotated_grid, expected, expected_colors):
        return None
    return rotated_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-sprite ARC input")
