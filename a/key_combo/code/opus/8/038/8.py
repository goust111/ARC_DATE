from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
RED = 2
MAROON = 9
SPRITE_PALETTE = (1, 3, 4, 6, 7, 8)
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return {(int(r - r0), int(c - c0)) for r, c in coords}


def _is_corner_bracket(coords: np.ndarray) -> bool:
    cells = _normalize(coords)
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2 or len(cells) != h + w - 1:
        return False
    candidates = [
        {(0, c) for c in range(w)} | {(r, 0) for r in range(h)},
        {(0, c) for c in range(w)} | {(r, w - 1) for r in range(h)},
        {(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)},
        {(h - 1, c) for c in range(w)} | {(r, w - 1) for r in range(h)},
    ]
    return any(cells == cand for cand in candidates)


def _find_bracket_and_marker(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int, int, int], tuple[int, int], int]:
    red_components = _component_coords(grid == RED)
    brackets = [coords for coords in red_components if _is_corner_bracket(coords)]
    assert len(brackets) == 1, "expected exactly one red corner bracket"
    bracket = brackets[0]
    bracket_set = {tuple(map(int, rc)) for rc in bracket}
    r0, c0, r1, c1 = _bbox(bracket)
    markers = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in bracket_set and int(grid[r, c]) != BG
    ]
    assert len(markers) == 1, "expected exactly one marker inside bracket bbox"
    marker = markers[0]
    marker_color = int(grid[marker])
    assert marker_color not in (BG, RED)
    return bracket_set, (r0, c0, r1, c1), marker, marker_color


def _find_sprite(grid: np.ndarray, ignored: set[tuple[int, int]]) -> np.ndarray:
    mask = grid != BG
    for r, c in ignored:
        mask[r, c] = False
    components = _component_coords(mask)
    candidates = []
    for coords in components:
        colors = {int(grid[r, c]) for r, c in coords}
        if len(coords) >= 8 and len(colors) >= 2:
            candidates.append(coords)
    assert len(candidates) == 1, "expected exactly one multicolor sprite"
    return candidates[0]


def _mirror_copy(
    out: np.ndarray,
    source: np.ndarray,
    sprite_box: tuple[int, int, int, int],
    selected_color: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = sprite_box
    w = c1 - c0 + 1
    mirror_left = c1 + 2
    mirror_right = mirror_left + w - 1
    assert mirror_right < out.shape[1], "mirror copy must fit"
    erase_mask = np.zeros_like(out, dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            color = int(source[r, c])
            if color == BG:
                continue
            mc = mirror_left + (c1 - c)
            out[r, mc] = color
            if color == selected_color:
                erase_mask[r, mc] = True
    return erase_mask, (r0, mirror_left, r1, mirror_right)


def _fill_erased_segments(
    out: np.ndarray,
    erase_mask: np.ndarray,
    mirror_box: tuple[int, int, int, int],
) -> None:
    r0, c0, r1, c1 = mirror_box
    out[erase_mask] = BG
    for c in range(c0, c1 + 1):
        r = r0
        while r <= r1:
            if not erase_mask[r, c]:
                r += 1
                continue
            start = r
            while r + 1 <= r1 and erase_mask[r + 1, c]:
                r += 1
            end = r
            rr = start
            while rr <= end and int(out[rr, c]) == BG:
                out[rr, c] = MAROON
                rr += 1
            r = end + 1


def _transform_with_color(grid: np.ndarray, selected_color: int) -> np.ndarray:
    bracket_cells, _, marker, _ = _find_bracket_and_marker(grid)
    ignored = set(bracket_cells)
    ignored.add(marker)
    sprite = _find_sprite(grid, ignored)
    sprite_box = _bbox(sprite)
    out = np.array(grid, dtype=int, copy=True)
    erase_mask, mirror_box = _mirror_copy(out, grid, sprite_box, selected_color)
    _fill_erased_segments(out, erase_mask, mirror_box)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the left sprite, erase the selected color only in the copy, then refill each erased column segment with maroon.

    Input structure:
    - The background is black, with one irregular multicolor 4-connected sprite on the left.
    - A red corner bracket encloses one non-red marker pixel whose color matches one sprite color.
    - The area to the right of the sprite is empty so a horizontal mirror copy can be placed there without overlap.

    Rule:
    - Find the marker color from inside the red corner bracket.
    - Copy the sprite to the right by horizontal mirroring, with the mirrored bbox starting two columns to the right of the original bbox.
    - In the mirrored copy only, erase every pixel of the marker color, then for each vertical erased segment fill downward from that segment's top with maroon until the segment ends.

    Key insight:
    - The bracket selects which color gets carved out of the mirrored copy, and those carved holes define exactly where the maroon refill appears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _, _, _, marker_color = _find_bracket_and_marker(g)
    return _transform_with_color(g, marker_color)


def _shape_with_full_bbox(h: int, w: int) -> np.ndarray:
    target_area = random.randint(max(h + w + 1, int(np.ceil(0.58 * h * w))), max(h + w + 1, int(np.floor(0.82 * h * w))))
    target_area = min(target_area, h * w - 1)
    cells: set[tuple[int, int]] = set()

    col = random.randrange(w)
    spine = []
    for r in range(h):
        if r > 0:
            step = random.choice([-1, 0, 1])
            col = min(w - 1, max(0, col + step))
        spine.append((r, col))
        cells.add((r, col))

    missing_cols = [c for c in range(w) if all(cc != c for _, cc in cells)]
    for c in missing_cols:
        r = random.randrange(h)
        anchor = min((rr, cc) for rr, cc in cells if rr == r) if any(rr == r for rr, _ in cells) else spine[r]
        start_c = anchor[1]
        for cc in range(min(start_c, c), max(start_c, c) + 1):
            cells.add((r, cc))

    frontier = set()
    for cell in cells:
        for nb in _neighbors4(*cell):
            if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                frontier.add(nb)
    while len(cells) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(*cell):
            if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                frontier.add(nb)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
        raise ValueError("bbox collapsed")
    if mask.all():
        raise ValueError("shape degenerated to rectangle")
    return mask


def _assign_sprite_colors(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    color_count = random.choice([2, 3])
    colors = tuple(random.sample(SPRITE_PALETTE, color_count))
    target_color = random.choice(colors)
    grid = np.zeros(mask.shape, dtype=int)
    assigned: dict[tuple[int, int], int] = {}
    reserved_not_target: set[tuple[int, int]] = set()

    by_col: dict[int, list[int]] = {}
    for r, c in coords:
        by_col.setdefault(c, []).append(r)
    split_candidates = [c for c, rows in by_col.items() if len(rows) >= 3]
    if split_candidates and random.random() < 0.6:
        col = random.choice(split_candidates)
        rows = sorted(by_col[col])
        top = (rows[0], col)
        mid = (rows[len(rows) // 2], col)
        bot = (rows[-1], col)
        assigned[top] = target_color
        assigned[bot] = target_color
        reserved_not_target.add(mid)

    needed_target = 3 - sum(1 for color in assigned.values() if color == target_color)
    target_rows = {r for (r, _), color in assigned.items() if color == target_color}
    random.shuffle(coords)
    for cell in coords:
        if cell in assigned:
            continue
        if needed_target > 0:
            if len(target_rows) < 2 and cell[0] in target_rows:
                continue
            assigned[cell] = target_color
            target_rows.add(cell[0])
            needed_target -= 1
    if len(target_rows) < 2:
        raise ValueError("target color stayed on one row")

    remaining = [cell for cell in coords if cell not in assigned]
    random.shuffle(remaining)
    for color in colors:
        if color == target_color:
            continue
        need = 3
        while need > 0:
            if not remaining:
                raise ValueError("ran out of cells for mandatory colors")
            cell = remaining.pop()
            if cell in reserved_not_target:
                assigned[cell] = color
                reserved_not_target.remove(cell)
            else:
                assigned[cell] = color
            need -= 1

    for cell in list(reserved_not_target):
        if cell in assigned:
            continue
        non_target_choices = [color for color in colors if color != target_color]
        assigned[cell] = random.choice(non_target_choices)
        reserved_not_target.remove(cell)

    for cell in remaining:
        assigned[cell] = random.choice(colors)

    counts = Counter(assigned.values())
    if any(counts[color] < 3 for color in colors):
        raise ValueError("a sprite color became too rare")
    if len({r for (r, _), color in assigned.items() if color == target_color}) < 2:
        raise ValueError("target color must span multiple rows")

    for (r, c), color in assigned.items():
        grid[r, c] = color
    return grid, colors, target_color


def _bracket_pattern() -> np.ndarray:
    pattern = np.zeros((3, 3), dtype=int)
    pattern[0, :] = RED
    pattern[:, 0] = RED
    return pattern


def _orient_pattern(pattern: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(pattern, k=k)


def _interior_cells(pattern: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(pattern == BG)]


def _make_instance() -> np.ndarray:
    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(4, 6)
    grid_h = random.randint(max(11, sprite_h + 5), min(18, sprite_h + 8))
    min_w = max(14, 2 * sprite_w + 7)
    max_w = min(22, 2 * sprite_w + 11)
    grid_w = random.randint(min_w, max_w)

    mask = _shape_with_full_bbox(sprite_h, sprite_w)
    sprite, _, target_color = _assign_sprite_colors(mask)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    max_sprite_top = grid_h - sprite_h - 4
    sprite_top = random.randint(1, max_sprite_top)
    max_sprite_left = min(4, grid_w - 2 * sprite_w - 1)
    sprite_left = random.randint(1, max_sprite_left)
    grid[sprite_top : sprite_top + sprite_h, sprite_left : sprite_left + sprite_w][mask] = sprite[mask]

    pattern = _orient_pattern(_bracket_pattern(), random.randrange(4))
    interior = _interior_cells(pattern)
    marker_r, marker_c = random.choice(interior)
    bracket_row_min = sprite_top + sprite_h + 1
    if bracket_row_min > grid_h - 3:
        raise ValueError("no room for bracket")
    bracket_top = random.randint(bracket_row_min, grid_h - 3)
    bracket_left = random.randint(0, grid_w - 3)
    bracket_slice = grid[bracket_top : bracket_top + 3, bracket_left : bracket_left + 3]
    if np.any((pattern != 0) & (bracket_slice != 0)):
        raise ValueError("bracket overlap")
    bracket_slice[pattern != 0] = RED
    grid[bracket_top + marker_r, bracket_left + marker_c] = target_color
    return grid


def _parse_generated(grid: np.ndarray):
    bracket_cells, bracket_box, marker, marker_color = _find_bracket_and_marker(grid)
    ignored = set(bracket_cells)
    ignored.add(marker)
    sprite = _find_sprite(grid, ignored)
    sprite_box = _bbox(sprite)
    r0, c0, r1, c1 = sprite_box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    sprite_patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = {int(grid[r, c]) for r, c in sprite}
    color_counts = Counter(int(grid[r, c]) for r, c in sprite)
    return {
        "bracket_cells": bracket_cells,
        "bracket_box": bracket_box,
        "marker": marker,
        "marker_color": marker_color,
        "sprite_coords": sprite,
        "sprite_box": sprite_box,
        "sprite_patch": sprite_patch,
        "sprite_colors": colors,
        "color_counts": color_counts,
        "bbox_h": h,
        "bbox_w": w,
    }


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    info = _parse_generated(grid)
    r0, c0, r1, c1 = info["sprite_box"]
    w = c1 - c0 + 1
    mirror_left = c1 + 2
    out = np.array(grid, dtype=int, copy=True)
    erase_mask = np.zeros_like(out, dtype=bool)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            color = int(grid[rr, cc])
            if color == BG:
                continue
            mc = mirror_left + (c1 - cc)
            out[rr, mc] = color
            if color == info["marker_color"]:
                erase_mask[rr, mc] = True
    out[erase_mask] = BG
    for cc in range(mirror_left, mirror_left + w):
        rows = [rr for rr in range(r0, r1 + 1) if erase_mask[rr, cc]]
        if not rows:
            continue
        start = rows[0]
        prev = rows[0]
        for rr in rows[1:] + [None]:
            if rr is not None and rr == prev + 1:
                prev = rr
                continue
            for fill_r in range(start, prev + 1):
                if int(out[fill_r, cc]) != BG:
                    break
                out[fill_r, cc] = MAROON
            if rr is None:
                break
            start = rr
            prev = rr
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        info = _parse_generated(grid)
    except Exception:
        return False

    if not (5 <= info["bbox_h"] <= 8 and 4 <= info["bbox_w"] <= 6):
        return False
    if info["marker_color"] not in info["sprite_colors"]:
        return False
    if MAROON in info["sprite_colors"]:
        return False
    if any(count < 3 for count in info["color_counts"].values()):
        return False
    marker_rows = {
        int(r)
        for r, c in info["sprite_coords"]
        if int(grid[r, c]) == info["marker_color"]
    }
    if len(marker_rows) < 2:
        return False

    r0, c0, r1, c1 = info["sprite_box"]
    if c1 >= grid.shape[1] // 2 + 1:
        return False
    if info["bracket_box"][0] <= r1:
        return False
    mirror_left = c1 + 2
    mirror_right = mirror_left + info["bbox_w"] - 1
    if mirror_right >= grid.shape[1]:
        return False
    if np.any(grid[r0 : r1 + 1, mirror_left : mirror_right + 1] != BG):
        return False

    expected = _independent_expected(grid)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False
    if not np.any(actual == MAROON):
        return False
    if np.array_equal(actual, grid):
        return False

    no_fill = np.array(grid, copy=True)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            color = int(grid[rr, cc])
            if color == BG:
                continue
            no_fill[rr, mirror_left + (c1 - cc)] = color
    if np.array_equal(no_fill, actual):
        return False

    erase_both = np.array(actual, copy=True)
    erase_both[grid == info["marker_color"]] = MAROON
    if np.array_equal(erase_both, actual):
        return False

    alt_outputs = set()
    for color in sorted(info["sprite_colors"]):
        alt_outputs.add(_transform_with_color(grid, color).tobytes())
    if len(alt_outputs) != len(info["sprite_colors"]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid = _make_instance()
        except ValueError:
            continue
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
