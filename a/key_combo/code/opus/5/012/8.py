from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if int(connectivity) == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = DIRS8

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _shape_patch_from_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    return patch


def _random_component_patch(size: int) -> np.ndarray:
    size = int(size)
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if abs(nr) <= 2 and abs(nc) <= 2 and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))
    patch = _shape_patch_from_coords(cells)
    if random.choice([False, True]):
        patch = np.flipud(patch)
    if random.choice([False, True]):
        patch = np.fliplr(patch)
    if random.choice([False, True]):
        patch = patch.T.copy()
    return patch.astype(bool)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr, dc in DIRS8:
        rs0 = max(0, -dr)
        rs1 = min(h, h - dr)
        cs0 = max(0, -dc)
        cs1 = min(w, w - dc)
        out[rs0:rs1, cs0:cs1] |= mask[rs0 + dr : rs1 + dr, cs0 + dc : cs1 + dc]
    return out


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    rr, cc = np.nonzero(patch)
    grid[rr + int(r0), cc + int(c0)] = int(color)


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    color: int,
    col_min: int,
    col_max: int,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return False
    col_min = max(0, int(col_min))
    col_max = min(int(col_max), w - pw)
    if col_min > col_max:
        return False

    blocked = _dilate8(grid != BG)
    candidates = [(r0, c0) for r0 in range(0, h - ph + 1) for c0 in range(col_min, col_max + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if not np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & patch):
            _paint_patch(grid, patch, r0, c0, color)
            return True
    return False


def _order_key(coords: np.ndarray, direction: int) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if int(direction) < 0:
        return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
    return (-int(cols.max()), int(rows.min()), -int(cols.min()), int(rows.max()))


def _slide_components(shape: tuple[int, int], components: list[np.ndarray], wall_col: int, direction: int) -> list[np.ndarray]:
    h, w = shape
    settled = np.zeros((h, w), dtype=bool)
    settled[:, int(wall_col)] = True
    out: list[np.ndarray] = []
    for coords in sorted(components, key=lambda comp: _order_key(comp, direction)):
        shift = 0
        while True:
            candidate = coords + np.array([0, shift + int(direction)], dtype=int)
            if int(candidate[:, 1].min()) < 0 or int(candidate[:, 1].max()) >= w:
                break
            if np.any(settled[candidate[:, 0], candidate[:, 1]]):
                break
            shift += int(direction)
        placed = coords + np.array([0, shift], dtype=int)
        assert not np.any(settled[placed[:, 0], placed[:, 1]])
        settled[placed[:, 0], placed[:, 1]] = True
        out.append(placed)
    return out


def _square_side(area: int) -> int:
    side = int(math.isqrt(int(area)))
    assert side * side == int(area), f"expected a perfect square, got {area}"
    return side


def _render_output(items: list[tuple[int, int]]) -> np.ndarray:
    specs = []
    for idx, (color, area) in enumerate(items):
        side = _square_side(area)
        specs.append((side * side, idx, int(color), side))

    specs.sort(key=lambda item: (item[0], item[1]))
    height = max(side for _area, _idx, _color, side in specs)
    width = sum(side for _area, _idx, _color, side in specs) + 2
    out = np.full((height, width), BG, dtype=int)

    c0 = 0
    for _area, _idx, color, side in specs:
        out[height - side : height, c0 : c0 + side] = int(color)
        c0 += side + 1
    return out


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 3

    bg = BG if np.any(g == BG) else _get_bg(g)
    left_color = int(g[0, 0])
    right_color = int(g[0, -1])
    assert np.all(g[:, 0] == left_color)
    assert np.all(g[:, -1] == right_color)
    remaining = [int(c) for c in np.unique(g) if int(c) not in {bg, left_color, right_color}]
    assert len(remaining) == 1, "expected exactly one non-wall foreground color"
    green_color = remaining[0]

    red_mask = g == left_color
    red_mask[:, 0] = False
    blue_mask = g == right_color
    blue_mask[:, -1] = False
    green_mask = g == green_color

    red_components = _components_mask(red_mask, connectivity=8)
    blue_components = _components_mask(blue_mask, connectivity=8)
    green_components = _components_mask(green_mask, connectivity=8)

    assert red_components and blue_components and green_components

    slidden = np.full_like(g, bg)
    slidden[:, 0] = left_color
    slidden[:, -1] = right_color
    for coords in _slide_components(g.shape, red_components, wall_col=0, direction=-1):
        slidden[coords[:, 0], coords[:, 1]] = left_color
    for coords in _slide_components(g.shape, blue_components, wall_col=g.shape[1] - 1, direction=1):
        assert not np.any(slidden[coords[:, 0], coords[:, 1]] != bg)
        slidden[coords[:, 0], coords[:, 1]] = right_color

    red_total = int(np.sum(slidden == left_color))
    blue_total = int(np.sum(slidden == right_color))
    green_total = int(np.sum(g == green_color))
    out = _render_output([(left_color, red_total), (right_color, blue_total), (green_color, green_total)])

    return out, {
        "bg": bg,
        "left_color": left_color,
        "right_color": right_color,
        "green_color": green_color,
        "red_components": red_components,
        "blue_components": blue_components,
        "green_components": green_components,
        "slidden": slidden,
        "red_total": red_total,
        "blue_total": blue_total,
        "green_total": green_total,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the wall-colored objects into their matching side walls, then convert the resulting color totals into sorted solid squares.

    Input structure:
    - The background is black.
    - The full left column is a red wall and the full right column is a blue wall.
    - Away from the walls there are several 8-connected red objects, several 8-connected blue objects,
      and several green objects; each object has size 2 to 5 and objects are mutually non-adjacent.

    Rule:
    - Slide each red object horizontally left, processing red objects from nearest to farthest from the red wall,
      until it touches the wall column or an already-settled red object.
    - Slide each blue object horizontally right, processing blue objects from nearest to farthest from the blue wall,
      until it touches the wall column or an already-settled blue object.
    - Erase every green object, but keep its original pixel count.
    - Count red pixels after sliding including the red wall, count blue pixels after sliding including the blue wall,
      and count the original green pixels; each count is a perfect square.
    - Output three solid squares in those three colors, with side lengths equal to the square roots of the counts,
      sorted from smaller area to larger area, bottom-aligned, with one black column between neighbors.

    Key insight:
    - The side walls are counted as part of the red and blue totals, so the observer must combine the wall columns
      with the slid same-color objects before turning the counts into the ordered squares.
    """
    out, _details = _solve_details(grid)
    return out


_SCENARIOS = [
    {
        "height": 21,
        "width_range": (30, 30),
        "small_obj": 4,
        "small_side": 5,
        "large_obj": 100,
        "large_side": 11,
        "green_sides": [2, 3, 4, 6, 7, 8, 9],
    },
    {
        "height": 27,
        "width_range": (30, 30),
        "small_obj": 9,
        "small_side": 6,
        "large_obj": 169,
        "large_side": 14,
        "green_sides": [2, 3, 4, 5, 7, 8],
    },
]

_LAYOUT_OPTIONS: list[tuple[int, bool, int]] = [
    (0, red_large, green_side)
    for red_large in (False, True)
    for green_side in [2, 3, 4, 6, 7]
] + [
    (1, True, green_side) for green_side in [2, 3, 4, 5]
]
_UNSEEN_LAYOUTS: list[tuple[int, bool, int]] = list(_LAYOUT_OPTIONS)
_SEEN_OUTPUT_SIGS: set[bytes] = set()
_TARGET_UNIQUE_OUTPUTS = len(_LAYOUT_OPTIONS)


def _partition_total(total: int, *, min_parts: int, prefer_dense: bool) -> list[int]:
    min_k = max(int(min_parts), math.ceil(int(total) / 5))
    max_k = int(total) // 2
    assert min_k <= max_k
    if prefer_dense:
        upper = min(max_k, min_k + 2)
    else:
        upper = min(max_k, min_k + 3)
    k = random.randint(min_k, upper)

    sizes = [2] * k
    remaining = int(total) - 2 * k
    order = list(range(k))
    while remaining > 0:
        random.shuffle(order)
        changed = False
        for idx in order:
            if remaining == 0:
                break
            room = 5 - sizes[idx]
            if room <= 0:
                continue
            add = random.randint(1, min(room, remaining))
            sizes[idx] += add
            remaining -= add
            changed = True
        assert changed, "partition stalled"
    random.shuffle(sizes)
    return sizes


def _choose_layout() -> tuple[int, bool, int]:
    if _UNSEEN_LAYOUTS:
        idx = random.randrange(len(_UNSEEN_LAYOUTS))
        return _UNSEEN_LAYOUTS.pop(idx)
    return random.choice(_LAYOUT_OPTIONS)


def _object_components_away_from_walls(grid: np.ndarray) -> list[np.ndarray]:
    mask = grid != BG
    mask[:, 0] = False
    mask[:, -1] = False
    return _components_mask(mask, connectivity=8)


def _validate_generated_case(inp: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray)
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert np.all(inp[:, 0] == RED)
    assert np.all(inp[:, -1] == BLUE)

    details_out, details = _solve_details(inp)
    assert details_out.ndim == 2 and max(details_out.shape) <= 30

    for coords in details["red_components"] + details["blue_components"] + details["green_components"]:
        size = int(coords.shape[0])
        assert 2 <= size <= 5

    all_objects = _object_components_away_from_walls(inp)
    assert len(all_objects) == len(details["red_components"]) + len(details["blue_components"]) + len(details["green_components"])

    red_obj_total = int(np.sum(inp == RED)) - inp.shape[0]
    blue_obj_total = int(np.sum(inp == BLUE)) - inp.shape[0]
    green_total = details["green_total"]

    for total in (red_obj_total, blue_obj_total, green_total, details["red_total"], details["blue_total"]):
        side = int(math.isqrt(int(total)))
        assert side * side == int(total)

    assert red_obj_total != details["red_total"]
    assert blue_obj_total != details["blue_total"]
    assert len(details["red_components"]) >= 2
    assert len(details["blue_components"]) >= 2

    red_side = _square_side(details["red_total"])
    blue_side = _square_side(details["blue_total"])
    green_side = _square_side(details["green_total"])
    assert len({red_side, blue_side, green_side}) == 3

    slidden = details["slidden"]
    red_slid = _components_mask(slidden == RED, connectivity=8)
    blue_slid = _components_mask(slidden == BLUE, connectivity=8)
    assert len(red_slid) == 1
    assert len(blue_slid) == 1


def _build_case(scenario_idx: int, red_large: bool, green_side: int) -> np.ndarray | None:
    scenario = _SCENARIOS[int(scenario_idx)]
    h = int(scenario["height"])
    w = random.randint(*scenario["width_range"])
    grid = np.full((h, w), BG, dtype=int)
    grid[:, 0] = RED
    grid[:, -1] = BLUE

    if red_large:
        red_total = int(scenario["large_obj"])
        blue_total = int(scenario["small_obj"])
    else:
        red_total = int(scenario["small_obj"])
        blue_total = int(scenario["large_obj"])
    green_total = int(green_side) * int(green_side)

    red_sizes = _partition_total(red_total, min_parts=2, prefer_dense=True)
    blue_sizes = _partition_total(blue_total, min_parts=2, prefer_dense=True)
    green_sizes = _partition_total(
        green_total,
        min_parts=1 if green_total <= 9 else 2,
        prefer_dense=green_total >= 36,
    )

    if red_large:
        red_col_min, red_col_hi = 2, max(2, w - 10)
        blue_col_lo, blue_col_hi = max(2, w - 6), w - 3
    else:
        red_col_min, red_col_hi = 2, min(w - 3, 5)
        blue_col_lo, blue_col_hi = max(2, 9), w - 3

    placements = [
        (RED, red_sizes, red_col_min, red_col_hi),
        (BLUE, blue_sizes, blue_col_lo, blue_col_hi),
        (GREEN, green_sizes, 2, w - 3),
    ]
    placements.sort(key=lambda item: sum(item[1]), reverse=True)

    for color, sizes, col_min, col_max in placements:
        for size in sorted(sizes, reverse=True):
            placed = False
            for _ in range(25):
                patch = _random_component_patch(size)
                if _place_patch(grid, patch, color, col_min, col_max):
                    placed = True
                    break
            if not placed:
                return None

    try:
        _validate_generated_case(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        layout = _choose_layout()
        inp = _build_case(*layout)
        if inp is None:
            if layout not in _UNSEEN_LAYOUTS:
                _UNSEEN_LAYOUTS.append(layout)
            continue

        out = main(inp)
        sig = out.tobytes() + str(out.shape).encode()
        if sig in _SEEN_OUTPUT_SIGS and len(_SEEN_OUTPUT_SIGS) < _TARGET_UNIQUE_OUTPUTS:
            if layout not in _UNSEEN_LAYOUTS:
                _UNSEEN_LAYOUTS.append(layout)
            continue
        _SEEN_OUTPUT_SIGS.add(sig)
        return inp
    raise RuntimeError("failed to generate a valid ARC sample")
