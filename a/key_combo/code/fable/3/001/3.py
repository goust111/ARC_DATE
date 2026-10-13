from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BACKGROUND = 0
OUTLINE = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ("up", "down", "left", "right")
VERTICAL_DIRS = ("up", "down")
HORIZONTAL_DIRS = ("left", "right")
GENERATION_ATTEMPTS = 600


@dataclass(frozen=True)
class Cup:
    direction: str
    bbox: tuple[int, int, int, int]
    width: int
    depth: int


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not bool(mask[sr, sc]) or bool(seen[sr, sc]):
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _outline_mask(shape: tuple[int, int], direction: str) -> np.ndarray | None:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    if direction == "up":
        if h < 4 or w < 5 or (w - 2) % 2 != 1:
            return None
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif direction == "down":
        if h < 4 or w < 5 or (w - 2) % 2 != 1:
            return None
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif direction == "left":
        if h < 5 or w < 4 or (h - 2) % 2 != 1:
            return None
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    elif direction == "right":
        if h < 5 or w < 4 or (h - 2) % 2 != 1:
            return None
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        return None
    return mask


def _classify_cup(g: np.ndarray, coords: np.ndarray) -> Cup:
    r0, c0, r1, c1 = _bbox(coords)
    patch = g[r0 : r1 + 1, c0 : c1 + 1] == OUTLINE
    candidates: list[Cup] = []
    for direction in DIRS:
        expected = _outline_mask(patch.shape, direction)
        if expected is None or not np.array_equal(patch, expected):
            continue
        if direction in VERTICAL_DIRS:
            width = patch.shape[1] - 2
            depth = patch.shape[0] - 2
        else:
            width = patch.shape[0] - 2
            depth = patch.shape[1] - 2
        candidates.append(Cup(direction, (r0, c0, r1, c1), int(width), int(depth)))
    assert len(candidates) == 1, f"expected one cup orientation for outline bbox {(r0, c0, r1, c1)}"
    return candidates[0]


def _cup_count_cells(cup: Cup) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = cup.bbox
    cells: list[tuple[int, int]] = []
    if cup.direction == "up":
        for r in range(r0, r1):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif cup.direction == "down":
        for r in range(r0 + 1, r1 + 1):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif cup.direction == "left":
        for r in range(r0 + 1, r1):
            for c in range(c0, c1):
                cells.append((r, c))
    else:
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1 + 1):
                cells.append((r, c))
    return cells


def _cup_capacity_cells(cup: Cup, layers: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = cup.bbox
    layers = int(min(layers, cup.depth))
    cells: list[tuple[int, int]] = []
    if layers <= 0:
        return cells
    if cup.direction == "up":
        for r in range(r1 - layers, r1):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif cup.direction == "down":
        for r in range(r0 + 1, r0 + 1 + layers):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif cup.direction == "left":
        for r in range(r0 + 1, r1):
            for c in range(c1 - layers, c1):
                cells.append((r, c))
    else:
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c0 + 1 + layers):
                cells.append((r, c))
    return cells


def _opening_start_and_step(cup: Cup) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = cup.bbox
    if cup.direction == "up":
        return r0 - 1, (c0 + c1) // 2, -1, 0
    if cup.direction == "down":
        return r1 + 1, (c0 + c1) // 2, 1, 0
    if cup.direction == "left":
        return (r0 + r1) // 2, c0 - 1, 0, -1
    return (r0 + r1) // 2, c1 + 1, 0, 1


def _find_cups(g: np.ndarray) -> list[Cup]:
    comps = _components4(g == OUTLINE)
    assert comps, "expected at least one gray cup outline"
    cups = [_classify_cup(g, comp) for comp in comps]
    cups.sort(key=lambda cup: cup.bbox)
    owned = np.zeros(g.shape, dtype=bool)
    for cup in cups:
        for r, c in _cup_count_cells(cup):
            assert not bool(owned[r, c]), "cup interiors must not overlap"
            owned[r, c] = True
    stray = (g != BACKGROUND) & (g != OUTLINE) & (~owned)
    assert not np.any(stray), "marker pixels must lie inside a cup mouth/interior"
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle each open cup's marker pixels into bottom liquid layers and spill any extra as a dotted ray.

    Input structure:
    - A black background contains several gray U-shaped cup outlines; each outline is a
      three-sided rectangle with an odd-width open mouth facing up, down, left, or right.
    - The open side of a cup defines that cup's upward direction, so gravity points
      toward the opposite closed side.
    - Each cup contains one non-gray marker color scattered in its mouth/interior cells.

    Rule:
    - For each cup, count its marker pixels k, clear those mouth/interior cells, and
      divide k by the cup width w to get a number of liquid layers.
    - Starting at the closed side, paint that many full layers in the marker color,
      stopping after the cup's depth d layers.
    - If the layer count is larger than d, paint the excess from the mouth center
      outward in the opening direction, one colored dot every other cell.
    - The gray cup outlines and black background remain otherwise unchanged.

    Key insight:
    - The marker locations are only counters: the cup mouth fixes each cup's local
      gravity, the count becomes bottom-aligned liquid, and over-capacity count becomes
      an escaping dotted line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = map(int, g.shape)
    assert h <= 30 and w <= 30, "ARC grids must be at most 30 by 30"

    cups = _find_cups(g)
    out = g.copy()
    plans: list[tuple[Cup, int, int, int]] = []

    for cup in cups:
        count_cells = _cup_count_cells(cup)
        vals = [int(g[r, c]) for r, c in count_cells if int(g[r, c]) not in (BACKGROUND, OUTLINE)]
        marker_colors = set(vals)
        assert len(marker_colors) == 1, "each cup must have exactly one marker color"
        color = int(next(iter(marker_colors)))
        assert color not in (BACKGROUND, OUTLINE), "marker color must differ from background and outline"
        k = len(vals)
        assert k > 0 and k % cup.width == 0, "marker count must be a positive multiple of cup width"
        layers = k // cup.width
        assert layers <= cup.depth + 1, "generator-style cups expose at most one overflow layer"
        plans.append((cup, color, layers, max(0, layers - cup.depth)))
        for r, c in count_cells:
            out[r, c] = BACKGROUND

    for cup, color, layers, overflow in plans:
        for r, c in _cup_capacity_cells(cup, layers):
            assert int(out[r, c]) == BACKGROUND, "liquid fill must land in empty cup interior"
            out[r, c] = color
        if overflow:
            r, c, dr, dc = _opening_start_and_step(cup)
            for _ in range(overflow):
                assert 0 <= r < h and 0 <= c < w, "overflow ray must remain in bounds"
                assert int(out[r, c]) == BACKGROUND, "overflow dot must land on background"
                out[r, c] = color
                r += 2 * dr
                c += 2 * dc

    return out


# --------------- generator ---------------


def _render_outline(grid: np.ndarray, cup: Cup) -> None:
    r0, c0, r1, c1 = cup.bbox
    if cup.direction == "up":
        grid[r0 : r1 + 1, c0] = OUTLINE
        grid[r0 : r1 + 1, c1] = OUTLINE
        grid[r1, c0 : c1 + 1] = OUTLINE
    elif cup.direction == "down":
        grid[r0 : r1 + 1, c0] = OUTLINE
        grid[r0 : r1 + 1, c1] = OUTLINE
        grid[r0, c0 : c1 + 1] = OUTLINE
    elif cup.direction == "left":
        grid[r0, c0 : c1 + 1] = OUTLINE
        grid[r1, c0 : c1 + 1] = OUTLINE
        grid[r0 : r1 + 1, c1] = OUTLINE
    else:
        grid[r0, c0 : c1 + 1] = OUTLINE
        grid[r1, c0 : c1 + 1] = OUTLINE
        grid[r0 : r1 + 1, c0] = OUTLINE


def _cup_from_top_left(direction: str, r: int, c: int, width: int, depth: int) -> Cup:
    if direction in VERTICAL_DIRS:
        h = depth + 2
        w = width + 2
    else:
        h = width + 2
        w = depth + 2
    return Cup(direction, (int(r), int(c), int(r + h - 1), int(c + w - 1)), int(width), int(depth))


def _expanded_bbox_cells(cup: Cup, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = cup.bbox
    cells: set[tuple[int, int]] = set()
    for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            cells.add((r, c))
    return cells


def _overflow_cells(cup: Cup, overflow: int, shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    r, c, dr, dc = _opening_start_and_step(cup)
    cells: list[tuple[int, int]] = []
    for _ in range(int(overflow)):
        if not (0 <= r < h and 0 <= c < w):
            return None
        cells.append((r, c))
        r += 2 * dr
        c += 2 * dc
    return cells


def _candidate_cup(
    shape: tuple[int, int], direction: str, width: int, depth: int, overflow: int
) -> Cup | None:
    h, w = map(int, shape)
    if direction in VERTICAL_DIRS:
        outer_h = depth + 2
        outer_w = width + 2
    else:
        outer_h = width + 2
        outer_w = depth + 2

    min_r = 1
    max_r = h - outer_h - 1
    min_c = 1
    max_c = w - outer_w - 1
    if direction == "up" and overflow:
        min_r = max(min_r, 2)
    if direction == "down" and overflow:
        max_r = min(max_r, h - outer_h - 2)
    if direction == "left" and overflow:
        min_c = max(min_c, 2)
    if direction == "right" and overflow:
        max_c = min(max_c, w - outer_w - 2)
    if min_r > max_r or min_c > max_c:
        return None

    r = random.randint(min_r, max_r)
    c = random.randint(min_c, max_c)
    cup = _cup_from_top_left(direction, r, c, width, depth)
    if _overflow_cells(cup, overflow, shape) is None:
        return None
    return cup


def _expected_from_plan(
    shape: tuple[int, int], specs: list[tuple[Cup, int, int]]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for cup, _color, _layers in specs:
        _render_outline(out, cup)
    for cup, color, layers in specs:
        for r, c in _cup_capacity_cells(cup, layers):
            out[r, c] = int(color)
        overflow = max(0, int(layers) - int(cup.depth))
        cells = _overflow_cells(cup, overflow, shape)
        assert cells is not None
        for r, c in cells:
            out[r, c] = int(color)
    return out


def _place_markers(grid: np.ndarray, cup: Cup, color: int, layers: int) -> None:
    cells = _cup_count_cells(cup)
    needed = int(cup.width) * int(layers)
    assert 0 < needed <= len(cells)
    chosen = random.sample(cells, needed)
    for r, c in chosen:
        grid[r, c] = int(color)


def _choose_direction_plan(n: int) -> list[str]:
    dirs = [random.choice(VERTICAL_DIRS), random.choice(HORIZONTAL_DIRS)]
    while len(dirs) < n:
        dirs.append(random.choice(DIRS))
    random.shuffle(dirs)
    return dirs


def _choose_overflow_plan(n: int) -> list[bool]:
    flags = [True, False]
    while len(flags) < n:
        flags.append(random.random() < 0.45)
    random.shuffle(flags)
    return flags


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    shape = (height, width)
    n_cups = random.randint(2, 4)
    directions = _choose_direction_plan(n_cups)
    overflow_flags = _choose_overflow_plan(n_cups)
    colors = random.sample(MARKER_COLORS, n_cups)

    grid = np.zeros(shape, dtype=int)
    reserved: set[tuple[int, int]] = set()
    specs: list[tuple[Cup, int, int]] = []

    for direction, overflow_flag, color in zip(directions, overflow_flags, colors):
        placed = False
        for _ in range(160):
            cup_width = random.choice((3, 5, 7))
            cup_depth = random.randint(2, 5)
            layers = cup_depth + 1 if overflow_flag else random.randint(1, cup_depth)
            cup = _candidate_cup(shape, direction, cup_width, cup_depth, int(overflow_flag))
            if cup is None:
                continue
            reserve_cells = _expanded_bbox_cells(cup, shape)
            overflow_cells = _overflow_cells(cup, max(0, layers - cup.depth), shape)
            if overflow_cells is None:
                continue
            reserve_cells.update(overflow_cells)
            if reserve_cells & reserved:
                continue
            reserved.update(reserve_cells)
            _render_outline(grid, cup)
            _place_markers(grid, cup, color, layers)
            specs.append((cup, color, layers))
            placed = True
            break
        if not placed:
            return None

    if not any(layers > cup.depth for cup, _color, layers in specs):
        return None
    if not any(layers <= cup.depth for cup, _color, layers in specs):
        return None
    if not ({cup.direction for cup, _color, _layers in specs} & set(VERTICAL_DIRS)):
        return None
    if not ({cup.direction for cup, _color, _layers in specs} & set(HORIZONTAL_DIRS)):
        return None

    expected = _expected_from_plan(shape, specs)
    if np.array_equal(grid, expected):
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup gravity input")
