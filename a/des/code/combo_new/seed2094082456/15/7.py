from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
POINTER = 6
SOURCE_COLORS = (1, 2, 3, 4, 7, 8, 9)

ROOM_COUNT_RANGE = (2, 4)
INNER_H_RANGE = (3, 6)
INNER_W_RANGE = (3, 7)
MARGIN = 2
AXIS_GAP = 2
SIDE_PAD_RANGE = (2, 6)
EXTRA_SLACK_MAX = 6
TWIG_COUNT_RANGE = (0, 3)
TWIG_LENGTH_RANGE = (1, 3)
GEN_ATTEMPTS = 400

SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _enclosed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    passable = (g == BG) | (g == POINTER)
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def _read_source_color(grid: np.ndarray, pr: int, pc: int) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[int] = []
    for dr, dc in SIDE_TO_DIR.values():
        nr, nc = pr + dr, pc + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if int(g[nr, nc]) != WALL:
            continue
        sr, sc = nr, nc
        while 0 <= sr < h and 0 <= sc < w and int(g[sr, sc]) == WALL:
            sr += dr
            sc += dc
        if not (0 <= sr < h and 0 <= sc < w):
            continue
        value = int(g[sr, sc])
        if value not in (BG, WALL, POINTER):
            candidates.append(value)
    assert len(candidates) == 1, "each pointer must reference exactly one source color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room with the color source referenced through its wall by the magenta pointer.

    Input structure:
    - The background is black.
    - Grey wall pixels form one connected irregular wall structure that encloses several black regions.
    - Each enclosed region contains exactly one magenta pointer pixel.
    - Outside the rooms, colored source pixels sit immediately across some wall segments.

    Rule:
    - Find every black-or-magenta connected region that does not touch the grid boundary.
    - In each such region, locate the unique magenta pointer and identify the unique cardinal direction where it touches grey wall.
    - Follow that direction through the contiguous grey wall until the first non-grey cell; use that outside source color to repaint the entire enclosed region, including the pointer cell.

    Key insight:
    - The pointer does not refer to anything inside its room; its single wall contact defines a straight through-the-wall lookup to an external color source.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for region in _enclosed_regions(g):
        pointers = [(r, c) for r, c in region if int(g[r, c]) == POINTER]
        assert len(pointers) == 1, "each enclosed region must contain exactly one pointer"
        color = _read_source_color(g, *pointers[0])
        for r, c in region:
            out[r, c] = color
    return out


def _random_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.randint(0, total) for _ in range(parts - 1))
    values: list[int] = []
    prev = 0
    for cut in cuts:
        values.append(cut - prev)
        prev = cut
    values.append(total - prev)
    return values


def _balanced_labels(a: str, b: str, n: int, rng: random.Random) -> list[str]:
    base = n // 2
    extra_first = rng.choice([False, True]) if n % 2 else False
    count_a = base + (1 if extra_first else 0)
    count_b = base + (1 if n % 2 and not extra_first else 0)
    labels = [a] * count_a + [b] * count_b
    rng.shuffle(labels)
    return labels


def _room_specs(orientation: str, n_rooms: int, rng: random.Random) -> list[dict[str, int | str]]:
    if orientation == "vertical":
        side_labels = _balanced_labels("left", "right", n_rooms, rng)
    else:
        side_labels = _balanced_labels("top", "bottom", n_rooms, rng)
    colors = rng.sample(list(SOURCE_COLORS), n_rooms)
    specs: list[dict[str, int | str]] = []
    for side, color in zip(side_labels, colors):
        inner_h = rng.randint(*INNER_H_RANGE)
        inner_w = rng.randint(*INNER_W_RANGE)
        if orientation == "vertical":
            allowed = ["top", "bottom", "left"] if side == "left" else ["top", "bottom", "right"]
        else:
            allowed = ["top", "left", "right"] if side == "top" else ["bottom", "left", "right"]
        specs.append(
            {
                "side": side,
                "pointer_side": rng.choice(allowed),
                "color": color,
                "h": inner_h + 2,
                "w": inner_w + 2,
            }
        )
    rng.shuffle(specs)
    return specs


def _layout_vertical(
    specs: list[dict[str, int | str]], rng: random.Random
) -> tuple[int, int, int, list[dict[str, int | str]]] | None:
    heights = [int(spec["h"]) for spec in specs]
    min_h = 2 * MARGIN + sum(heights) + AXIS_GAP * (len(specs) - 1)
    if min_h > 30:
        return None
    h = rng.randint(min_h, min(30, min_h + EXTRA_SLACK_MAX))

    left_width = max((int(spec["w"]) for spec in specs if spec["side"] == "left"), default=0)
    right_width = max((int(spec["w"]) for spec in specs if spec["side"] == "right"), default=0)
    left_area = left_width + rng.randint(*SIDE_PAD_RANGE)
    right_area = right_width + rng.randint(*SIDE_PAD_RANGE)
    min_w = MARGIN + left_area + 1 + right_area + MARGIN
    if min_w > 30:
        return None
    extra_w = rng.randint(0, min(EXTRA_SLACK_MAX, 30 - min_w))
    extra_left = rng.randint(0, extra_w)
    left_area += extra_left
    right_area += extra_w - extra_left
    w = MARGIN + left_area + 1 + right_area + MARGIN
    trunk_col = MARGIN + left_area

    slack = h - 2 * MARGIN - sum(heights) - AXIS_GAP * (len(specs) - 1)
    extras = _random_partition(slack, len(specs) + 1, rng)
    cur = MARGIN + extras[0]
    placed: list[dict[str, int | str]] = []
    for idx, spec in enumerate(specs):
        sh = int(spec["h"])
        sw = int(spec["w"])
        r0 = cur
        cur += sh
        if idx < len(specs) - 1:
            cur += AXIS_GAP + extras[idx + 1]
        if spec["side"] == "left":
            cmin = MARGIN
            cmax = trunk_col - sw - 1
        else:
            cmin = trunk_col + 2
            cmax = w - MARGIN - sw
        if cmin > cmax:
            return None
        placed_spec = dict(spec)
        placed_spec["r0"] = r0
        placed_spec["c0"] = rng.randint(cmin, cmax)
        placed.append(placed_spec)
    return h, w, trunk_col, placed


def _layout_horizontal(
    specs: list[dict[str, int | str]], rng: random.Random
) -> tuple[int, int, int, list[dict[str, int | str]]] | None:
    widths = [int(spec["w"]) for spec in specs]
    min_w = 2 * MARGIN + sum(widths) + AXIS_GAP * (len(specs) - 1)
    if min_w > 30:
        return None
    w = rng.randint(min_w, min(30, min_w + EXTRA_SLACK_MAX))

    top_height = max((int(spec["h"]) for spec in specs if spec["side"] == "top"), default=0)
    bottom_height = max((int(spec["h"]) for spec in specs if spec["side"] == "bottom"), default=0)
    top_area = top_height + rng.randint(*SIDE_PAD_RANGE)
    bottom_area = bottom_height + rng.randint(*SIDE_PAD_RANGE)
    min_h = MARGIN + top_area + 1 + bottom_area + MARGIN
    if min_h > 30:
        return None
    extra_h = rng.randint(0, min(EXTRA_SLACK_MAX, 30 - min_h))
    extra_top = rng.randint(0, extra_h)
    top_area += extra_top
    bottom_area += extra_h - extra_top
    h = MARGIN + top_area + 1 + bottom_area + MARGIN
    trunk_row = MARGIN + top_area

    slack = w - 2 * MARGIN - sum(widths) - AXIS_GAP * (len(specs) - 1)
    extras = _random_partition(slack, len(specs) + 1, rng)
    cur = MARGIN + extras[0]
    placed: list[dict[str, int | str]] = []
    for idx, spec in enumerate(specs):
        sh = int(spec["h"])
        sw = int(spec["w"])
        c0 = cur
        cur += sw
        if idx < len(specs) - 1:
            cur += AXIS_GAP + extras[idx + 1]
        if spec["side"] == "top":
            rmin = MARGIN
            rmax = trunk_row - sh - 1
        else:
            rmin = trunk_row + 2
            rmax = h - MARGIN - sh
        if rmin > rmax:
            return None
        placed_spec = dict(spec)
        placed_spec["r0"] = rng.randint(rmin, rmax)
        placed_spec["c0"] = c0
        placed.append(placed_spec)
    return h, w, trunk_row, placed


def _paint_rect_room(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, value: int) -> None:
    canvas[r0, c0 : c0 + w] = value
    canvas[r0 + h - 1, c0 : c0 + w] = value
    canvas[r0 : r0 + h, c0] = value
    canvas[r0 : r0 + h, c0 + w - 1] = value


def _pointer_position(spec: dict[str, int | str], rng: random.Random) -> tuple[int, int]:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    h = int(spec["h"])
    w = int(spec["w"])
    side = str(spec["pointer_side"])
    if side == "top":
        return r0 + 1, rng.randint(c0 + 2, c0 + w - 3)
    if side == "bottom":
        return r0 + h - 2, rng.randint(c0 + 2, c0 + w - 3)
    if side == "left":
        return rng.randint(r0 + 2, r0 + h - 3), c0 + 1
    return rng.randint(r0 + 2, r0 + h - 3), c0 + w - 2


def _branch_cells(spec: dict[str, int | str], trunk_index: int, orientation: str, rng: random.Random) -> list[tuple[int, int]]:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    h = int(spec["h"])
    w = int(spec["w"])
    side = str(spec["side"])
    cells: list[tuple[int, int]] = []
    if orientation == "vertical":
        conn_r = rng.randint(r0 + 1, r0 + h - 2)
        if side == "left":
            for c in range(c0 + w, trunk_index + 1):
                cells.append((conn_r, c))
        else:
            for c in range(trunk_index, c0):
                cells.append((conn_r, c))
    else:
        conn_c = rng.randint(c0 + 1, c0 + w - 2)
        if side == "top":
            for r in range(r0 + h, trunk_index + 1):
                cells.append((r, conn_c))
        else:
            for r in range(trunk_index, r0):
                cells.append((r, conn_c))
    return cells


def _can_add_twig(grid: np.ndarray, cells: list[tuple[int, int]], anchor: tuple[int, int]) -> bool:
    h, w = grid.shape
    allowed = {anchor, *cells}
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BG:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG and (nr, nc) not in allowed:
                return False
    return True


def _add_twigs(
    grid: np.ndarray, expected: np.ndarray, orientation: str, trunk_index: int, rng: random.Random
) -> None:
    h, w = grid.shape
    twig_target = rng.randint(*TWIG_COUNT_RANGE)
    if twig_target == 0:
        return
    attempts = 0
    added = 0
    while added < twig_target and attempts < 60:
        attempts += 1
        length = rng.randint(*TWIG_LENGTH_RANGE)
        if orientation == "vertical":
            row = rng.randint(2, h - 3)
            direction = rng.choice([-1, 1])
            anchor = (row, trunk_index)
            cells = [(row, trunk_index + direction * step) for step in range(1, length + 1)]
        else:
            col = rng.randint(2, w - 3)
            direction = rng.choice([-1, 1])
            anchor = (trunk_index, col)
            cells = [(trunk_index + direction * step, col) for step in range(1, length + 1)]
        if not _can_add_twig(grid, cells, anchor):
            continue
        for r, c in cells:
            grid[r, c] = WALL
            expected[r, c] = WALL
        added += 1


def _validate_structure(grid: np.ndarray, n_rooms: int) -> bool:
    regions = _enclosed_regions(grid)
    if len(regions) != n_rooms:
        return False
    for region in regions:
        pointers = [(r, c) for r, c in region if int(grid[r, c]) == POINTER]
        if len(pointers) != 1:
            return False
        try:
            _read_source_color(grid, *pointers[0])
        except AssertionError:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["vertical", "horizontal"])
    n_rooms = rng.randint(*ROOM_COUNT_RANGE)
    specs = _room_specs(orientation, n_rooms, rng)
    if orientation == "vertical":
        layout = _layout_vertical(specs, rng)
    else:
        layout = _layout_horizontal(specs, rng)
    if layout is None:
        return None

    if orientation == "vertical":
        h, w, trunk_index, placed = layout
    else:
        h, w, trunk_index, placed = layout

    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    if orientation == "vertical":
        grid[1 : h - 1, trunk_index] = WALL
        expected[1 : h - 1, trunk_index] = WALL
    else:
        grid[trunk_index, 1 : w - 1] = WALL
        expected[trunk_index, 1 : w - 1] = WALL

    for spec in placed:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        sh = int(spec["h"])
        sw = int(spec["w"])
        color = int(spec["color"])

        _paint_rect_room(grid, r0, c0, sh, sw, WALL)
        _paint_rect_room(expected, r0, c0, sh, sw, WALL)

        expected[r0 + 1 : r0 + sh - 1, c0 + 1 : c0 + sw - 1] = color

        pr, pc = _pointer_position(spec, rng)
        dr, dc = SIDE_TO_DIR[str(spec["pointer_side"])]
        sr, sc = pr + 2 * dr, pc + 2 * dc
        if not (0 <= sr < h and 0 <= sc < w):
            return None
        if int(grid[sr, sc]) != BG:
            return None
        grid[pr, pc] = POINTER
        grid[sr, sc] = color
        expected[sr, sc] = color

        for br, bc in _branch_cells(spec, trunk_index, orientation, rng):
            if int(grid[br, bc]) not in (BG, WALL):
                return None
            grid[br, bc] = WALL
            expected[br, bc] = WALL

    _add_twigs(grid, expected, orientation, trunk_index, rng)

    if not _validate_structure(grid, n_rooms):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
