from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import Color, colors_present, get_bg
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])

    def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
        colors = set(map(int, np.asarray(grid).ravel()))
        if exclude is not None:
            colors -= set(map(int, exclude))
        return sorted(colors)


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BACKGROUND = Color.BLACK
SIDE_RANGE = (3, 5)
MAX_GEN_ATTEMPTS = 500
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _swap_two_colors(grid: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    swapped = np.array(grid, dtype=int, copy=True)
    a_mask = swapped == int(color_a)
    b_mask = swapped == int(color_b)
    swapped[a_mask] = int(color_b)
    swapped[b_mask] = int(color_a)
    return swapped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a two-color motif into self-similar stamp blocks, swapping colors only under minority-color parents.

    Input structure:
    - The input is an h x w grid with a dominant background and exactly two foreground colors forming one motif.
    - The two foreground colors occupy different numbers of cells.

    Rule:
    - Produce an (h*h) x (w*w) output partitioned into h x w blocks, one block for each input cell.
    - A background input cell maps to an all-background block.
    - A majority-color input cell maps to an unchanged copy of the full input motif.
    - A minority-color input cell maps to the same motif with the two foreground colors swapped everywhere.

    Key insight:
    - The motif is both the placement map and the stamp, but each parent cell's majority/minority color role decides whether its child copy keeps or inverts the two foreground colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected exactly two foreground colors"

    color0, color1 = map(int, fg_colors)
    count0 = int(np.count_nonzero(g == color0))
    count1 = int(np.count_nonzero(g == color1))
    assert count0 != count1, "foreground colors must have different counts"

    majority = color0 if count0 > count1 else color1
    minority = color1 if majority == color0 else color0
    original_stamp = g
    swapped_stamp = _swap_two_colors(g, majority, minority)

    h, w = map(int, g.shape)
    out = np.full((h * h, w * w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            rs = r * h
            cs = c * w
            cell = int(g[r, c])
            if cell == bg:
                continue
            if cell == majority:
                out[rs : rs + h, cs : cs + w] = original_stamp
            else:
                assert cell == minority
                out[rs : rs + h, cs : cs + w] = swapped_stamp

    return out


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(cell: Coord, shape: tuple[int, int]) -> list[Coord]:
    r, c = cell
    out: list[Coord] = []
    for dr, dc in DIRS4:
        nxt = (r + dr, c + dc)
        if _in_bounds(shape, nxt):
            out.append(nxt)
    return out


def _random_path(start: Coord, end: Coord) -> list[Coord]:
    r, c = start
    er, ec = end
    steps = [(-1 if r > er else 1, 0)] * abs(er - r)
    steps += [(0, -1 if c > ec else 1)] * abs(ec - c)
    random.shuffle(steps)
    path = [(r, c)]
    for dr, dc in steps:
        r += dr
        c += dc
        path.append((r, c))
    return path


def _grow_mask(mask: np.ndarray, target_count: int) -> np.ndarray | None:
    out = mask.copy()
    while int(out.sum()) < int(target_count):
        frontier: set[Coord] = set()
        for r_raw, c_raw in np.argwhere(out):
            for nr, nc in _neighbors4((int(r_raw), int(c_raw)), out.shape):
                if not out[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        out[random.choice(tuple(frontier))] = True
    return out


def _build_full_span_mask(h: int, w: int) -> np.ndarray | None:
    top = (0, random.randrange(w))
    bottom = (h - 1, random.randrange(w))
    left = (random.randrange(h), 0)
    right = (random.randrange(h), w - 1)
    hub = (random.randrange(h), random.randrange(w))

    mask = np.zeros((h, w), dtype=bool)
    for anchor in (top, bottom, left, right):
        for cell in _random_path(hub, anchor):
            mask[cell] = True

    area = h * w
    base_count = int(mask.sum())
    minority_floor = max(1, 2 * base_count - area + 1)
    if minority_floor > base_count // 2:
        return None

    extra = random.randint(0, max(0, min(area - 1, base_count + 4) - base_count))
    grown = _grow_mask(mask, base_count + extra)
    if grown is None:
        return None
    box = _bbox(grown)
    if box != (0, 0, h - 1, w - 1):
        return None
    return grown


def _build_offset_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(40):
        box_h = random.randint(2, h - 1)
        box_w = random.randint(2, w - 1)
        r0 = random.randint(0, h - box_h)
        c0 = random.randint(0, w - box_w)
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1
        if (r0, c0, r1, c1) == (0, 0, h - 1, w - 1):
            continue

        local = np.zeros((box_h, box_w), dtype=bool)
        start = (random.randrange(box_h), random.randrange(box_w))
        local[start] = True
        target = random.randint(3, max(3, min(box_h * box_w - 1, box_h * box_w)))
        grown = _grow_mask(local, target)
        if grown is None:
            continue

        mask = np.zeros((h, w), dtype=bool)
        mask[r0 : r1 + 1, c0 : c1 + 1] = grown
        if _bbox(mask) != (r0, c0, r1, c1):
            continue
        return mask
    return None


def _choose_minority(mask: np.ndarray, count: int) -> set[Coord] | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    mode = random.choice(("cluster", "scatter", "double_cluster"))

    if mode == "scatter":
        return set(random.sample(cells, count))

    chosen: set[Coord] = set()
    seeds = 1 if mode == "cluster" else min(2, count)
    seed_cells = random.sample(cells, seeds)
    for cell in seed_cells:
        chosen.add(cell)

    while len(chosen) < count:
        frontier: set[Coord] = set()
        for cell in chosen:
            for nxt in _neighbors4(cell, mask.shape):
                if mask[nxt] and nxt not in chosen:
                    frontier.add(nxt)
        if not frontier:
            remainder = [cell for cell in cells if cell not in chosen]
            if len(remainder) < count - len(chosen):
                return None
            chosen.update(random.sample(remainder, count - len(chosen)))
            break
        chosen.add(random.choice(tuple(frontier)))

    return chosen


def _paint_grid(mask: np.ndarray) -> np.ndarray | None:
    area = int(mask.size)
    fg_count = int(mask.sum())
    if fg_count < 3 or fg_count >= area:
        return None

    min_minority = max(1, 2 * fg_count - area + 1)
    max_minority = fg_count // 2
    if min_minority > max_minority:
        return None

    minority_count = random.randint(min_minority, max_minority)
    minority_cells = _choose_minority(mask, minority_count)
    if minority_cells is None:
        return None

    colors = random.sample([int(c) for c in Color.NOT_BLACK], 2)
    majority_color, minority_color = colors if random.random() < 0.5 else colors[::-1]

    grid = np.full(mask.shape, BACKGROUND, dtype=int)
    grid[mask] = int(majority_color)
    for cell in minority_cells:
        grid[cell] = int(minority_color)

    bg = int(np.count_nonzero(grid == BACKGROUND))
    majority = int(np.count_nonzero(grid == majority_color))
    minority = int(np.count_nonzero(grid == minority_color))
    if not (bg > majority > minority >= 1):
        return None
    return grid


def _all_foreground_original_output(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = map(int, grid.shape)
    out = np.full((h * h, w * w), bg, dtype=int)
    for r, c in np.argwhere(grid != bg):
        rr = int(r) * h
        cc = int(c) * w
        out[rr : rr + h, cc : cc + w] = grid
    return out


def _validate_semantics(grid: np.ndarray, *, require_padding: bool) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    bg = int(get_bg(grid))
    if bg != BACKGROUND:
        return False

    fg_colors = colors_present(grid, exclude={bg})
    if len(fg_colors) != 2:
        return False
    color0, color1 = map(int, fg_colors)
    count0 = int(np.count_nonzero(grid == color0))
    count1 = int(np.count_nonzero(grid == color1))
    if count0 == count1:
        return False

    majority = color0 if count0 > count1 else color1
    minority = color1 if majority == color0 else color0
    box = _bbox(grid != bg)
    if box is None:
        return False
    if require_padding and box == (0, 0, grid.shape[0] - 1, grid.shape[1] - 1):
        return False
    if not require_padding and box != (0, 0, grid.shape[0] - 1, grid.shape[1] - 1):
        return False

    if not np.any(grid == bg) or not np.any(grid == majority) or not np.any(grid == minority):
        return False

    out = main(grid)
    h, w = map(int, grid.shape)
    if out.shape != (h * h, w * w) or max(out.shape) > 30:
        return False

    swapped = _swap_two_colors(grid, majority, minority)
    blank = np.full_like(grid, bg)
    templates = {
        bg: blank,
        majority: grid,
        minority: swapped,
    }
    if np.array_equal(grid, swapped):
        return False

    decoded = np.full_like(grid, -1)
    for r in range(h):
        for c in range(w):
            block = out[r * h : (r + 1) * h, c * w : (c + 1) * w]
            matches = [value for value, tpl in templates.items() if np.array_equal(block, tpl)]
            if len(matches) != 1:
                return False
            decoded[r, c] = matches[0]

    if not np.array_equal(decoded, grid):
        return False

    if np.array_equal(out, _all_foreground_original_output(grid, bg)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*SIDE_RANGE)
    w = random.randint(*SIDE_RANGE)
    require_padding = random.random() < 0.5

    mask: np.ndarray | None
    if require_padding:
        mask = _build_offset_mask(h, w)
    else:
        if h < 4 or w < 4:
            return None
        mask = _build_full_span_mask(h, w)
    if mask is None:
        return None

    grid = _paint_grid(mask)
    if grid is None:
        return None
    return grid if _validate_semantics(grid, require_padding=require_padding) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid majority-stamp input")
