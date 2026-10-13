from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRS = {
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "up": (-1, 0),
}
DIR_ORDER = ("right", "down", "left", "up")


def _rotate_cw(cells: set[tuple[int, int]], tip: tuple[int, int], height: int, width: int):
    new_cells = {(c, height - 1 - r) for r, c in cells}
    new_tip = (tip[1], height - 1 - tip[0])
    return new_cells, new_tip, width, height


def _build_arrow_templates():
    base_variants = (
        {
            "cells": {(0, 2), (1, 0), (1, 1), (1, 2), (1, 3)},
            "tip": (1, 3),
            "height": 2,
            "width": 4,
        },
        {
            "cells": {(0, 0), (0, 1), (0, 2), (0, 3), (1, 2)},
            "tip": (0, 3),
            "height": 2,
            "width": 4,
        },
    )
    templates: list[dict] = []
    by_dir: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for variant_id, base in enumerate(base_variants):
        cells = set(base["cells"])
        tip = tuple(base["tip"])
        height = int(base["height"])
        width = int(base["width"])
        for rotation, direction_name in enumerate(DIR_ORDER):
            direction = DIRS[direction_name]
            template = {
                "direction": direction,
                "cells": frozenset(cells),
                "tip": tip,
                "height": height,
                "width": width,
                "variant": variant_id,
                "rotation": rotation,
            }
            templates.append(template)
            by_dir[direction].append(template)
            cells, tip, height, width = _rotate_cw(cells, tip, height, width)
    return tuple(templates), {k: tuple(v) for k, v in by_dir.items()}


ARROW_TEMPLATES, ARROW_TEMPLATES_BY_DIR = _build_arrow_templates()


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _background_color(grid: np.ndarray) -> int:
    flat = [int(v) for v in np.asarray(grid, dtype=int).ravel()]
    counts = Counter(flat)
    if BG in counts:
        return BG
    return int(counts.most_common(1)[0][0])


def _detect_fences(grid: np.ndarray, bg: int):
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    horizontal: dict[int, tuple[int, int]] = {}
    vertical: dict[int, tuple[int, int]] = {}
    fence_cells: set[tuple[int, int]] = set()

    for r in range(height):
        row = g[r]
        gap_cols = np.flatnonzero(row == bg)
        if gap_cols.size != 1:
            continue
        colors = {int(v) for v in row if int(v) != bg}
        if len(colors) != 1:
            continue
        if int(np.count_nonzero(row != bg)) != width - 1:
            continue
        color = int(next(iter(colors)))
        gap_col = int(gap_cols[0])
        horizontal[r] = (color, gap_col)
        for c in range(width):
            if c != gap_col:
                fence_cells.add((r, c))

    for c in range(width):
        col = g[:, c]
        gap_rows = np.flatnonzero(col == bg)
        if gap_rows.size != 1:
            continue
        colors = {int(v) for v in col if int(v) != bg}
        if len(colors) != 1:
            continue
        if int(np.count_nonzero(col != bg)) != height - 1:
            continue
        color = int(next(iter(colors)))
        gap_row = int(gap_rows[0])
        vertical[c] = (color, gap_row)
        for r in range(height):
            if r != gap_row:
                fence_cells.add((r, c))

    return horizontal, vertical, fence_cells


def _extract_arrows(grid: np.ndarray, bg: int, fence_cells: set[tuple[int, int]]):
    g = np.asarray(grid, dtype=int)
    mask = g != bg
    for r, c in fence_cells:
        mask[r, c] = False

    arrows = []
    for coords in _components4(mask):
        if coords.shape[0] != 5:
            raise AssertionError("every non-fence foreground object must be a 5-cell arrow")
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        norm = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
        matches = [tpl for tpl in ARROW_TEMPLATES if tpl["cells"] == norm]
        if len(matches) != 1:
            raise AssertionError("unrecognized arrow shape")
        tpl = matches[0]
        tip_r = r0 + int(tpl["tip"][0])
        tip_c = c0 + int(tpl["tip"][1])
        color = int(g[int(coords[0, 0]), int(coords[0, 1])])
        assert all(int(g[int(r), int(c)]) == color for r, c in coords)
        arrows.append(
            {
                "color": color,
                "direction": tuple(tpl["direction"]),
                "tip": (tip_r, tip_c),
                "coords": coords,
            }
        )
    if not arrows:
        raise AssertionError("expected at least one arrow")
    return arrows


def _extract_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    horizontal, vertical, fence_cells = _detect_fences(g, bg)
    if not horizontal and not vertical:
        raise AssertionError("expected at least one fence")
    arrows = _extract_arrows(g, bg, fence_cells)
    return bg, arrows, horizontal, vertical


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, arrows, horizontal, vertical = _extract_scene(g)
    height, width = g.shape
    out = g.copy()

    for arrow in arrows:
        color = int(arrow["color"])
        dr, dc = map(int, arrow["direction"])
        r = int(arrow["tip"][0]) + dr
        c = int(arrow["tip"][1]) + dc
        while 0 <= r < height and 0 <= c < width:
            crossed_gap = False
            if dr != 0 and r in horizontal:
                fence_color, gap_col = horizontal[r]
                if c != gap_col:
                    break
                color = int(fence_color)
                crossed_gap = True
            elif dc != 0 and c in vertical:
                fence_color, gap_row = vertical[c]
                if r != gap_row:
                    break
                color = int(fence_color)
                crossed_gap = True

            if int(g[r, c]) == bg:
                out[r, c] = color
            elif not crossed_gap:
                break

            r += dr
            c += dc

    return out


GRID_MIN = 14
GRID_MAX = 22
FENCE_COUNT_MAX = 4
ARROW_COUNT_MAX = 3
GEN_RETRIES = 400


def _choose_spaced(values: list[int], count: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    if len(values) < count:
        return None
    for _ in range(200):
        sample = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(sample, sample[1:])):
            return sample
    return None


def _cells_from_tip(template: dict, tip: tuple[int, int]) -> list[tuple[int, int]]:
    tip_r, tip_c = map(int, tip)
    rel_tip_r, rel_tip_c = map(int, template["tip"])
    return [
        (tip_r + int(r) - rel_tip_r, tip_c + int(c) - rel_tip_c)
        for r, c in template["cells"]
    ]


def _place_arrow(grid: np.ndarray, color: int, direction: tuple[int, int], tip: tuple[int, int], occupied: set[tuple[int, int]]):
    template = random.choice(ARROW_TEMPLATES_BY_DIR[direction])
    cells = _cells_from_tip(template, tip)
    height, width = grid.shape
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width):
            return None
        if int(grid[r, c]) != BG or (r, c) in occupied:
            return None
    for r, c in cells:
        grid[r, c] = color
        occupied.add((r, c))
    return template


def _gen_vertical_scene() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((height, width), BG, dtype=int)

    fence_rows_pool = list(range(6, height - 6))
    if not fence_rows_pool:
        return None

    fence_rows = None
    fence_count = 0
    for _ in range(100):
        trial_count = random.randint(1, FENCE_COUNT_MAX)
        trial_rows = _choose_spaced(fence_rows_pool, trial_count, 2)
        if trial_rows is not None:
            fence_rows = trial_rows
            fence_count = trial_count
            break
    if fence_rows is None:
        return None

    total_arrows = random.randint(1, ARROW_COUNT_MAX)
    sides = []
    while not sides:
        top_count = random.randint(0, min(2, total_arrows))
        bottom_count = total_arrows - top_count
        if bottom_count <= 2 and top_count + bottom_count == total_arrows and top_count + bottom_count > 0:
            sides = ["top"] * top_count + ["bottom"] * bottom_count
    random.shuffle(sides)

    arrow_cols_pool = list(range(1, width - 1))
    arrow_cols = _choose_spaced(arrow_cols_pool, len(sides), 3)
    if arrow_cols is None:
        return None
    random.shuffle(arrow_cols)

    top_indices = [idx for idx, side in enumerate(sides) if side == "top"]
    bottom_indices = [idx for idx, side in enumerate(sides) if side == "bottom"]

    k_top = 0
    k_bottom = 0
    if top_indices and bottom_indices:
        if random.random() < 0.5:
            k_top = random.randint(0, fence_count)
            k_bottom = random.randint(0, fence_count - k_top)
        else:
            k_bottom = random.randint(0, fence_count)
            k_top = random.randint(0, fence_count - k_bottom)
    elif top_indices:
        k_top = random.randint(0, fence_count)
    elif bottom_indices:
        k_bottom = random.randint(0, fence_count)

    colors = random.sample(PALETTE, fence_count + len(sides))
    fence_colors = colors[:fence_count]
    arrow_colors = colors[fence_count:]

    lanes = [arrow_cols[idx] for idx in range(len(sides))]
    lane_set = set(lanes)
    spare_cols = [c for c in range(width) if c not in lane_set]
    if not spare_cols:
        return None

    top_primary_col = lanes[top_indices[0]] if top_indices else None
    bottom_primary_col = lanes[bottom_indices[0]] if bottom_indices else None

    gap_cols = []
    for i in range(fence_count):
        if top_primary_col is not None and i < k_top:
            gap_cols.append(int(top_primary_col))
        elif bottom_primary_col is not None and i >= fence_count - k_bottom:
            gap_cols.append(int(bottom_primary_col))
        else:
            gap_cols.append(int(random.choice(spare_cols)))

    for row, color, gap_col in zip(fence_rows, fence_colors, gap_cols):
        grid[row, :] = color
        grid[row, gap_col] = BG

    occupied: set[tuple[int, int]] = set()
    for idx, side in enumerate(sides):
        color = int(arrow_colors[idx])
        col = int(lanes[idx])
        if side == "top":
            direction = DIRS["down"]
            tip_row = random.choice((3, 4))
        else:
            direction = DIRS["up"]
            tip_row = height - random.choice((4, 5))
        placed = _place_arrow(grid, color, direction, (tip_row, col), occupied)
        if placed is None:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_vertical_scene()
        if grid is None:
            continue
        if random.random() < 0.5:
            grid = grid.T.copy()
        try:
            inp = np.asarray(grid, dtype=int)
            out = main(inp)
            _extract_scene(inp)
            if np.array_equal(inp, out):
                continue
            if int(np.count_nonzero(out != inp)) < 2:
                continue
            return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
