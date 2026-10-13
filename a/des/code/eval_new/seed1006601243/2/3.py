import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
MAROON = 9
PATH_COLORS = tuple(range(1, 9))
MAX_GENERATION_ATTEMPTS = 300

PathSpec = Tuple[int, int, int, int]
Box = Tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_priority_legend(grid: np.ndarray, bg: int) -> Tuple[Box, List[int]]:
    candidates: List[Tuple[Box, List[int]]] = []
    for comp in _components(grid == MAROON):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1:
            continue
        if bh < 4 or bw < 3:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(sub[0, :] == MAROON)
            and np.all(sub[-1, :] == MAROON)
            and np.all(sub[:, 0] == MAROON)
            and np.all(sub[:, -1] == MAROON)
        ):
            continue
        if np.any(sub[1:-1, 1:-1] == MAROON):
            continue

        order: List[int] = []
        ok = True
        for rr in range(1, bh - 1):
            row = sub[rr, 1:-1]
            uniq = np.unique(row)
            if len(uniq) != 1:
                ok = False
                break
            color = int(uniq[0])
            if color in (bg, MAROON):
                ok = False
                break
            order.append(color)
        if ok and len(order) >= 2 and len(set(order)) == len(order):
            candidates.append(((r0, c0, r1, c1), order))

    assert len(candidates) == 1, "expected exactly one maroon priority legend"
    return candidates[0]


def _marker_for_color(grid: np.ndarray, color: int) -> PathSpec:
    h, w = grid.shape
    coords = np.argwhere(grid == color)
    edge_cells = [
        (int(r), int(c))
        for r, c in coords
        if int(r) in (0, h - 1) or int(c) in (0, w - 1)
    ]
    assert len(edge_cells) == 1, f"expected one edge start for color {color}"
    r, c = edge_cells[0]

    candidates: List[PathSpec] = []
    if r == 0 and h > 1 and int(grid[1, c]) == color:
        candidates.append((r, c, 1, 0))
    if r == h - 1 and h > 1 and int(grid[h - 2, c]) == color:
        candidates.append((r, c, -1, 0))
    if c == 0 and w > 1 and int(grid[r, 1]) == color:
        candidates.append((r, c, 0, 1))
    if c == w - 1 and w > 1 and int(grid[r, w - 2]) == color:
        candidates.append((r, c, 0, -1))

    assert len(candidates) == 1, f"expected one inward direction for color {color}"
    return candidates[0]


def _render_paths(shape: Tuple[int, int], priority: Sequence[int], specs: Dict[int, PathSpec], bg: int) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    for color in priority:
        r, c, dr, dc = specs[color]
        while 0 <= r < h and 0 <= c < w:
            if out[r, c] == bg:
                out[r, c] = int(color)
            r += dr
            c += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend edge-started colored paths across the grid using a maroon legend to decide weave priority.

    Input structure:
    - The background is black.
    - Several distinct colors appear as two-cell edge markers: one cell on the border and one same-color cell immediately inward, which together encode a straight inward path direction.
    - A single maroon rectangular legend sits away from the border; its maroon border encloses one-pixel-tall colored bars stacked from top to bottom.

    Rule:
    - Read the legend bars from top to bottom to obtain the path drawing order from low priority to high priority.
    - For each legend color, find its unique edge marker and extend that color straight across the grid toward the opposite side.
    - When a later path reaches a cell already painted by an earlier path, keep the earlier color and let the later path continue past the crossing; clear the legend area from the output.

    Key insight:
    - The visible arrow markers give only the travel directions; the maroon legend secretly controls which color survives at crossings because paths are rendered in that top-to-bottom order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    _legend_box, priority = _find_priority_legend(g, bg)

    edge_colors = {
        int(g[r, c])
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if (r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1)) and int(g[r, c]) not in (bg, MAROON)
    }
    assert edge_colors == set(priority), "legend colors must match edge path colors"

    specs = {color: _marker_for_color(g, color) for color in priority}
    return _render_paths(g.shape, priority, specs, bg)


def _choose_specs(
    h: int,
    w: int,
    horizontal_rows: Sequence[int],
    vertical_cols: Sequence[int],
) -> List[PathSpec] | None:
    for _ in range(50):
        specs: List[PathSpec] = []
        used = set()
        ok = True

        for row in horizontal_rows:
            if random.random() < 0.5:
                spec = (int(row), 0, 0, 1)
            else:
                spec = (int(row), w - 1, 0, -1)
            cells = {(spec[0], spec[1]), (spec[0] + spec[2], spec[1] + spec[3])}
            if used & cells:
                ok = False
                break
            used |= cells
            specs.append(spec)
        if not ok:
            continue

        for col in vertical_cols:
            if random.random() < 0.5:
                spec = (0, int(col), 1, 0)
            else:
                spec = (h - 1, int(col), -1, 0)
            cells = {(spec[0], spec[1]), (spec[0] + spec[2], spec[1] + spec[3])}
            if used & cells:
                ok = False
                break
            used |= cells
            specs.append(spec)
        if ok:
            return specs
    return None


def _draw_legend(grid: np.ndarray, box: Box, priority: Sequence[int], inner_width: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = MAROON
    grid[r1, c0 : c1 + 1] = MAROON
    grid[r0 : r1 + 1, c0] = MAROON
    grid[r0 : r1 + 1, c1] = MAROON
    for i, color in enumerate(priority):
        rr = r0 + 1 + i
        grid[rr, c0 + 1 : c0 + 1 + inner_width] = int(color)


def _draw_markers(grid: np.ndarray, specs: Dict[int, PathSpec]) -> None:
    for color, (r, c, dr, dc) in specs.items():
        grid[r, c] = int(color)
        grid[r + dr, c + dc] = int(color)


def _gen_attempt() -> np.ndarray | None:
    total_paths = random.randint(3, 6)
    horizontal_count = random.randint(1, total_paths - 1)
    vertical_count = total_paths - horizontal_count

    inner_width = random.randint(1, 3)
    legend_h = total_paths + 2
    legend_w = inner_width + 2

    min_h = max(12, legend_h + horizontal_count + 4)
    min_w = max(12, legend_w + vertical_count + 4)
    max_h = 24
    max_w = 24
    if min_h > max_h or min_w > max_w:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)

    legend_r0 = random.randint(1, h - legend_h - 1)
    legend_c0 = random.randint(1, w - legend_w - 1)
    legend_box = (
        legend_r0,
        legend_c0,
        legend_r0 + legend_h - 1,
        legend_c0 + legend_w - 1,
    )

    blocked_rows = set(range(legend_box[0], legend_box[2] + 1))
    blocked_cols = set(range(legend_box[1], legend_box[3] + 1))
    available_rows = [r for r in range(1, h - 1) if r not in blocked_rows]
    available_cols = [c for c in range(1, w - 1) if c not in blocked_cols]
    if len(available_rows) < horizontal_count or len(available_cols) < vertical_count:
        return None

    horizontal_rows = random.sample(available_rows, horizontal_count)
    vertical_cols = random.sample(available_cols, vertical_count)
    specs_list = _choose_specs(h, w, horizontal_rows, vertical_cols)
    if specs_list is None:
        return None

    colors = random.sample(PATH_COLORS, total_paths)
    random.shuffle(specs_list)
    specs = {color: spec for color, spec in zip(colors, specs_list)}

    priority = colors[:]
    random.shuffle(priority)

    grid = np.full((h, w), BG, dtype=int)
    _draw_legend(grid, legend_box, priority, inner_width)
    _draw_markers(grid, specs)

    expected = _render_paths((h, w), priority, specs, BG)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(solved, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-path input")
