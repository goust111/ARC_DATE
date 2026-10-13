import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = tuple(color for color in range(1, 10) if color != GREY)

GRID_H_RANGE = (16, 25)
GRID_W_RANGE = (16, 25)
COUNT_CONFIGS = ((1, 1), (2, 1), (1, 2), (2, 2))
LEGEND_SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _border_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = [(0, c) for c in range(w)]
    cells.extend((r, w - 1) for r in range(1, h))
    cells.extend((h - 1, c) for c in range(w - 2, -1, -1))
    cells.extend((r, 0) for r in range(h - 2, 0, -1))
    return cells


def _legend_info(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    border_coords = _border_cells(grid.shape)
    border_non_bg = [(r, c) for r, c in border_coords if int(grid[r, c]) != bg]
    assert border_non_bg, "expected a border legend"

    border_colors = {int(grid[r, c]) for r, c in border_non_bg}
    assert len(border_colors) == 1, "legend border strip must use one color"
    grey = next(iter(border_colors))

    strip = np.argwhere(grid == grey)
    assert len(strip) >= 2, "legend strip must have length at least 2"
    r0, c0 = np.min(strip, axis=0)
    r1, c1 = np.max(strip, axis=0)
    mask = np.zeros_like(grid, dtype=bool)

    if r0 == r1:
        assert c1 > c0, "horizontal legend must span multiple cells"
        strip_row = int(r0)
        color_row = 1 if strip_row == 0 else h - 2
        order = [int(grid[color_row, c]) for c in range(int(c0), int(c1) + 1)]
        assert all(color not in (bg, grey) for color in order), "legend colors must sit next to the grey strip"
        assert len(set(order)) == len(order), "legend colors must be distinct"
        mask[strip_row, int(c0) : int(c1) + 1] = True
        mask[color_row, int(c0) : int(c1) + 1] = True
    else:
        assert c0 == c1, "legend strip must be 1xn or nx1"
        strip_col = int(c0)
        color_col = 1 if strip_col == 0 else w - 2
        order = [int(grid[r, color_col]) for r in range(int(r0), int(r1) + 1)]
        assert all(color not in (bg, grey) for color in order), "legend colors must sit next to the grey strip"
        assert len(set(order)) == len(order), "legend colors must be distinct"
        mask[int(r0) : int(r1) + 1, strip_col] = True
        mask[int(r0) : int(r1) + 1, color_col] = True

    return order, mask


def _walk_non_bg(
    scene: np.ndarray,
    bg: int,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = scene.shape
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and int(scene[r, c]) != bg:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _source_and_segment(scene: np.ndarray, bg: int, color: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(scene == color)]
    coord_set = set(coords)
    assert coords, "every legend color must appear in the scene"

    candidates: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []
    for extra in coords:
        same_neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(*extra)
            if _in_bounds(scene.shape, nr, nc) and int(scene[nr, nc]) == color
        ]
        if len(same_neighbors) != 1:
            continue

        start = same_neighbors[0]
        if start[0] != extra[0]:
            directions = ((0, -1), (0, 1))
            axis_ok = all(cell == extra or cell[0] == start[0] for cell in coords)
        else:
            directions = ((-1, 0), (1, 0))
            axis_ok = all(cell == extra or cell[1] == start[1] for cell in coords)
        if not axis_ok:
            continue

        forward = []
        for dr, dc in directions:
            nr, nc = start[0] + dr, start[1] + dc
            if not _in_bounds(scene.shape, nr, nc):
                continue
            if (nr, nc) == extra:
                continue
            if int(scene[nr, nc]) != bg:
                forward.append((dr, dc))
        if len(forward) != 1:
            continue

        segment = _walk_non_bg(scene, bg, start, forward[0])
        axis_cells = {cell for cell in coord_set if cell != extra}
        if not axis_cells.issubset(segment):
            continue
        candidates.append((extra, start, segment))

    assert len(candidates) == 1, "expected exactly one source marker per path color"
    extra, _start, segment = candidates[0]
    return extra, segment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each colored path from its doubled source marker and keep only the winning prefixes at crossings.

    Input structure:
    - The background is the dominant color.
    - A border legend consists of a grey 1xn or nx1 strip on the grid edge with one adjacent row or column of distinct colored pixels; their visible order gives path priority from high to low.
    - Each legend color appears exactly once in the scene as a single straight horizontal or vertical path.
    - One end of each path is marked by a perpendicular two-pixel same-color source block, while crossings with other paths stay non-background but may visibly show either crossing color.

    Rule:
    - Ignore the legend when analyzing the scene, then reconstruct each path from its doubled source marker and follow the straight non-background corridor to the far end.
    - Process paths in legend priority order from highest to lowest.
    - A path survives until its first intersection with any already-kept higher-priority path; that crossing cell belongs to the higher-priority path, and the lower-priority path is erased from that cell onward toward its non-source end.
    - Paths that never hit a higher-priority survivor stay intact, and all source markers remain.

    Key insight:
    - The crossing outcome is determined by the subtle edge legend, so the final picture can be built by reconstructing every straight path and then truncating lower-priority paths at their first encounter with an already-surviving higher-priority prefix.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    order, legend_mask = _legend_info(g, bg)

    scene = g.copy()
    scene[legend_mask] = bg

    out = g.copy()
    out[scene != bg] = bg

    occupied: set[tuple[int, int]] = set()
    for color in order:
        extra, segment = _source_and_segment(scene, bg, color)
        cut_idx = None
        for idx, cell in enumerate(segment[1:], start=1):
            if cell in occupied:
                cut_idx = idx
                break

        keep = segment if cut_idx is None else segment[:cut_idx]
        out[extra] = color
        occupied.add(extra)
        for r, c in keep:
            out[r, c] = color
            occupied.add((r, c))

    return out


def _pick_spaced(values: list[int], count: int) -> list[int] | None:
    if count == 0:
        return []
    values = values[:]
    random.shuffle(values)
    chosen: list[int] = []
    for value in values:
        if all(abs(value - other) >= 3 for other in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _legend_layout(h: int, w: int, count: int) -> tuple[str, int]:
    side = random.choice(LEGEND_SIDES)
    if side in ("top", "bottom"):
        start = random.randint(1, w - count - 1)
    else:
        start = random.randint(1, h - count - 1)
    return side, start


def _usable_bounds(h: int, w: int, legend_side: str) -> tuple[int, int, int, int]:
    top = 3 + (1 if legend_side == "top" else 0)
    bottom = h - 4 - (1 if legend_side == "bottom" else 0)
    left = 3 + (1 if legend_side == "left" else 0)
    right = w - 4 - (1 if legend_side == "right" else 0)
    return top, bottom, left, right


def _candidate_axis_positions(lo: int, hi: int) -> list[int]:
    return list(range(lo, hi + 1))


def _extra_choices_h(row: int, h_rows: list[int], shape: tuple[int, int], start_col: int) -> list[int]:
    choices = []
    for delta in (-1, 1):
        rr = row + delta
        if not (0 <= rr < shape[0]):
            continue
        if rr in h_rows:
            continue
        choices.append(delta)
    return choices


def _extra_choices_v(col: int, v_cols: list[int], shape: tuple[int, int], start_row: int) -> list[int]:
    choices = []
    for delta in (-1, 1):
        cc = col + delta
        if not (0 <= cc < shape[1]):
            continue
        if cc in v_cols:
            continue
        choices.append(delta)
    return choices


def _place_legend(grid: np.ndarray, side: str, start: int, colors: list[int]) -> None:
    count = len(colors)
    h, w = grid.shape
    if side == "top":
        grid[0, start : start + count] = GREY
        grid[1, start : start + count] = colors
    elif side == "bottom":
        grid[h - 1, start : start + count] = GREY
        grid[h - 2, start : start + count] = colors
    elif side == "left":
        grid[start : start + count, 0] = GREY
        grid[start : start + count, 1] = np.array(colors, dtype=int)
    else:
        grid[start : start + count, w - 1] = GREY
        grid[start : start + count, w - 2] = np.array(colors, dtype=int)


def _render_input(specs: list[dict], legend_side: str, legend_start: int, order: list[int], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    _place_legend(grid, legend_side, legend_start, order)

    occupancy: dict[tuple[int, int], list[int]] = {}
    for spec in specs:
        cells = [spec["extra"]] + spec["segment"]
        for cell in cells:
            occupancy.setdefault(cell, []).append(spec["color"])

    for cell, colors in occupancy.items():
        r, c = cell
        if len(colors) == 1:
            grid[r, c] = colors[0]
        else:
            grid[r, c] = random.choice(colors)
    return grid


def _build_specs(h: int, w: int, h_count: int, v_count: int, legend_side: str) -> tuple[list[dict], list[int], int] | None:
    total = h_count + v_count
    order = random.sample(PALETTE, total)
    top, bottom, left, right = _usable_bounds(h, w, legend_side)
    if bottom - top < 4 or right - left < 4:
        return None

    row_candidates = _candidate_axis_positions(top + 1, bottom - 1)
    col_candidates = _candidate_axis_positions(left + 1, right - 1)
    h_rows = _pick_spaced(row_candidates, h_count)
    v_cols = _pick_spaced(col_candidates, v_count)
    if h_rows is None or v_cols is None:
        return None

    specs: list[dict] = []
    for idx, row in enumerate(h_rows):
        color = order[idx]
        source_side = random.choice(("left", "right"))
        start_col = left if source_side == "left" else right
        end_col = right if source_side == "left" else left
        delta_choices = _extra_choices_h(row, h_rows, (h, w), start_col)
        if not delta_choices:
            return None
        extra_row = row + random.choice(delta_choices)
        step = 1 if end_col > start_col else -1
        segment = [(row, c) for c in range(start_col, end_col + step, step)]
        specs.append(
            {
                "color": color,
                "extra": (extra_row, start_col),
                "segment": segment,
            }
        )

    for offset, col in enumerate(v_cols):
        color = order[h_count + offset]
        source_side = random.choice(("top", "bottom"))
        start_row = top if source_side == "top" else bottom
        end_row = bottom if source_side == "top" else top
        delta_choices = _extra_choices_v(col, v_cols, (h, w), start_row)
        if not delta_choices:
            return None
        extra_col = col + random.choice(delta_choices)
        step = 1 if end_row > start_row else -1
        segment = [(r, col) for r in range(start_row, end_row + step, step)]
        specs.append(
            {
                "color": color,
                "extra": (start_row, extra_col),
                "segment": segment,
            }
        )

    occupied = {}
    for spec in specs:
        for cell in [spec["extra"]] + spec["segment"]:
            occupied[cell] = occupied.get(cell, 0) + 1

    intersections = sum(1 for count in occupied.values() if count == 2)
    extras = {spec["extra"] for spec in specs}
    if intersections == 0:
        return None
    if any(occupied[cell] != 1 for cell in extras):
        return None
    return specs, order, intersections


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    h_count, v_count = random.choice(COUNT_CONFIGS)
    legend_side, legend_start = _legend_layout(h, w, h_count + v_count)
    built = _build_specs(h, w, h_count, v_count, legend_side)
    if built is None:
        return None
    specs, order, intersections = built

    grid = _render_input(specs, legend_side, legend_start, order, (h, w))
    try:
        out = main(grid)
    except Exception:
        return None

    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if intersections >= 2 and np.array_equal(grid, out):
        return None
    if np.count_nonzero(grid == GREY) != len(order):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid path-priority sample")
