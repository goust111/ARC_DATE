import random
from collections import defaultdict

import numpy as np


def _background_color(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(mask):
    rows, cols = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components = []
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _find_framed_rectangles(grid, background):
    frames = []
    for cells in _find_components(grid != background):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3 or len(cells) != height * width:
            continue

        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(region == background):
            continue

        border = np.concatenate(
            (
                region[0, :],
                region[-1, :],
                region[1:-1, 0],
                region[1:-1, -1],
            )
        )
        border_values = np.unique(border)
        inner_values = np.unique(region[1:-1, 1:-1])
        if len(border_values) != 1 or len(inner_values) != 1:
            continue

        border_color = int(border_values[0])
        fill_color = int(inner_values[0])
        if border_color == fill_color:
            continue

        frames.append(
            {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "height": height,
                "width": width,
                "border": border_color,
                "fill": fill_color,
            }
        )
    return frames


def _split_close_runs(indices, frames, axis):
    if axis == "h":
        ordered = sorted(indices, key=lambda i: frames[i]["c0"])
        start_key, end_key = "c0", "c1"
    else:
        ordered = sorted(indices, key=lambda i: frames[i]["r0"])
        start_key, end_key = "r0", "r1"

    runs = []
    current = []
    for idx in ordered:
        if current:
            gap = frames[idx][start_key] - frames[current[-1]][end_key] - 1
            limit = max(3, frames[idx]["width"] if axis == "h" else frames[idx]["height"])
            if gap < 0 or gap > limit:
                runs.append(current)
                current = []
        current.append(idx)
    if current:
        runs.append(current)
    return runs


def _choose_legend(frames, shape):
    rows, cols = shape
    candidates = []

    horizontal = defaultdict(list)
    vertical = defaultdict(list)
    for idx, frame in enumerate(frames):
        horizontal[(frame["r0"], frame["r1"], frame["height"], frame["width"])].append(idx)
        vertical[(frame["c0"], frame["c1"], frame["height"], frame["width"])].append(idx)

    for axis, groups in (("h", horizontal), ("v", vertical)):
        for indices in groups.values():
            if len(indices) < 2:
                continue
            for run in _split_close_runs(indices, frames, axis):
                if len(run) < 2:
                    continue
                fills = [frames[i]["fill"] for i in run]
                borders = [frames[i]["border"] for i in run]
                if len(set(fills)) != len(run) or len(set(borders)) != len(run):
                    continue

                r0 = min(frames[i]["r0"] for i in run)
                r1 = max(frames[i]["r1"] for i in run)
                c0 = min(frames[i]["c0"] for i in run)
                c1 = max(frames[i]["c1"] for i in run)
                edge_distance = min(r0, rows - 1 - r1, c0, cols - 1 - c1)
                span = (r1 - r0 + 1) * (c1 - c0 + 1)
                candidates.append((len(run), -edge_distance, -span, tuple(run)))

    if not candidates:
        return set()
    candidates.sort(reverse=True)
    return set(candidates[0][3])


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each filled rectangle's border by looking up its fill color in the legend.

    Input structure:
    A black background contains a compact color-band legend and several separate closed
    rectangular regions. Each legend block has a single outer border color and a single
    inner fill color. Each ordinary region initially has the same neutral border color,
    while its fill color matches exactly one legend fill color.
    Rule:
    Read the fill color inside every non-legend rectangle. Find the legend block with
    the same fill color, then replace that rectangle's border with the matched legend
    border color. The interior fill and the legend itself are left unchanged.
    Key insight:
    The interior color is the lookup key, and the legend border color is the value to
    copy onto the matching region boundary.
    """
    grid = np.asarray(grid)
    out = np.array(grid, copy=True)
    background = _background_color(grid)
    frames = _find_framed_rectangles(grid, background)
    legend = _choose_legend(frames, grid.shape)
    lookup = {frames[i]["fill"]: frames[i]["border"] for i in legend}

    for idx, frame in enumerate(frames):
        if idx in legend or frame["fill"] not in lookup:
            continue
        color = lookup[frame["fill"]]
        r0, r1 = frame["r0"], frame["r1"]
        c0, c1 = frame["c0"], frame["c1"]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def _draw_rectangle(grid, r, c, height, width, border, fill):
    grid[r : r + height, c : c + width] = fill
    grid[r, c : c + width] = border
    grid[r + height - 1, c : c + width] = border
    grid[r : r + height, c] = border
    grid[r : r + height, c + width - 1] = border


def _reserve(occupied, r, c, height, width, margin=1):
    rows, cols = occupied.shape
    r0 = max(0, r - margin)
    c0 = max(0, c - margin)
    r1 = min(rows, r + height + margin)
    c1 = min(cols, c + width + margin)
    occupied[r0:r1, c0:c1] = True


def _is_free(occupied, r, c, height, width, margin=1):
    rows, cols = occupied.shape
    r0 = max(0, r - margin)
    c0 = max(0, c - margin)
    r1 = min(rows, r + height + margin)
    c1 = min(cols, c + width + margin)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def generate_input() -> np.ndarray:
    background = 0
    neutral_border = 5
    palette = [1, 2, 3, 4, 6, 7, 8, 9]

    for _ in range(200):
        colors = palette[:]
        random.shuffle(colors)
        pair_count = random.randint(3, 4)
        border_colors = colors[:pair_count]
        fill_colors = colors[pair_count : 2 * pair_count]

        orientation = random.choice(("top", "left", "right"))
        block_height = random.randint(3, 5)
        block_width = random.randint(3, 5)
        gap = 1

        if orientation == "top":
            legend_span = pair_count * block_width + (pair_count - 1) * gap
            rows = random.randint(max(18, block_height + 12), 30)
            cols = random.randint(max(18, legend_span + 2), 30)
            legend_r = random.randint(0, 1)
            legend_c = random.randint(0, cols - legend_span)
            object_bounds = (legend_r + block_height + 2, rows, 0, cols)
        else:
            legend_span = pair_count * block_height + (pair_count - 1) * gap
            rows = random.randint(max(18, legend_span + 2), 30)
            cols = random.randint(max(18, block_width + 12), 30)
            legend_r = random.randint(0, rows - legend_span)
            if orientation == "left":
                legend_c = random.randint(0, 1)
                object_bounds = (0, rows, legend_c + block_width + 2, cols)
            else:
                legend_c = cols - block_width - random.randint(0, 1)
                object_bounds = (0, rows, 0, legend_c - 1)

        grid = np.full((rows, cols), background, dtype=int)
        occupied = np.zeros((rows, cols), dtype=bool)

        for i, (border, fill) in enumerate(zip(border_colors, fill_colors)):
            if orientation == "top":
                r = legend_r
                c = legend_c + i * (block_width + gap)
            else:
                r = legend_r + i * (block_height + gap)
                c = legend_c
            _draw_rectangle(grid, r, c, block_height, block_width, border, fill)
            _reserve(occupied, r, c, block_height, block_width)

        object_count = random.randint(pair_count, min(pair_count + 3, 7))
        object_fills = fill_colors[:] + [random.choice(fill_colors) for _ in range(object_count - pair_count)]
        random.shuffle(object_fills)

        r_min, r_max, c_min, c_max = object_bounds
        placed_all = True
        for fill in object_fills:
            placed = False
            for _try in range(300):
                height = random.randint(3, 7)
                width = random.randint(3, 8)
                if r_max - r_min < height or c_max - c_min < width:
                    continue
                r = random.randint(r_min, r_max - height)
                c = random.randint(c_min, c_max - width)
                if not _is_free(occupied, r, c, height, width):
                    continue
                _draw_rectangle(grid, r, c, height, width, neutral_border, fill)
                _reserve(occupied, r, c, height, width)
                placed = True
                break
            if not placed:
                placed_all = False
                break

        if placed_all:
            return grid

    rows, cols = 22, 24
    grid = np.full((rows, cols), background, dtype=int)
    pairs = [(1, 6), (2, 7), (3, 8)]
    for i, (border, fill) in enumerate(pairs):
        _draw_rectangle(grid, 1, 1 + i * 5, 4, 4, border, fill)
    _draw_rectangle(grid, 8, 2, 5, 6, neutral_border, 6)
    _draw_rectangle(grid, 8, 11, 4, 5, neutral_border, 7)
    _draw_rectangle(grid, 15, 5, 5, 7, neutral_border, 8)
    return grid
