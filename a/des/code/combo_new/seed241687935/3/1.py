from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
GEN_RETRIES = 400
PLACEMENT_RETRIES = 250


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _parse_runs(line: np.ndarray) -> list[tuple[int, int, int, int]]:
    runs: list[tuple[int, int, int, int]] = []
    if line.size == 0:
        return runs
    start = 0
    vals = [int(v) for v in line]
    while start < len(vals):
        end = start + 1
        while end < len(vals) and vals[end] == vals[start]:
            end += 1
        runs.append((start, end - 1, vals[start], end - start))
        start = end
    return runs


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, str, int, int, list[tuple[int, int, int, int]]]] = []
    edge_specs = [
        ("top", grid[0, :]),
        ("bottom", grid[h - 1, :]),
        ("left", grid[:, 0]),
        ("right", grid[:, w - 1]),
    ]
    for side, line in edge_specs:
        idx = np.flatnonzero(line != bg)
        if idx.size == 0:
            continue
        start = int(idx[0])
        end = int(idx[-1])
        segment = np.asarray(line[start : end + 1], dtype=int)
        if np.any(segment == bg):
            continue
        runs = _parse_runs(segment)
        lengths = [length for _, _, _, length in runs]
        if len(runs) < 2 or len(set(lengths)) != len(lengths):
            continue
        candidates.append((segment.size, len(runs), side, start, end, runs))
    assert candidates, "expected one edge legend made of distinct-length color runs"

    _, _, side, start, end, runs = max(candidates, key=lambda item: (item[0], item[1]))
    mapping = {int(length): int(color) for _, _, color, length in runs}
    mask = np.zeros_like(grid, dtype=bool)
    if side == "top":
        mask[0, start : end + 1] = True
    elif side == "bottom":
        mask[h - 1, start : end + 1] = True
    elif side == "left":
        mask[start : end + 1, 0] = True
    else:
        mask[start : end + 1, w - 1] = True
    return mapping, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced enclosure with the legend color indexed by its internal marker count.

    Input structure:
    - The background is the most common color.
    - One outer edge contains a one-cell-thick horizontal or vertical legend strip made of contiguous same-color runs with distinct lengths.
    - The main area contains several closed ring fences, all drawn in one shared color; inside each fence are background cells and possibly some single-pixel marker cells of another fixed color.

    Rule:
    - Detect the edge legend and build a length-to-color mapping from its run lengths.
    - Treat each 4-connected non-fence region that does not touch the grid boundary as the interior of one fence.
    - Count how many marker pixels lie in that interior; use the legend color whose run length equals that count, except that count 0 uses the shortest run of length 1.
    - Recolor every non-fence cell inside that enclosure, including markers, to the selected color; keep the fences and the legend unchanged.

    Key insight:
    - The legend does not map by color identity or order: the visible run length is the index, so enclosure marker-counts must be converted into run lengths before choosing the fill color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode_color(g)

    legend_by_length, legend_mask = _detect_legend(g, bg)

    nonlegend_vals = [int(v) for v in g[~legend_mask].ravel() if int(v) != bg]
    assert nonlegend_vals, "expected fences away from the legend"
    nonlegend_counts = Counter(nonlegend_vals)
    fence_color = int(nonlegend_counts.most_common(1)[0][0])
    marker_colors = [color for color in nonlegend_counts if color != fence_color]
    assert len(marker_colors) <= 1, "expected at most one marker color outside the legend"
    marker_color = marker_colors[0] if marker_colors else None

    out = g.copy()
    for comp in _components(g != fence_color):
        if _touches_border(comp, h, w):
            continue
        marker_count = 0
        if marker_color is not None:
            marker_count = sum(int(g[r, c]) == int(marker_color) for r, c in comp)
        key = max(1, int(marker_count))
        assert key in legend_by_length, "marker count must be covered by the legend lengths"
        fill = int(legend_by_length[key])
        for r, c in comp:
            out[r, c] = fill

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _usable_bounds(h: int, w: int, side: str) -> tuple[int, int, int, int]:
    top_margin = 2 if side == "top" else 1
    bottom_margin = 2 if side == "bottom" else 1
    left_margin = 2 if side == "left" else 1
    right_margin = 2 if side == "right" else 1
    return (
        top_margin,
        h - 1 - bottom_margin,
        left_margin,
        w - 1 - right_margin,
    )


def _sample_marker_cells(
    rng: random.Random, inner_h: int, inner_w: int, count: int
) -> list[tuple[int, int]]:
    if count == 0:
        return []
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    rng.shuffle(cells)
    picked: list[tuple[int, int]] = []
    for cell in cells:
        if all(abs(cell[0] - pr) + abs(cell[1] - pc) > 1 for pr, pc in picked):
            picked.append(cell)
            if len(picked) == count:
                return picked
    leftovers = [cell for cell in cells if cell not in picked]
    for cell in leftovers:
        picked.append(cell)
        if len(picked) == count:
            return picked
    raise AssertionError("not enough interior cells for markers")


def _draw_legend(
    grid: np.ndarray, side: str, entries: list[tuple[int, int]]
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    pos = 0
    h, w = grid.shape
    for length, color in entries:
        mapping[int(length)] = int(color)
        if side == "top":
            grid[0, pos : pos + length] = color
        elif side == "bottom":
            grid[h - 1, pos : pos + length] = color
        elif side == "left":
            grid[pos : pos + length, 0] = color
        else:
            grid[pos : pos + length, w - 1] = color
        pos += int(length)
    return mapping


def _draw_fence(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    bg_count = counts.pop(int(bg), 0)
    return bool(counts) and bg_count > max(counts.values())


def _sample_counts(rng: random.Random, n_fences: int, max_count: int) -> list[int]:
    counts = [0, rng.randint(2, max_count)]
    while len(counts) < n_fences:
        counts.append(rng.randint(0, max_count))
    rng.shuffle(counts)
    return counts


def _place_fences(
    rng: random.Random,
    h: int,
    w: int,
    side: str,
    counts: list[int],
) -> list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]] | None:
    rmin, rmax, cmin, cmax = _usable_bounds(h, w, side)
    usable_h = rmax - rmin + 1
    usable_w = cmax - cmin + 1
    if usable_h < 4 or usable_w < 4:
        return None

    size_choices = [
        (oh, ow)
        for oh in range(4, min(8, usable_h) + 1)
        for ow in range(4, min(8, usable_w) + 1)
    ]
    if not size_choices:
        return None

    placed: list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]] = []
    padded_boxes: list[tuple[int, int, int, int]] = []

    for count in counts:
        fitting = [
            (oh, ow)
            for oh, ow in size_choices
            if (oh - 2) * (ow - 2) >= count + 1
        ]
        if not fitting:
            return None

        success = False
        for _ in range(PLACEMENT_RETRIES):
            oh, ow = rng.choice(fitting)
            rr0 = rng.randint(rmin, rmax - oh + 1)
            cc0 = rng.randint(cmin, cmax - ow + 1)
            box = (rr0, cc0, rr0 + oh - 1, cc0 + ow - 1)
            padded = (box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1)
            if any(_boxes_overlap(padded, other) for other in padded_boxes):
                continue

            inner_h = oh - 2
            inner_w = ow - 2
            markers = _sample_marker_cells(rng, inner_h, inner_w, count)
            placed.append((box, markers))
            padded_boxes.append(padded)
            success = True
            break
        if not success:
            return None

    return placed


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    max_count = rng.choice([4, 5])
    lengths = list(range(1, max_count + 1))
    total_len = sum(lengths)
    side = rng.choice(["top", "bottom", "left", "right"])

    if side in ("top", "bottom"):
        h = rng.randint(14, 24)
        w = total_len
    else:
        h = total_len
        w = rng.randint(14, 24)

    grid = np.full((h, w), BACKGROUND, dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    fence_color = palette.pop()
    marker_color = palette.pop()
    legend_colors = palette[:max_count]

    legend_entries = list(zip(lengths, legend_colors))
    rng.shuffle(legend_entries)
    _draw_legend(grid, side, legend_entries)

    rmin, rmax, cmin, cmax = _usable_bounds(h, w, side)
    usable_h = rmax - rmin + 1
    usable_w = cmax - cmin + 1
    if min(usable_h, usable_w) <= 0:
        return None

    if usable_h * usable_w >= 130 and min(usable_h, usable_w) >= 9:
        n_options = [2, 3, 4]
    else:
        n_options = [2, 3]
    n_fences = rng.choice(n_options)
    counts = _sample_counts(rng, n_fences, max_count)

    placements = _place_fences(rng, h, w, side, counts)
    if placements is None:
        return None

    for box, markers in placements:
        _draw_fence(grid, box, fence_color)
        r0, c0, _, _ = box
        for mr, mc in markers:
            grid[r0 + 1 + mr, c0 + 1 + mc] = marker_color

    if not _bg_is_dominant(grid, BACKGROUND):
        return None

    legend_mask = np.zeros_like(grid, dtype=bool)
    if side == "top":
        legend_mask[0, :] = True
    elif side == "bottom":
        legend_mask[h - 1, :] = True
    elif side == "left":
        legend_mask[:, 0] = True
    else:
        legend_mask[:, w - 1] = True

    counts_outside_legend = Counter(
        int(v) for v in grid[~legend_mask].ravel() if int(v) != BACKGROUND
    )
    if counts_outside_legend.most_common(1)[0][0] != fence_color:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
