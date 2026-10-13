from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scan_runs(row: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    c = 0
    while c < row.size:
        if int(row[c]) == bg:
            c += 1
            continue
        c0 = c
        color = int(row[c])
        while c < row.size and int(row[c]) == color:
            c += 1
        c1 = c - 1
        runs.append((c0, c1, color))
    return runs


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[int, dict[int, int]]:
    nonempty_rows = np.flatnonzero(np.any(grid != bg, axis=1))
    assert nonempty_rows.size > 0, "expected foreground objects"
    legend_row = int(nonempty_rows[-1])
    runs = _scan_runs(grid[legend_row], bg)
    assert len(runs) >= 3, "expected several bottom legend bars"

    length_to_color: dict[int, int] = {}
    seen_colors: set[int] = set()
    for c0, c1, color in runs:
        length = c1 - c0 + 1
        assert color not in seen_colors, "legend bar colors must be distinct"
        assert length not in length_to_color, "legend bar lengths must be distinct"
        seen_colors.add(color)
        length_to_color[length] = color
    return legend_row, length_to_color


def _solid_rectangles(grid: np.ndarray, bg: int, legend_row: int) -> list[tuple[int, int, int, int, int]]:
    body = grid[:legend_row, :]
    rectangles: list[tuple[int, int, int, int, int]] = []
    for coords in _components(body != bg):
        r0, c0, r1, c1 = _bbox(coords)
        color = int(body[coords[0, 0], coords[0, 1]])
        region = body[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(region == color), "foreground components above the legend must be solid rectangles"
        rectangles.append((r0, c0, r1, c1, color))
    assert rectangles, "expected at least one solid rectangle above the legend"
    return rectangles


def _apply_ring_rule(
    out: np.ndarray,
    rectangles: list[tuple[int, int, int, int, int]],
    length_to_color: dict[int, int],
    bg: int,
) -> np.ndarray:
    for r0, c0, r1, c1, _color in rectangles:
        depth = _depth_map(r1 - r0 + 1, c1 - c0 + 1)
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        for d in range(1, int(depth.max()) + 1):
            region[depth == d] = int(length_to_color.get(d + 1, bg))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each rectangle's concentric rings from bottom legend-bar lengths, hollowing rings whose lengths are missing.

    Input structure
    - The background is the most common color.
    - The bottom occupied row contains several separated horizontal legend bars with distinct colors and distinct lengths; their left-right order is irrelevant.
    - Above the legend lie one or more filled monochrome axis-aligned rectangles.

    Rule
    - Treat each rectangle as concentric rectangular rings measured by edge distance.
    - Ring 1, the outer border, keeps the rectangle's original color.
    - For every ring k >= 2: if a legend bar has length exactly k, recolor the whole ring to that bar's color; otherwise recolor the whole ring to the background.
    - Leave the legend bars unchanged.

    Key insight
    - The legend is indexed by bar length rather than by position, and missing lengths matter just as much as present ones because an absent entry hollows out that entire ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    legend_row, length_to_color = _parse_legend(g, bg)
    rectangles = _solid_rectangles(g, bg, legend_row)

    out = g.copy()
    return _apply_ring_rule(out, rectangles, length_to_color, bg)


def _ring_count(h: int, w: int) -> int:
    return (min(h, w) + 1) // 2


def _depth_map(h: int, w: int) -> np.ndarray:
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    return np.minimum(np.minimum(rows, h - 1 - rows), np.minimum(cols, w - 1 - cols))


def _rects_separated(
    candidate: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], pad: int = 1
) -> bool:
    r0, c0, r1, c1 = candidate
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + pad < pr0 or pr1 + pad < r0 or c1 + pad < pc0 or pc1 + pad < c0):
            return False
    return True


def _choose_lengths(max_used_ring: int) -> tuple[list[int], set[int]]:
    base = list(range(2, max_used_ring + 1))
    missing_count = random.randint(1, max(1, len(base) - 2))
    missing = set(random.sample(base, k=missing_count))
    present = [k for k in base if k not in missing]
    if len(present) < 2:
        return _choose_lengths(max_used_ring)

    all_lengths = list(present)
    extra_choices = [k for k in range(max_used_ring + 1, min(9, max_used_ring + 3))]
    if extra_choices and random.random() < 0.55:
        all_lengths.append(random.choice(extra_choices))
    random.shuffle(all_lengths)
    return all_lengths, missing


def _bar_order(lengths: list[int]) -> list[int]:
    if len(lengths) < 3:
        return lengths
    for _ in range(50):
        order = lengths[:]
        random.shuffle(order)
        if order != sorted(order) and order != sorted(order, reverse=True):
            return order
    return lengths


def _place_rectangles(
    height: int, width: int, rect_dims: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    for h, w in sorted(rect_dims, key=lambda x: x[0] * x[1], reverse=True):
        positions = [
            (r0, c0)
            for r0 in range(height - h + 1)
            for c0 in range(width - w + 1)
            if _rects_separated((r0, c0, r0 + h - 1, c0 + w - 1), placed, pad=1)
        ]
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        placed.append((r0, c0, r0 + h - 1, c0 + w - 1))
    return placed


def _design_output(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    rect_colors: list[int],
    length_to_color: dict[int, int],
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    for (r0, c0, r1, c1), color in zip(rects, rect_colors):
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return _apply_ring_rule(
        out,
        [(r0, c0, r1, c1, color) for (r0, c0, r1, c1), color in zip(rects, rect_colors)],
        length_to_color,
        bg,
    )


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    rect_colors: list[int],
    length_to_color: dict[int, int],
    missing: set[int],
    legend_order: list[int],
    bg: int,
) -> bool:
    if np.array_equal(inp, out):
        return False
    if legend_order == sorted(legend_order) or legend_order == sorted(legend_order, reverse=True):
        return False

    ring_counts = [_ring_count(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in rects]
    if len(set(ring_counts)) < 2:
        return False

    used_present = {k for k in length_to_color if any(rc >= k for rc in ring_counts)}
    used_missing = {k for k in missing if any(rc >= k for rc in ring_counts)}
    if len(used_present & set(range(2, max(used_present | {2}) + 1))) < 2:
        return False
    if not used_missing:
        return False

    recolor_seen: set[int] = set()
    hollow_seen = False
    for (r0, c0, r1, c1), color in zip(rects, rect_colors):
        depth = _depth_map(r1 - r0 + 1, c1 - c0 + 1)
        in_region = inp[r0 : r1 + 1, c0 : c1 + 1]
        out_region = out[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(in_region[depth == 0] == color):
            return False
        for d in range(1, int(depth.max()) + 1):
            expected = length_to_color.get(d + 1, bg)
            vals = set(map(int, np.unique(out_region[depth == d])))
            if vals != {int(expected)}:
                return False
            if expected == bg:
                hollow_seen = True
            else:
                recolor_seen.add(int(expected))
    if not hollow_seen or len(recolor_seen) < 2:
        return False

    if not np.array_equal(main(inp), out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    bg = random.randint(0, 9)

    max_used_ring = random.choices([4, 5, 6], weights=[4, 4, 2], k=1)[0]
    if 2 * max_used_ring - 1 > min(height - 2, width):
        return None
    legend_lengths, missing = _choose_lengths(max_used_ring)
    legend_order = _bar_order(legend_lengths)

    bar_count = len(legend_order)
    max_rect_count = min(4, 9 - bar_count)
    rect_count = random.randint(2, max_rect_count)

    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    legend_colors = colors[:bar_count]
    rect_colors = colors[bar_count : bar_count + rect_count]
    length_to_color = {length: color for length, color in zip(legend_order, legend_colors)}

    top_height = height - 2
    large_min = 2 * max_used_ring - 1
    rect_dims: list[tuple[int, int]] = [
        (
            random.randint(large_min, min(12, top_height)),
            random.randint(large_min, min(12, width)),
        )
    ]
    for _ in range(rect_count - 1):
        rect_dims.append(
            (
                random.randint(4, min(10, top_height)),
                random.randint(4, min(10, width)),
            )
        )

    if len({_ring_count(h, w) for h, w in rect_dims}) < 2:
        rect_dims[-1] = (4, random.randint(6, min(10, width)))

    rects = _place_rectangles(top_height, width, rect_dims)
    if rects is None:
        return None

    gap_sizes = [random.randint(1, 3) for _ in range(bar_count - 1)]
    bar_span = sum(legend_order) + sum(gap_sizes)
    if bar_span > width:
        return None
    left_margin = random.randint(0, width - bar_span)

    grid = np.full((height, width), bg, dtype=int)
    for (r0, c0, r1, c1), color in zip(rects, rect_colors):
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    c = left_margin
    for i, length in enumerate(legend_order):
        grid[height - 1, c : c + length] = length_to_color[length]
        c += length
        if i < len(gap_sizes):
            c += gap_sizes[i]

    out = _design_output(grid, rects, rect_colors, length_to_color, bg)
    if not _semantic_checks(grid, out, rects, rect_colors, length_to_color, missing, legend_order, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
