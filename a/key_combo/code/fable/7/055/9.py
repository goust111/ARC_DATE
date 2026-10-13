from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _uniform_separator_rows(grid: np.ndarray) -> tuple[int, list[int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]):
            candidates.append((r, int(row[0])))
    assert candidates, "expected at least one full-row separator"
    sep_color = Counter(color for _r, color in candidates).most_common(1)[0][0]
    sep_rows = [r for r, color in candidates if color == sep_color]
    assert sep_rows, "expected separator rows of one repeated color"
    return int(sep_color), sep_rows


def _extract_structure(grid: np.ndarray) -> tuple[list[np.ndarray], list[int], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_color, sep_rows = _uniform_separator_rows(g)
    last_sep = sep_rows[-1]
    assert last_sep + 1 == g.shape[0] - 1, "expected exactly one legend row below the last separator"

    panels: list[np.ndarray] = []
    start = 0
    for sep_r in sep_rows:
        panel = g[start:sep_r, :]
        assert panel.shape[0] > 0, "separator rows may not be adjacent"
        panels.append(panel.copy())
        start = sep_r + 1
    assert start == g.shape[0] - 1, "legend row must follow the last separator"

    panel_stack = np.concatenate(panels, axis=0)
    bg = _get_bg(panel_stack)

    legend_row = g[-1]
    legend_cols = [c for c in range(g.shape[1]) if int(legend_row[c]) not in (bg, sep_color)]
    assert legend_cols, "expected a non-empty color legend on the last row"
    assert legend_cols == list(range(legend_cols[0], legend_cols[-1] + 1)), "legend pixels must be contiguous"
    legend = [int(legend_row[c]) for c in legend_cols]
    assert len(legend) == len(set(legend)), "legend colors must be distinct"
    return panels, legend, int(bg), int(sep_color)


def _panel_region_count(panel: np.ndarray, bg: int) -> int:
    return _count_components(panel == int(bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the background regions in each panel and use that count as an index into the bottom color legend.

    Input structure:
    - Several full grey separator rows split the grid into horizontal panels.
    - The bottom row contains a contiguous left-to-right legend of distinct single-pixel colors.
    - Each panel has one single-color barrier polyline, while the remaining cells are background.

    Rule:
    - For each panel, count how many 4-connected background regions remain after the barrier splits the space.
    - Use that count k as a 1-based index into the legend row from left to right.
    - Output an m x 1 column whose ith cell is the legend color selected by the ith panel's region count.

    Key insight:
    - Barrier shape and barrier color are distractors; each panel contributes only one number, and that number is used as an address into the legend rather than painted directly.
    """
    panels, legend, bg, _sep_color = _extract_structure(grid)
    out = np.zeros((len(panels), 1), dtype=int)
    for r, panel in enumerate(panels):
        k = _panel_region_count(panel, bg)
        assert 1 <= k <= len(legend), "panel region count must index into the legend"
        out[r, 0] = int(legend[k - 1])
    return out


GEN_BG = 0
GEN_SEPARATOR = 5
GEN_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_PANEL_COUNT_RANGE = (2, 4)
GEN_LEGEND_LEN_RANGE = (3, 4)
GEN_PANEL_HEIGHT_RANGE = (4, 6)
GEN_MAX_WIDTH = 16
GEN_ATTEMPTS = 400


def _rand_composition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    remaining = total
    values: list[int] = []
    for i in range(parts):
        slots_left = parts - i - 1
        lo = 1
        hi = remaining - slots_left
        value = random.randint(lo, hi)
        values.append(value)
        remaining -= value
    random.shuffle(values)
    return values


def _barrier_signature(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _make_serpentine_barrier(height: int, width: int, regions: int) -> np.ndarray:
    if regions == 1:
        mask = np.zeros((height, width), dtype=bool)
        row = random.randint(1, height - 2)
        c0 = random.randint(1, width - 3)
        c1 = random.randint(c0 + 1, width - 2)
        mask[row, c0 : c1 + 1] = True
        bend_col = random.choice([c0, c1])
        r0 = random.randint(1, row)
        r1 = random.randint(row, height - 2)
        mask[r0 : r1 + 1, bend_col] = True
        if random.choice([False, True]):
            mask = np.fliplr(mask)
        if random.choice([False, True]):
            mask = np.flipud(mask)
        return mask

    assert regions >= 2
    assert width >= 2 * regions - 1

    region_widths = _rand_composition(width - (regions - 1), regions)
    cuts: list[int] = []
    col = region_widths[0]
    cuts.append(col)
    for idx in range(1, regions - 1):
        col += 1 + region_widths[idx]
        cuts.append(col)

    mask = np.zeros((height, width), dtype=bool)
    for idx, cut in enumerate(cuts):
        mask[:, cut] = True
        if idx + 1 == len(cuts):
            continue
        next_cut = cuts[idx + 1]
        row = height - 1 if idx % 2 == 0 else 0
        c0, c1 = sorted((cut, next_cut))
        mask[row, c0 : c1 + 1] = True

    if random.choice([False, True]):
        mask = np.fliplr(mask)
    if random.choice([False, True]):
        mask = np.flipud(mask)
    return mask


def _sample_counts(panel_count: int, legend_len: int) -> list[int]:
    available = list(range(1, legend_len + 1))
    counts = random.sample(available, min(panel_count, len(available)))
    while len(counts) < panel_count:
        counts.append(random.choice(available))
    if 1 not in counts and random.choice([False, True]):
        counts[random.randrange(panel_count)] = 1
    random.shuffle(counts)
    if panel_count > 1:
        assert len(set(counts)) >= 2
    return counts[:panel_count]


def _sample_dimensions(panel_count: int, legend_len: int) -> tuple[list[int], int] | None:
    heights = [random.randint(*GEN_PANEL_HEIGHT_RANGE) for _ in range(panel_count)]
    total_height = sum(heights) + panel_count + 1
    if total_height > 30:
        return None
    min_width = 2 * legend_len - 1
    width = random.randint(min_width, GEN_MAX_WIDTH)
    return heights, width


def _render_case(
    heights: list[int],
    width: int,
    legend_colors: list[int],
    barrier_colors: list[int],
    counts: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    legend_len = len(legend_colors)
    total_height = sum(heights) + len(heights) + 1
    grid = np.full((total_height, width), GEN_BG, dtype=int)

    row = 0
    seen_shapes: set[tuple[str, ...]] = set()
    for height, barrier_color, region_count in zip(heights, barrier_colors, counts):
        barrier = None
        for _ in range(50):
            candidate = _make_serpentine_barrier(height, width, region_count)
            signature = _barrier_signature(candidate)
            if signature not in seen_shapes:
                barrier = candidate
                seen_shapes.add(signature)
                break
        if barrier is None:
            return None

        panel = grid[row : row + height, :]
        panel[barrier] = int(barrier_color)
        if _panel_region_count(panel, GEN_BG) != region_count:
            return None
        row += height
        grid[row, :] = GEN_SEPARATOR
        row += 1

    legend_row = np.full(width, GEN_BG, dtype=int)
    legend_left = random.randint(0, width - legend_len)
    legend_row[legend_left : legend_left + legend_len] = np.array(legend_colors, dtype=int)
    grid[-1, :] = legend_row

    expected = np.array([[legend_colors[count - 1]] for count in counts], dtype=int)
    return grid, expected


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_ATTEMPTS):
        panel_count = random.randint(*GEN_PANEL_COUNT_RANGE)
        legend_len = random.randint(*GEN_LEGEND_LEN_RANGE)
        dims = _sample_dimensions(panel_count, legend_len)
        if dims is None:
            continue
        heights, width = dims

        needed = panel_count + legend_len
        if needed > len(GEN_COLOR_POOL):
            continue
        chosen = random.sample(GEN_COLOR_POOL, needed)
        barrier_colors = chosen[:panel_count]
        legend_colors = chosen[panel_count:]
        counts = _sample_counts(panel_count, legend_len)

        rendered = _render_case(heights, width, legend_colors, barrier_colors, counts)
        if rendered is None:
            continue
        grid, expected = rendered

        try:
            solved = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(solved, expected):
            continue

        panels, parsed_legend, parsed_bg, parsed_sep = _extract_structure(grid)
        if parsed_bg != GEN_BG or parsed_sep != GEN_SEPARATOR:
            continue
        if parsed_legend != legend_colors:
            continue
        parsed_counts = [_panel_region_count(panel, GEN_BG) for panel in panels]
        if parsed_counts != counts:
            continue
        return grid, expected
    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
