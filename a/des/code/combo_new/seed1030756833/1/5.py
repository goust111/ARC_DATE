from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLORS = tuple(range(1, 10))
COUNT_OPTIONS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2))
GENERATION_RETRIES = 300
PLACEMENT_RETRIES = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            seen[sr, sc] = True
            stack = [(sr, sc)]
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _legend_candidates(grid: np.ndarray) -> list[tuple[np.ndarray, dict[frozenset[int], int]]]:
    h, w = grid.shape
    candidates: list[tuple[np.ndarray, dict[frozenset[int], int]]] = []
    for comp in _components(grid != BACKGROUND):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width != 3 or len(comp) != height * width:
            continue
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == BACKGROUND):
            continue

        mapping: dict[frozenset[int], int] = {}
        ok = True
        for row in sub:
            a, mix, b = map(int, row)
            if a == b:
                ok = False
                break
            key = frozenset((a, b))
            if key in mapping and mapping[key] != mix:
                ok = False
                break
            mapping[key] = mix
        if not ok or not mapping:
            continue

        legend_mask = np.zeros_like(grid, dtype=bool)
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        candidates.append((legend_mask, mapping))
    return candidates


def _draw_lines(
    shape: tuple[int, int],
    lines: list[tuple[int, str, int, int, int, int]],
    mapping: dict[frozenset[int], int],
) -> tuple[np.ndarray, int]:
    out = np.full(shape, BACKGROUND, dtype=int)
    covered: dict[tuple[int, int], list[tuple[int, str]]] = {}

    for color, orient, r0, c0, r1, c1 in lines:
        if orient == "H":
            lo, hi = sorted((c0, c1))
            coords = ((r0, c) for c in range(lo, hi + 1))
        else:
            lo, hi = sorted((r0, r1))
            coords = ((r, c0) for r in range(lo, hi + 1))
        for r, c in coords:
            covered.setdefault((r, c), []).append((color, orient))

    mix_hits = 0
    for (r, c), items in covered.items():
        assert len(items) <= 2, "no cell may be covered by more than two lines"
        if len(items) == 1:
            out[r, c] = items[0][0]
            continue

        (a, oa), (b, ob) = items
        assert a != b, "a crossing must involve two different colors"
        assert {oa, ob} == {"H", "V"}, "only perpendicular lines may mix"
        key = frozenset((a, b))
        assert key in mapping, "every crossing color pair must appear in the legend"
        out[r, c] = mapping[key]
        mix_hits += 1

    return out, mix_hits


def _solve_with_legend(
    grid: np.ndarray, legend_mask: np.ndarray, mapping: dict[frozenset[int], int]
) -> np.ndarray:
    points_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere((grid != BACKGROUND) & ~legend_mask):
        color = int(grid[int(r), int(c)])
        points_by_color.setdefault(color, []).append((int(r), int(c)))

    assert points_by_color, "expected colored endpoint pairs outside the legend"
    lines: list[tuple[int, str, int, int, int, int]] = []
    for color, pts in points_by_color.items():
        assert len(pts) == 2, "each non-legend color must occur as exactly one endpoint pair"
        (r0, c0), (r1, c1) = pts
        if r0 == r1 and c0 != c1:
            orient = "H"
        elif c0 == c1 and r0 != r1:
            orient = "V"
        else:
            raise AssertionError("endpoint pairs must be horizontally or vertically aligned")
        lines.append((color, orient, r0, c0, r1, c1))

    assert {line[1] for line in lines} == {"H", "V"}, "expected both horizontal and vertical pairs"
    out, mix_hits = _draw_lines(grid.shape, lines, mapping)
    assert mix_hits > 0, "expected at least one visible crossing"
    out[legend_mask] = BACKGROUND
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color endpoint pairs with straight lines and mix crossing colors by an edge legend.

    Input structure:
    - The background is black.
    - Outside the legend, every foreground line color appears as exactly two isolated pixels.
    - Each same-color pair is aligned in one row or one column, forming a horizontal or vertical segment when connected.
    - A filled N x 3 legend block touches one grid edge; each row is color A, mixed result, color B.

    Rule:
    - Clear the legend block to background in the output.
    - Draw a same-colored straight segment between each pair of matching endpoints, including the endpoints.
    - Cells covered by one segment take that segment color.
    - Cells where two perpendicular segments of different colors cross use the legend row containing those two colors and take its middle color.

    Key insight:
    - The edge block is a lookup table for unordered color pairs, so crossings are not simple overwrites but legend-controlled mixtures.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    solutions = []
    for legend_mask, mapping in _legend_candidates(g):
        try:
            solutions.append(_solve_with_legend(g, legend_mask, mapping))
        except AssertionError:
            continue
    assert len(solutions) == 1, "expected a unique valid edge legend"
    return solutions[0]


def _sample_spaced(lo: int, hi: int, count: int, min_gap: int = 2) -> list[int] | None:
    values = list(range(lo, hi + 1))
    if len(values) < count:
        return None
    for _ in range(PLACEMENT_RETRIES):
        picked = sorted(random.sample(values, count))
        if all(picked[i + 1] - picked[i] >= min_gap for i in range(len(picked) - 1)):
            return picked
    return None


def _active_area(side: str, height: int, width: int, legend_h: int) -> tuple[int, int, int, int]:
    if side == "top":
        return legend_h + 1, height - 2, 1, width - 2
    if side == "bottom":
        return 1, height - legend_h - 2, 1, width - 2
    if side == "left":
        return 1, height - 2, 4, width - 2
    return 1, height - 2, 1, width - 5


def _place_legend(
    grid: np.ndarray,
    side: str,
    rows: list[tuple[int, int, int]],
) -> np.ndarray:
    legend = np.array(rows, dtype=int)
    lh, lw = legend.shape
    h, w = grid.shape
    if side == "top":
        r0, c0 = 0, random.randint(0, w - lw)
    elif side == "bottom":
        r0, c0 = h - lh, random.randint(0, w - lw)
    elif side == "left":
        r0, c0 = random.randint(0, h - lh), 0
    else:
        r0, c0 = random.randint(0, h - lh), w - lw
    grid[r0 : r0 + lh, c0 : c0 + lw] = legend
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r0 + lh, c0 : c0 + lw] = True
    return mask


def _gen_attempt() -> np.ndarray | None:
    h_count, v_count = random.choice(COUNT_OPTIONS)
    rule_count = h_count * v_count
    needed_colors = h_count + v_count + rule_count
    colors = random.sample(COLORS, needed_colors)
    h_colors = colors[:h_count]
    v_colors = colors[h_count : h_count + v_count]
    mix_colors = colors[h_count + v_count :]

    mapping: dict[frozenset[int], int] = {}
    legend_rows: list[tuple[int, int, int]] = []
    mix_i = 0
    for hc in h_colors:
        for vc in v_colors:
            mix = mix_colors[mix_i]
            mix_i += 1
            mapping[frozenset((hc, vc))] = mix
            a, b = (hc, vc) if random.random() < 0.5 else (vc, hc)
            legend_rows.append((a, mix, b))
    random.shuffle(legend_rows)

    side = random.choice(("top", "bottom", "left", "right"))
    height = random.randint(max(12, rule_count + 8), 30)
    width = random.randint(12, 30)
    grid = np.zeros((height, width), dtype=int)
    legend_mask = _place_legend(grid, side, legend_rows)

    ar0, ar1, ac0, ac1 = _active_area(side, height, width, rule_count)
    if ar1 - ar0 < 4 or ac1 - ac0 < 4:
        return None

    h_rows = _sample_spaced(ar0 + 1, ar1 - 1, h_count)
    v_cols = _sample_spaced(ac0 + 1, ac1 - 1, v_count)
    if h_rows is None or v_cols is None:
        return None

    min_h_row, max_h_row = min(h_rows), max(h_rows)
    min_v_col, max_v_col = min(v_cols), max(v_cols)
    if not (ac0 < min_v_col and max_v_col < ac1 and ar0 < min_h_row and max_h_row < ar1):
        return None

    lines: list[tuple[int, str, int, int, int, int]] = []
    for row, color in zip(h_rows, h_colors):
        left_choices = list(range(ac0, min_v_col))
        right_choices = list(range(max_v_col + 1, ac1 + 1))
        if not left_choices or not right_choices:
            return None
        c0 = random.choice(left_choices)
        c1 = random.choice(right_choices)
        if grid[row, c0] != BACKGROUND or grid[row, c1] != BACKGROUND:
            return None
        grid[row, c0] = color
        grid[row, c1] = color
        lines.append((color, "H", row, c0, row, c1))

    for col, color in zip(v_cols, v_colors):
        top_choices = list(range(ar0, min_h_row))
        bottom_choices = list(range(max_h_row + 1, ar1 + 1))
        if not top_choices or not bottom_choices:
            return None
        r0 = random.choice(top_choices)
        r1 = random.choice(bottom_choices)
        if grid[r0, col] != BACKGROUND or grid[r1, col] != BACKGROUND:
            return None
        grid[r0, col] = color
        grid[r1, col] = color
        lines.append((color, "V", r0, col, r1, col))

    expected, mix_hits = _draw_lines(grid.shape, lines, mapping)
    if mix_hits != rule_count:
        return None
    expected[legend_mask] = BACKGROUND

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_RETRIES} attempts")
