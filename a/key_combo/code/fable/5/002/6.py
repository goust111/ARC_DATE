from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


COLORS = tuple(range(10))
GEN_ATTEMPTS = 500


def _mode(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).reshape(-1)))
    assert counts, "expected nonempty values"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique mode"
    return int(ranked[0][0])


def _border_values(g: np.ndarray) -> np.ndarray:
    top = g[0, :]
    bottom = g[-1, :]
    left = g[1:-1, 0]
    right = g[1:-1, -1]
    return np.concatenate([top, bottom, left, right])


def _background_from_border(g: np.ndarray) -> int:
    border = _border_values(g)
    bg = _mode(border)
    assert int(np.count_nonzero(border == bg)) > int(border.size // 2), "outer border must be mostly background"
    assert int(g[0, 0]) == bg and int(g[0, -1]) == bg
    assert int(g[-1, 0]) == bg and int(g[-1, -1]) == bg
    return int(bg)


def _border_points_by_color(g: np.ndarray, bg: int) -> dict[int, list[tuple[int, int]]]:
    H, W = map(int, g.shape)
    points: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for c in range(1, W - 1):
        if int(g[0, c]) != int(bg):
            points[int(g[0, c])].append((0, c))
        if int(g[H - 1, c]) != int(bg):
            points[int(g[H - 1, c])].append((H - 1, c))
    for r in range(1, H - 1):
        if int(g[r, 0]) != int(bg):
            points[int(g[r, 0])].append((r, 0))
        if int(g[r, W - 1]) != int(bg):
            points[int(g[r, W - 1])].append((r, W - 1))
    return dict(points)


def _paired_lines(g: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    H, W = map(int, g.shape)
    horizontal: set[int] = set()
    vertical: set[int] = set()

    for cells in _border_points_by_color(g, bg).values():
        left_rows = {r for r, c in cells if c == 0 and 0 < r < H - 1}
        right_rows = {r for r, c in cells if c == W - 1 and 0 < r < H - 1}
        top_cols = {c for r, c in cells if r == 0 and 0 < c < W - 1}
        bottom_cols = {c for r, c in cells if r == H - 1 and 0 < c < W - 1}

        horizontal.update(left_rows & right_rows)
        vertical.update(top_cols & bottom_cols)

    assert horizontal or vertical, "expected at least one aligned same-color anchor pair"
    return sorted(horizontal), sorted(vertical)


def _region_spans(length: int, lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 1
    for line in sorted(map(int, lines)):
        assert 0 < line < length - 1, "invisible lines must be inside the border"
        assert start <= line - 1, "line positions must leave nonempty regions"
        spans.append((start, line - 1))
        start = line + 1
    assert start <= length - 2, "line positions must leave a final nonempty region"
    spans.append((start, length - 2))
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress noisy regions by recovering invisible separators from aligned border anchor pairs.

    Input structure:
    - The outer border is mostly background, with non-background anchor pixels
      and distractor pixels on the non-corner border cells.
    - A true separator is indicated only by two same-color border anchors that
      are strictly aligned: left/right anchors on the same row, or top/bottom
      anchors on the same column.
    - The interior cells are multicolor noise; separator rows and columns are
      not visibly drawn inside the grid.

    Rule:
    - Ignore every border pixel that has no same-color aligned partner across
      the opposite border.
    - Each aligned left/right pair defines one invisible horizontal line, and
      each aligned top/bottom pair defines one invisible vertical line.
    - These lines split the non-border interior into rectangular open regions;
      the line cells themselves are excluded.
    - Return the small grid whose cells are the unique dominant colors of the
      corresponding regions.

    Key insight:
    - Both color pairing and geometric alignment are required to recover the
      missing partition, and the number of recovered pairs determines the
      compressed output size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5, "expected a 2D ARC grid"
    H, W = map(int, g.shape)

    bg = _background_from_border(g)
    horizontal, vertical = _paired_lines(g, bg)
    row_spans = _region_spans(H, horizontal)
    col_spans = _region_spans(W, vertical)

    out = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            out[rr, cc] = _mode(g[r0 : r1 + 1, c0 : c1 + 1])
    return out


def _layout_from_region_sizes(sizes: list[int]) -> tuple[int, list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    lines: list[int] = []
    pos = 1
    for i, size in enumerate(sizes):
        start = pos
        end = pos + int(size) - 1
        spans.append((start, end))
        pos = end + 1
        if i + 1 < len(sizes):
            lines.append(pos)
            pos += 1
    return pos + 1, spans, lines


def _region_values(area: int, dominant: int, runner: int, palette: list[int]) -> list[int]:
    dominant_count = int(area // 2 + 1)
    runner_count = int(area - dominant_count)
    values = [int(dominant)] * dominant_count + [int(runner)] * runner_count
    if runner_count >= 4 and len(palette) >= 3 and random.random() < 0.45:
        third = random.choice([c for c in palette if c not in {dominant, runner}])
        idxs = [i for i, v in enumerate(values) if int(v) == int(runner)]
        values[random.choice(idxs)] = int(third)
    random.shuffle(values)
    return values


def _paint_regions(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    bg: int,
) -> tuple[np.ndarray, dict[tuple[int, int], dict[str, int]]]:
    palette = [c for c in COLORS if c != int(bg)]
    expected = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    info: dict[tuple[int, int], dict[str, int]] = {}
    recent_dominants: list[int] = []

    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            avoid = set(recent_dominants[-2:])
            dominant_choices = [c for c in palette if c not in avoid] or palette[:]
            dominant = random.choice(dominant_choices)
            runner = random.choice([c for c in palette if c != dominant])
            coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            values = _region_values(len(coords), dominant, runner, palette)
            for (r, c), value in zip(coords, values):
                grid[r, c] = int(value)
            expected[rr, cc] = int(dominant)
            info[(rr, cc)] = {"dominant": int(dominant), "runner": int(runner)}
            recent_dominants.append(int(dominant))

    return expected, info


def _paint_invisible_lines(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    h_lines: list[int],
    v_lines: list[int],
    region_info: dict[tuple[int, int], dict[str, int]],
    bg: int,
) -> None:
    H, W = map(int, grid.shape)
    palette = [c for c in COLORS if c != int(bg)]

    for r in h_lines:
        for c in range(1, W - 1):
            grid[r, c] = random.choice(palette)
    for c in v_lines:
        for r in range(1, H - 1):
            grid[r, c] = random.choice(palette)

    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            runner = int(region_info[(rr, cc)]["runner"])
            segments: list[tuple[str, int, int, int]] = []
            if rr > 0:
                segments.append(("h", r0 - 1, c0, c1))
            if rr + 1 < len(row_spans):
                segments.append(("h", r1 + 1, c0, c1))
            if cc > 0:
                segments.append(("v", c0 - 1, r0, r1))
            if cc + 1 < len(col_spans):
                segments.append(("v", c1 + 1, r0, r1))
            if not segments:
                continue
            kind, fixed, lo, hi = random.choice(segments)
            if kind == "h":
                grid[fixed, lo : hi + 1] = runner
            else:
                grid[lo : hi + 1, fixed] = runner

    for r in h_lines:
        for c in v_lines:
            grid[r, c] = random.choice(palette)


def _free_border_cells(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    H, W = map(int, grid.shape)
    cells: list[tuple[int, int]] = []
    for c in range(1, W - 1):
        if int(grid[0, c]) == int(bg):
            cells.append((0, c))
        if int(grid[H - 1, c]) == int(bg):
            cells.append((H - 1, c))
    for r in range(1, H - 1):
        if int(grid[r, 0]) == int(bg):
            cells.append((r, 0))
        if int(grid[r, W - 1]) == int(bg):
            cells.append((r, W - 1))
    random.shuffle(cells)
    return cells


def _defines_aligned_pair(a: tuple[int, int], b: tuple[int, int], H: int, W: int) -> bool:
    ar, ac = a
    br, bc = b
    if ac == 0 and bc == W - 1 and ar == br:
        return True
    if ac == W - 1 and bc == 0 and ar == br:
        return True
    if ar == 0 and br == H - 1 and ac == bc:
        return True
    if ar == H - 1 and br == 0 and ac == bc:
        return True
    return False


def _place_unaligned_duplicate(grid: np.ndarray, color: int, bg: int) -> bool:
    H, W = map(int, grid.shape)
    cells = _free_border_cells(grid, bg)
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            if not _defines_aligned_pair(a, b, H, W):
                grid[a] = int(color)
                grid[b] = int(color)
                return True
    return False


def _place_singletons(grid: np.ndarray, colors: list[int], bg: int, count: int) -> bool:
    cells = _free_border_cells(grid, bg)
    if len(cells) < count:
        return False
    for cell, color in zip(cells[:count], colors[:count]):
        grid[cell] = int(color)
    return True


def _place_anchors_and_distractors(
    grid: np.ndarray,
    h_lines: list[int],
    v_lines: list[int],
    bg: int,
) -> bool:
    H, W = map(int, grid.shape)
    true_count = len(h_lines) + len(v_lines)
    palette = [c for c in COLORS if c != int(bg)]
    if true_count + 2 > len(palette):
        return False

    random.shuffle(palette)
    true_colors = palette[:true_count]
    spare = palette[true_count:]
    idx = 0
    for r in h_lines:
        color = true_colors[idx]
        idx += 1
        grid[r, 0] = int(color)
        grid[r, W - 1] = int(color)
    for c in v_lines:
        color = true_colors[idx]
        idx += 1
        grid[0, c] = int(color)
        grid[H - 1, c] = int(color)

    if not spare:
        return False
    duplicate_color = spare[0]
    if not _place_unaligned_duplicate(grid, duplicate_color, bg):
        return False

    singleton_count = random.randint(1, min(3, max(1, len(spare) - 1)))
    singleton_colors = spare[1 : 1 + singleton_count]
    if len(singleton_colors) < singleton_count:
        singleton_colors = [random.choice(spare) for _ in range(singleton_count)]
    return _place_singletons(grid, singleton_colors, bg, singleton_count)


def _has_unaligned_duplicate_distractor(g: np.ndarray, bg: int) -> bool:
    H, W = map(int, g.shape)
    for cells in _border_points_by_color(g, bg).values():
        if len(cells) < 2:
            continue
        aligned = False
        for i, a in enumerate(cells):
            for b in cells[i + 1 :]:
                if _defines_aligned_pair(a, b, H, W):
                    aligned = True
        if not aligned:
            return True
    return False


def _has_singleton_distractor(g: np.ndarray, bg: int) -> bool:
    return any(len(cells) == 1 for cells in _border_points_by_color(g, bg).values())


def _line_exclusion_matters(
    g: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    expected: np.ndarray,
) -> bool:
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            base = g[r0 : r1 + 1, c0 : c1 + 1].reshape(-1)
            segments: list[np.ndarray] = []
            if rr > 0:
                segments.append(g[r0 - 1, c0 : c1 + 1].reshape(-1))
            if rr + 1 < len(row_spans):
                segments.append(g[r1 + 1, c0 : c1 + 1].reshape(-1))
            if cc > 0:
                segments.append(g[r0 : r1 + 1, c0 - 1].reshape(-1))
            if cc + 1 < len(col_spans):
                segments.append(g[r0 : r1 + 1, c1 + 1].reshape(-1))
            for segment in segments:
                values = np.concatenate([base, segment])
                counts = Counter(map(int, values))
                ranked = counts.most_common()
                if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                    return True
                if int(ranked[0][0]) != int(expected[rr, cc]):
                    return True
    return False


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    h_lines: list[int],
    v_lines: list[int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    try:
        detected_bg = _background_from_border(inp)
        detected_h, detected_v = _paired_lines(inp, detected_bg)
        actual = main(inp)
    except Exception:
        return False
    if int(detected_bg) != int(bg):
        return False
    if detected_h != sorted(h_lines) or detected_v != sorted(v_lines):
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(set(map(int, expected.reshape(-1)))) < min(3, expected.size):
        return False
    if not _has_unaligned_duplicate_distractor(inp, bg):
        return False
    if not _has_singleton_distractor(inp, bg):
        return False
    if not _line_exclusion_matters(inp, row_spans, col_spans, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(1, 3)
    m = random.randint(1, 3)
    row_sizes = [random.randint(3, 5) for _ in range(k + 1)]
    col_sizes = [random.randint(3, 5) for _ in range(m + 1)]
    H, row_spans, h_lines = _layout_from_region_sizes(row_sizes)
    W, col_spans, v_lines = _layout_from_region_sizes(col_sizes)
    if max(H, W) > 30:
        return None

    bg = random.choice(COLORS)
    grid = np.full((H, W), int(bg), dtype=int)

    expected, region_info = _paint_regions(grid, row_spans, col_spans, bg)
    _paint_invisible_lines(grid, row_spans, col_spans, h_lines, v_lines, region_info, bg)
    if not _place_anchors_and_distractors(grid, h_lines, v_lines, bg):
        return None

    if not _semantic_checks(grid, expected, bg, h_lines, v_lines, row_spans, col_spans):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid invisible border-pair compression input after {GEN_ATTEMPTS} attempts")
