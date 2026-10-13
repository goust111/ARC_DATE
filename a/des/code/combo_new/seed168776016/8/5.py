from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


def _intervals_from_full_lines(size: int, full_lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for line in sorted(full_lines) + [size]:
        if line - prev > 1:
            spans.append((prev + 1, line - 1))
        prev = line
    return spans


def _parse_with_line_color(grid: np.ndarray, line_color: int) -> dict | None:
    h, w = grid.shape
    line_rows = [r for r in range(h) if np.all(grid[r] == line_color)]
    line_cols = [c for c in range(w) if np.all(grid[:, c] == line_color)]
    if not line_rows and not line_cols:
        return None

    line_mask = np.zeros((h, w), dtype=bool)
    for r in line_rows:
        line_mask[r, :] = True
    for c in line_cols:
        line_mask[:, c] = True
    if np.any((grid == line_color) & ~line_mask):
        return None

    row_spans = _intervals_from_full_lines(h, line_rows)
    col_spans = _intervals_from_full_lines(w, line_cols)
    if not row_spans or not col_spans or len(row_spans) * len(col_spans) < 2:
        return None

    region_bg = np.full((h, w), -1, dtype=int)
    region_colors: list[int] = []
    markers: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if patch.size == 0 or np.any(patch == line_color):
                return None

            values, counts = np.unique(patch, return_counts=True)
            order = np.argsort(-counts)
            if len(order) > 1 and counts[order[0]] == counts[order[1]]:
                return None

            bg_color = int(values[order[0]])
            if bg_color in region_colors:
                return None
            region_colors.append(bg_color)
            region_bg[r0 : r1 + 1, c0 : c1 + 1] = bg_color

            coords = np.argwhere(patch != bg_color)
            for rr, cc in coords:
                markers[int(patch[rr, cc])].append((int(r0 + rr), int(c0 + cc)))

    if np.any(region_bg[~line_mask] < 0):
        return None

    line_row_set = set(line_rows)
    line_col_set = set(line_cols)
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    for color, coords in markers.items():
        if color == line_color or color in region_colors or len(coords) != 2:
            return None

        (r1, c1), (r2, c2) = coords
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if c in line_col_set:
                    continue
                if region_bg[r1, c] < 0 or int(grid[r1, c]) != int(region_bg[r1, c]):
                    return None
        elif c1 == c2:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if r in line_row_set:
                    continue
                if region_bg[r, c1] < 0 or int(grid[r, c1]) != int(region_bg[r, c1]):
                    return None
        else:
            return None

        pairs.append((int(color), (int(r1), int(c1)), (int(r2), int(c2))))

    if not pairs:
        return None

    pairs.sort(key=lambda item: item[0])
    return {
        "line_color": int(line_color),
        "line_rows": line_rows,
        "line_cols": line_cols,
        "line_mask": line_mask,
        "region_bg": region_bg,
        "pairs": pairs,
    }


def _parse_grid(grid: np.ndarray) -> dict:
    candidates = []
    for color in map(int, np.unique(grid)):
        parsed = _parse_with_line_color(grid, color)
        if parsed is not None:
            candidates.append(parsed)

    assert candidates, "expected a partition grid with one full-line divider color"
    candidates.sort(
        key=lambda item: (
            len(item["line_rows"]) + len(item["line_cols"]),
            len(item["pairs"]),
        ),
        reverse=True,
    )
    return candidates[0]


def _apply_rule(grid: np.ndarray, parsed: dict) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    region_bg = parsed["region_bg"]
    line_row_set = set(parsed["line_rows"])
    line_col_set = set(parsed["line_cols"])

    for _color, (r1, c1), (r2, c2) in parsed["pairs"]:
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if c in line_col_set:
                    continue
                out[r1, c] = int(region_bg[r1, c])
        else:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if r in line_row_set:
                    continue
                out[r, c1] = int(region_bg[r, c1])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair with a straight path that adopts each crossed partition's background color.

    Input structure:
    - The grid is partitioned into rectangular regions by one-pixel-thick full grey rows
      and/or columns.
    - Every partition has its own solid background color, and different partitions use
      different background colors.
    - A few marker colors appear exactly twice each as single pixels; each pair is aligned
      on one row or one column and may span multiple partitions.

    Rule:
    - Reconstruct the rectangular partition layout from the full grey divider rows and
      columns.
    - For each same-color marker pair, follow the straight row-wise or column-wise segment
      between its two endpoints.
    - Divider pixels stay grey, marker endpoints stay unchanged, and every non-divider
      segment cell takes the background color of the partition containing that cell.

    Key insight:
    - The line color is determined structurally by the full-width/full-height dividers, and
      the path's output color is not tied to the marker color but to the partition the path
      is currently crossing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    parsed = _parse_grid(g)
    return _apply_rule(g, parsed)


LINE_COLOR = 5
BACKGROUND_POOL = (0, 1, 3, 6, 7)
MARKER_POOL = (2, 4, 8, 9)
LAYOUTS = ("horizontal", "vertical", "both")
SIZE_MIN = 4
SIZE_MAX = 8
PAIR_RETRIES = 200
GEN_RETRIES = 400


def _band_sizes(count: int) -> list[int]:
    return [random.randint(SIZE_MIN, SIZE_MAX) for _ in range(count)]


def _build_partition_canvas(
    row_sizes: list[int],
    col_sizes: list[int],
    region_colors: list[int],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[int, int]], set[int], set[int]]:
    height = sum(row_sizes) + len(row_sizes) - 1
    width = sum(col_sizes) + len(col_sizes) - 1
    grid = np.full((height, width), LINE_COLOR, dtype=int)
    region_bg = np.full((height, width), LINE_COLOR, dtype=int)

    row_spans: list[tuple[int, int]] = []
    row_lines: list[int] = []
    cursor = 0
    for idx, size in enumerate(row_sizes):
        row_spans.append((cursor, cursor + size - 1))
        cursor += size
        if idx < len(row_sizes) - 1:
            row_lines.append(cursor)
            cursor += 1

    col_spans: list[tuple[int, int]] = []
    col_lines: list[int] = []
    cursor = 0
    for idx, size in enumerate(col_sizes):
        col_spans.append((cursor, cursor + size - 1))
        cursor += size
        if idx < len(col_sizes) - 1:
            col_lines.append(cursor)
            cursor += 1

    color_idx = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            color = int(region_colors[color_idx])
            color_idx += 1
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            region_bg[r0 : r1 + 1, c0 : c1 + 1] = color

    return grid, region_bg, row_spans, col_spans, set(row_lines), set(col_lines)


def _choose_layout() -> tuple[str, list[int], list[int]]:
    layout = random.choice(LAYOUTS)
    if layout == "horizontal":
        return layout, _band_sizes(random.randint(2, 3)), _band_sizes(1)
    if layout == "vertical":
        return layout, _band_sizes(1), _band_sizes(random.randint(2, 3))
    return layout, _band_sizes(2), _band_sizes(2)


def _choose_orientations(pair_count: int, row_count: int, col_count: int) -> list[str]:
    if row_count == 1:
        return ["h"] * pair_count
    if col_count == 1:
        return ["v"] * pair_count
    if pair_count == 1:
        return [random.choice(("h", "v"))]

    orientations = ["h", "v"]
    orientations.extend(random.choice(("h", "v")) for _ in range(pair_count - 2))
    random.shuffle(orientations)
    return orientations


def _place_pair(
    orientation: str,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    row_lines: set[int],
    col_lines: set[int],
    reserved: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]] | None:
    for _ in range(PAIR_RETRIES):
        if orientation == "h":
            row_idx = random.randrange(len(row_spans))
            col_idx_a, col_idx_b = sorted(random.sample(range(len(col_spans)), 2))
            r = random.randint(*row_spans[row_idx])
            c1 = random.randint(*col_spans[col_idx_a])
            c2 = random.randint(*col_spans[col_idx_b])
            path = {(r, c) for c in range(min(c1, c2) + 1, max(c1, c2)) if c not in col_lines}
            p1 = (r, c1)
            p2 = (r, c2)
        else:
            col_idx = random.randrange(len(col_spans))
            row_idx_a, row_idx_b = sorted(random.sample(range(len(row_spans)), 2))
            c = random.randint(*col_spans[col_idx])
            r1 = random.randint(*row_spans[row_idx_a])
            r2 = random.randint(*row_spans[row_idx_b])
            path = {(r, c) for r in range(min(r1, r2) + 1, max(r1, r2)) if r not in row_lines}
            p1 = (r1, c)
            p2 = (r2, c)

        used = set(path)
        used.add(p1)
        used.add(p2)
        if len(path) < 1 or used & reserved:
            continue
        return p1, p2, used
    return None


def _render_expected(
    grid: np.ndarray,
    region_bg: np.ndarray,
    row_lines: set[int],
    col_lines: set[int],
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for _color, (r1, c1), (r2, c2) in pairs:
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if c in col_lines:
                    continue
                out[r1, c] = int(region_bg[r1, c])
        else:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if r in row_lines:
                    continue
                out[r, c1] = int(region_bg[r, c1])
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        layout, row_sizes, col_sizes = _choose_layout()
        region_count = len(row_sizes) * len(col_sizes)
        if region_count > len(BACKGROUND_POOL):
            continue

        grid, region_bg, row_spans, col_spans, row_lines, col_lines = _build_partition_canvas(
            row_sizes,
            col_sizes,
            random.sample(BACKGROUND_POOL, region_count),
        )

        pair_count = random.randint(2, len(MARKER_POOL))
        orientations = _choose_orientations(pair_count, len(row_spans), len(col_spans))
        if layout == "both" and pair_count >= 2 and {"h", "v"} - set(orientations):
            continue

        reserved: set[tuple[int, int]] = set()
        pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        ok = True
        for color, orientation in zip(random.sample(MARKER_POOL, pair_count), orientations):
            placed = _place_pair(orientation, row_spans, col_spans, row_lines, col_lines, reserved)
            if placed is None:
                ok = False
                break
            p1, p2, used = placed
            reserved |= used
            pairs.append((int(color), p1, p2))

        if not ok:
            continue

        for color, p1, p2 in pairs:
            grid[p1] = int(color)
            grid[p2] = int(color)

        expected = _render_expected(grid, region_bg, row_lines, col_lines, pairs)
        parsed = _parse_grid(grid)
        if int(parsed["line_color"]) != LINE_COLOR:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise AssertionError("failed to generate a valid puzzle instance")
