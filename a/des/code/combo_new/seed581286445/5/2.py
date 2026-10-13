from __future__ import annotations

import random

import numpy as np


BG = 0
CROSS = 2
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _h(row: int, start: int, end: int) -> dict[str, int | str]:
    assert start < end
    return {"orientation": "h", "fixed": int(row), "start": int(start), "end": int(end)}


def _v(col: int, start: int, end: int) -> dict[str, int | str]:
    assert start < end
    return {"orientation": "v", "fixed": int(col), "start": int(start), "end": int(end)}


def _line_cells(line: dict[str, int | str]) -> list[tuple[int, int]]:
    if line["orientation"] == "h":
        row = int(line["fixed"])
        return [(row, c) for c in range(int(line["start"]), int(line["end"]) + 1)]

    col = int(line["fixed"])
    return [(r, col) for r in range(int(line["start"]), int(line["end"]) + 1)]


def _line_endpoints(line: dict[str, int | str]) -> tuple[tuple[int, int], tuple[int, int]]:
    if line["orientation"] == "h":
        row = int(line["fixed"])
        return (row, int(line["start"])), (row, int(line["end"]))

    col = int(line["fixed"])
    return (int(line["start"]), col), (int(line["end"]), col)


def _cross_cells(lines: list[dict[str, int | str]]) -> set[tuple[int, int]]:
    coverage: dict[tuple[int, int], set[int]] = {}
    for idx, line in enumerate(lines):
        for cell in _line_cells(line):
            coverage.setdefault(cell, set()).add(int(idx))
    return {cell for cell, owners in coverage.items() if len(owners) >= 2}


def _extract_segments(grid: np.ndarray) -> list[dict[str, int | str]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    segments: list[dict[str, int | str]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        (r0, c0), (r1, c1) = coords.tolist()
        if r0 == r1:
            assert c0 != c1, "paired pixels must be distinct"
            segments.append(
                {
                    "orientation": "h",
                    "fixed": int(r0),
                    "start": int(min(c0, c1)),
                    "end": int(max(c0, c1)),
                    "color": int(color),
                }
            )
        else:
            assert c0 == c1, "paired pixels must share a row or a column"
            segments.append(
                {
                    "orientation": "v",
                    "fixed": int(c0),
                    "start": int(min(r0, r1)),
                    "end": int(max(r0, r1)),
                    "color": int(color),
                }
            )

    assert len(segments) >= 2, "expected at least two foreground colors"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored point pair into a straight segment, mark crossings red, then trim crossed segments by their crossing counts.

    Input structure:
    - The background is black.
    - Each foreground color appears exactly twice as isolated single pixels.
    - The two pixels of any one color share exactly one row or one column.
    - Completing those horizontal and vertical pairs may create crossings between different colors.

    Rule:
    - Draw the full straight segment between the two pixels of every foreground color, including both endpoints.
    - After all segments are present, recolor every cell covered by two or more different segments to red.
    - For each original segment, count how many red crossing cells lie on it; if that count is N > 0, delete N cells from each end of the segment.
    - Keep surviving non-crossing cells in the original color, restore deleted non-crossing cells to black, and keep every red crossing mark even if a segment disappears completely.

    Key insight:
    - The shortening step depends on the fully drawn intermediate overlay, so the solver must first discover all crossings and only then crop each segment by its own crossing total.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    segments = _extract_segments(g)
    crosses = _cross_cells(segments)

    out = np.full(g.shape, BG, dtype=int)
    for r, c in crosses:
        out[r, c] = CROSS

    for segment in segments:
        cells = _line_cells(segment)
        n_cross = sum(cell in crosses for cell in cells)
        left = int(n_cross)
        right = int(len(cells) - n_cross)
        if left >= right:
            continue

        for r, c in cells[left:right]:
            if (r, c) not in crosses:
                out[r, c] = int(segment["color"])

    return out


def _analyze_lines(lines: list[dict[str, int | str]]) -> tuple[set[tuple[int, int]], list[int], list[int]]:
    crosses = _cross_cells(lines)
    counts: list[int] = []
    survivors: list[int] = []
    for line in lines:
        cells = _line_cells(line)
        n_cross = sum(cell in crosses for cell in cells)
        counts.append(int(n_cross))
        survivors.append(max(0, int(len(cells) - 2 * n_cross)))
    return crosses, counts, survivors


def _build_no_cross_local() -> tuple[list[dict[str, int | str]], int, int]:
    n_h = random.randint(1, 3)
    n_v = random.randint(1, 3)
    left_w = random.randint(4, 8)
    right_w = random.randint(max(3, 2 * n_v - 1), max(5, 2 * n_v + 3))
    gap_x = random.randint(2, 4)
    gap_y = random.randint(2, 4)
    top_h = 2 * n_h - 1
    bottom_h = random.randint(4, 8)

    lines: list[dict[str, int | str]] = []
    for i in range(n_h):
        row = 2 * i
        start = random.randint(0, left_w - 2)
        end = random.randint(start + 1, left_w - 1)
        lines.append(_h(row, start, end))

    base_col = left_w + gap_x
    for i in range(n_v):
        col = base_col + 2 * i
        r0 = top_h + gap_y + random.randint(0, bottom_h - 2)
        r1 = top_h + gap_y + random.randint((r0 - (top_h + gap_y)) + 1, bottom_h - 1)
        lines.append(_v(col, r0, r1))

    height = top_h + gap_y + bottom_h
    width = left_w + gap_x + right_w
    return lines, height, width


def _build_single_cross_local() -> tuple[list[dict[str, int | str]], int, int]:
    left_gap = random.randint(1, 4)
    right_gap = random.randint(1, 4)
    top_gap = random.randint(1, 4)
    bottom_gap = random.randint(1, 4)

    lines = [
        _h(top_gap, 0, left_gap + right_gap),
        _v(left_gap, 0, top_gap + bottom_gap),
    ]
    height = top_gap + bottom_gap + 1
    width = left_gap + right_gap + 1
    return lines, height, width


def _build_vanish_local() -> tuple[list[dict[str, int | str]], int, int]:
    n_cross = random.choice((2, 3))
    top_gap = random.randint(1, 4)
    bottom_gap = random.randint(1, 4)
    row = top_gap

    lines = [_h(row, 0, n_cross + 1)]
    for col in range(1, n_cross + 1):
        lines.append(_v(col, 0, top_gap + bottom_gap))

    height = top_gap + bottom_gap + 1
    width = n_cross + 2
    return lines, height, width


def _build_dense_local() -> tuple[list[dict[str, int | str]], int, int]:
    extra_left = random.randint(1, 3)
    extra_right = random.randint(1, 3)
    extra_top = random.randint(1, 3)
    extra_bottom = random.randint(1, 3)

    c1 = extra_left + 1
    c2 = c1 + 1
    r1 = extra_top + 1
    r2 = r1 + 1

    lines = [
        _h(r1, c1 - 1, c2 + 1),
        _h(r2, 0, c2 + 1 + extra_right),
        _v(c1, r1 - 1, r2 + 1),
        _v(c2, 0, r2 + 1 + extra_bottom),
    ]
    height = r2 + extra_bottom + 2
    width = c2 + extra_right + 2
    return lines, height, width


def _transpose_lines(
    lines: list[dict[str, int | str]], height: int, width: int
) -> tuple[list[dict[str, int | str]], int, int]:
    out: list[dict[str, int | str]] = []
    for line in lines:
        if line["orientation"] == "h":
            out.append(_v(int(line["fixed"]), int(line["start"]), int(line["end"])))
        else:
            out.append(_h(int(line["fixed"]), int(line["start"]), int(line["end"])))
    return out, width, height


def _flip_left_right(lines: list[dict[str, int | str]], width: int) -> list[dict[str, int | str]]:
    out: list[dict[str, int | str]] = []
    for line in lines:
        if line["orientation"] == "h":
            out.append(_h(int(line["fixed"]), width - 1 - int(line["end"]), width - 1 - int(line["start"])))
        else:
            out.append(_v(width - 1 - int(line["fixed"]), int(line["start"]), int(line["end"])))
    return out


def _flip_top_bottom(lines: list[dict[str, int | str]], height: int) -> list[dict[str, int | str]]:
    out: list[dict[str, int | str]] = []
    for line in lines:
        if line["orientation"] == "h":
            out.append(_h(height - 1 - int(line["fixed"]), int(line["start"]), int(line["end"])))
        else:
            out.append(_v(int(line["fixed"]), height - 1 - int(line["end"]), height - 1 - int(line["start"])))
    return out


def _random_transform(
    lines: list[dict[str, int | str]], height: int, width: int
) -> tuple[list[dict[str, int | str]], int, int]:
    transformed = [dict(line) for line in lines]
    h, w = int(height), int(width)

    if random.random() < 0.5:
        transformed, h, w = _transpose_lines(transformed, h, w)
    if random.random() < 0.5:
        transformed = _flip_left_right(transformed, w)
    if random.random() < 0.5:
        transformed = _flip_top_bottom(transformed, h)

    return transformed, h, w


def _embed_lines(
    lines: list[dict[str, int | str]], height: int, width: int
) -> tuple[list[dict[str, int | str]], int, int]:
    extra_h = random.randint(2, min(10, 30 - height))
    extra_w = random.randint(2, min(10, 30 - width))
    top = random.randint(1, extra_h - 1)
    left = random.randint(1, extra_w - 1)
    bottom = extra_h - top
    right = extra_w - left

    shifted: list[dict[str, int | str]] = []
    for line in lines:
        if line["orientation"] == "h":
            shifted.append(
                _h(
                    int(line["fixed"]) + top,
                    int(line["start"]) + left,
                    int(line["end"]) + left,
                )
            )
        else:
            shifted.append(
                _v(
                    int(line["fixed"]) + left,
                    int(line["start"]) + top,
                    int(line["end"]) + top,
                )
            )

    return shifted, height + top + bottom, width + left + right


def _render_input(lines: list[dict[str, int | str]], height: int, width: int) -> np.ndarray:
    assert len(lines) <= len(INPUT_COLORS)
    colors = random.sample(INPUT_COLORS, len(lines))
    grid = np.full((height, width), BG, dtype=int)

    for color, line in zip(colors, lines):
        p0, p1 = _line_endpoints(line)
        assert p0 != p1
        assert grid[p0] == BG and grid[p1] == BG, "input endpoints must stay disjoint"
        grid[p0] = int(color)
        grid[p1] = int(color)

    return grid


def _check_kind(kind: str, lines: list[dict[str, int | str]]) -> bool:
    crosses, counts, survivors = _analyze_lines(lines)

    if kind == "no_cross":
        return len(crosses) == 0 and all(count == 0 for count in counts)

    if kind == "single_cross":
        return len(lines) == 2 and len(crosses) == 1 and counts == [1, 1] and all(length > 0 for length in survivors)

    if kind == "vanish":
        vanished = [i for i, length in enumerate(survivors) if length == 0]
        return (
            len(crosses) in (2, 3)
            and len(vanished) == 1
            and counts[0] == len(crosses)
            and survivors[0] == 0
            and all(count == 1 for count in counts[1:])
            and all(length > 0 for length in survivors[1:])
        )

    if kind == "dense":
        vanished = [i for i, length in enumerate(survivors) if length == 0]
        survived_crossed = [i for i, length in enumerate(survivors) if length > 0 and counts[i] > 0]
        return len(lines) == 4 and len(crosses) == 4 and len(vanished) == 2 and len(survived_crossed) == 2

    return False


def _generated_grid_is_valid(grid: np.ndarray) -> bool:
    segments = _extract_segments(grid)
    endpoint_cells = set()
    for segment in segments:
        p0, p1 = _line_endpoints(segment)
        if p0 in endpoint_cells or p1 in endpoint_cells:
            return False
        endpoint_cells.add(p0)
        endpoint_cells.add(p1)

    crosses = _cross_cells(segments)
    if any(cell in endpoint_cells for cell in crosses):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    builders = {
        "no_cross": _build_no_cross_local,
        "single_cross": _build_single_cross_local,
        "vanish": _build_vanish_local,
        "dense": _build_dense_local,
    }
    kind = random.choice(tuple(builders))
    lines, height, width = builders[kind]()
    lines, height, width = _random_transform(lines, height, width)
    lines, height, width = _embed_lines(lines, height, width)

    if not _check_kind(kind, lines):
        return None

    grid = _render_input(lines, height, width)
    if not _generated_grid_is_valid(grid):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid random instance")
