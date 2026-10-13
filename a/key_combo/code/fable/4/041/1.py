from __future__ import annotations

from collections import Counter
import random

import numpy as np


LINE_KINDS = ("horizontal", "vertical", "diag_down", "diag_up")


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _line_cells(shape: tuple[int, int], kind: str, value: int) -> frozenset[tuple[int, int]]:
    height, width = map(int, shape)
    if kind == "horizontal":
        return frozenset((value, c) for c in range(width))
    if kind == "vertical":
        return frozenset((r, value) for r in range(height))
    if kind == "diag_down":
        return frozenset((r, r - value) for r in range(height) if 0 <= r - value < width)
    if kind == "diag_up":
        return frozenset((r, value - r) for r in range(height) if 0 <= value - r < width)
    raise AssertionError(f"unknown line kind: {kind}")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = map(int, shape)
    return r in (0, height - 1) or c in (0, width - 1)


def _is_isolated_border_pixel(grid: np.ndarray, bg: int, r: int, c: int) -> bool:
    if int(grid[r, c]) == int(bg) or not _is_border(grid.shape, int(r), int(c)):
        return False
    height, width = map(int, grid.shape)
    for rr, cc in _neighbors4(int(r), int(c)):
        if 0 <= rr < height and 0 <= cc < width and int(grid[rr, cc]) != int(bg):
            return False
    return True


def _candidate_signatures(coords: list[tuple[int, int]]) -> set[tuple[str, int]]:
    signatures: set[tuple[str, int]] = set()
    for r, c in coords:
        signatures.add(("horizontal", int(r)))
        signatures.add(("vertical", int(c)))
        signatures.add(("diag_down", int(r - c)))
        signatures.add(("diag_up", int(r + c)))
    return signatures


def _infer_lines_and_anchor(
    grid: np.ndarray, bg: int
) -> tuple[dict[int, frozenset[tuple[int, int]]], tuple[int, int], int]:
    colors = sorted(int(color) for color in np.unique(grid) if int(color) != int(bg))
    assert colors

    lines: dict[int, frozenset[tuple[int, int]]] = {}
    outliers: list[tuple[int, int, int]] = []
    used_signatures: set[tuple[str, int]] = set()

    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(coords) >= 2

        best: list[tuple[int, int, tuple[str, int], tuple[tuple[int, int], ...]]] = []
        best_score: tuple[int, int] | None = None
        for signature in _candidate_signatures(coords):
            cells = _line_cells(grid.shape, *signature)
            inliers = tuple(coord for coord in coords if coord in cells)
            outlier_coords = tuple(coord for coord in coords if coord not in cells)
            if len(inliers) < 2 or len(outlier_coords) > 1:
                continue
            if any(not _is_isolated_border_pixel(grid, bg, r, c) for r, c in outlier_coords):
                continue
            score = (len(inliers), -len(outlier_coords))
            if best_score is None or score > best_score:
                best_score = score
                best = [(len(inliers), len(outlier_coords), signature, outlier_coords)]
            elif score == best_score:
                best.append((len(inliers), len(outlier_coords), signature, outlier_coords))

        assert best
        assert len(best) == 1
        _ninliers, _noutliers, signature, outlier_coords = best[0]
        assert signature not in used_signatures
        used_signatures.add(signature)

        cells = _line_cells(grid.shape, *signature)
        visible_line_coords = {coord for coord in coords if coord in cells}
        assert len(visible_line_coords) >= 2
        lines[int(color)] = cells

        if outlier_coords:
            r, c = outlier_coords[0]
            outliers.append((r, c, int(color)))

    assert len(outliers) == 1
    anchor_r, anchor_c, anchor_color = outliers[0]
    return lines, (int(anchor_r), int(anchor_c)), int(anchor_color)


def _recover_order(
    grid: np.ndarray, bg: int, lines: dict[int, frozenset[tuple[int, int]]], anchor: tuple[int, int]
) -> list[int]:
    edges: dict[int, set[int]] = {color: set() for color in lines}
    indegree: dict[int, int] = {color: 0 for color in lines}
    anchor_set = {tuple(map(int, anchor))}

    height, width = map(int, grid.shape)
    for r in range(height):
        for c in range(width):
            visible = int(grid[r, c])
            covering = [color for color, cells in lines.items() if (r, c) in cells]

            if (r, c) in anchor_set:
                assert not covering
                continue

            if visible == int(bg):
                assert not covering
                continue

            assert visible in covering
            if len(covering) < 2:
                continue

            for below in covering:
                if below == visible or visible in edges[below]:
                    continue
                edges[below].add(visible)
                indegree[visible] += 1

    order: list[int] = []
    remaining = set(lines)
    while remaining:
        zeroes = sorted(color for color in remaining if indegree[color] == 0)
        assert len(zeroes) == 1
        color = zeroes[0]
        remaining.remove(color)
        order.append(color)
        for higher in edges[color]:
            indegree[higher] -= 1

    assert len(order) == len(lines)
    return order


def _render_lines(
    shape: tuple[int, int],
    lines: list[frozenset[tuple[int, int]]],
    colors: list[int],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for cells, color in zip(lines, colors):
        for r, c in cells:
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rewind overlapped full lines to the moment when the anchor-colored line was topmost.

    Input structure:
    - The background is the most common color.
    - Each foreground color belongs to one complete one-cell-thick horizontal, vertical,
      or 45-degree diagonal line spanning the grid, but higher lines can hide lower ones
      at crossings.
    - Exactly one extra isolated border pixel is an anchor; its color matches one line
      but the anchor itself is not part of any line geometry.

    Rule:
    - Recover every full line from its visible collinear pixels, allowing the anchor-colored
      line to have one isolated border outlier.
    - Use visible crossing colors to recover the unique bottom-to-top layer order of the
      lines.
    - Delete all lines strictly above the line whose color matches the anchor, erase the
      anchor pixel itself, and redraw the remaining complete lines in their original order.

    Key insight:
    - The answer is not the current picture with a few visible pixels erased; you must
      reconstruct full line geometry and layer order first, then re-render the surviving
      layers to reveal pixels that were only hidden by removed lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    lines_by_color, anchor, anchor_color = _infer_lines_and_anchor(g, bg)
    order = _recover_order(g, bg, lines_by_color, anchor)
    cutoff = order.index(int(anchor_color)) + 1
    kept_order = order[:cutoff]

    rendered = _render_lines(
        g.shape,
        [lines_by_color[color] for color in kept_order],
        kept_order,
        bg,
    )
    rendered[int(anchor[0]), int(anchor[1])] = int(bg)
    return rendered


GEN_ATTEMPTS = 500
GEN_MIN_LINES = 3
GEN_MAX_LINES = 5
GEN_MIN_SIDE = 8
GEN_MAX_SIDE = 18
GEN_MIN_RAW_DIAG = 4
GEN_MIN_VISIBLE_PIXELS = 3


def _valid_params(shape: tuple[int, int], kind: str) -> list[int]:
    height, width = map(int, shape)
    if kind == "horizontal":
        return list(range(height))
    if kind == "vertical":
        return list(range(width))
    if kind == "diag_down":
        values = range(-(width - 1), height)
    elif kind == "diag_up":
        values = range(height + width - 1)
    else:
        raise AssertionError(f"unknown line kind: {kind}")
    return [
        int(value)
        for value in values
        if len(_line_cells(shape, kind, int(value))) >= GEN_MIN_RAW_DIAG
    ]


def _lines_cross(
    shape: tuple[int, int], first: tuple[str, int], second: tuple[str, int]
) -> bool:
    if first[0] == second[0]:
        return False
    return bool(_line_cells(shape, *first) & _line_cells(shape, *second))


def _crosses_non_adjacent(
    shape: tuple[int, int], lines: list[tuple[str, int]], candidate: tuple[str, int]
) -> bool:
    for prior in lines[:-1]:
        if _lines_cross(shape, prior, candidate):
            return True
    return False


def _build_line_chain(shape: tuple[int, int], count: int) -> list[tuple[str, int]] | None:
    used_params: dict[str, set[int]] = {}
    first_kind = random.choice(LINE_KINDS)
    first_params = _valid_params(shape, first_kind)
    if not first_params:
        return None

    lines = [(first_kind, random.choice(first_params))]
    used_params[first_kind] = {int(lines[0][1])}

    for _ in range(1, int(count)):
        prev_kind, _prev_value = lines[-1]
        candidate_kinds = [kind for kind in LINE_KINDS if kind != prev_kind]
        random.shuffle(candidate_kinds)

        placed = False
        for avoid_non_adjacent in (True, False):
            for kind in candidate_kinds:
                params = [
                    value
                    for value in _valid_params(shape, kind)
                    if value not in used_params.get(kind, set())
                ]
                random.shuffle(params)
                for value in params:
                    candidate = (kind, int(value))
                    if not _lines_cross(shape, lines[-1], candidate):
                        continue
                    if avoid_non_adjacent and _crosses_non_adjacent(shape, lines, candidate):
                        continue
                    lines.append(candidate)
                    used_params.setdefault(kind, set()).add(int(value))
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    return lines


def _visible_line_counts(grid: np.ndarray, colors: list[int]) -> dict[int, int]:
    return {int(color): int(np.sum(grid == int(color))) for color in colors}


def _anchor_candidates(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            if not _is_border(grid.shape, int(r), int(c)):
                continue
            if int(grid[r, c]) != int(bg):
                continue
            ok = True
            for rr, cc in _neighbors4(int(r), int(c)):
                if 0 <= rr < height and 0 <= cc < width and int(grid[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok:
                candidates.append((int(r), int(c)))
    return candidates


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE)
    width = random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE)
    shape = (height, width)
    count = random.randint(GEN_MIN_LINES, GEN_MAX_LINES)
    lines = _build_line_chain(shape, count)
    if lines is None:
        return None

    palette = list(range(10))
    bg = random.choice(palette)
    colors = random.sample([color for color in palette if color != bg], count)
    line_cells = [_line_cells(shape, *line) for line in lines]
    rendered = _render_lines(shape, line_cells, colors, bg)

    if min(_visible_line_counts(rendered, colors).values()) < GEN_MIN_VISIBLE_PIXELS:
        return None

    anchor_options = _anchor_candidates(rendered, bg)
    if not anchor_options:
        return None

    anchor_index = random.randrange(count)
    anchor_color = int(colors[anchor_index])
    anchor_r, anchor_c = random.choice(anchor_options)

    inp = rendered.copy()
    inp[anchor_r, anchor_c] = anchor_color

    expected = _render_lines(shape, line_cells[: anchor_index + 1], colors[: anchor_index + 1], bg)
    expected[anchor_r, anchor_c] = bg

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
