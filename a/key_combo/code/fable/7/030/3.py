import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import colors_present, get_bg


def _line_cells(shape: tuple[int, int], kind: str, value: int) -> frozenset[tuple[int, int]]:
    h, w = map(int, shape)
    value = int(value)
    if kind == "H":
        return frozenset((value, c) for c in range(w))
    if kind == "V":
        return frozenset((r, value) for r in range(h))
    if kind == "D":
        return frozenset((r, r - value) for r in range(h) if 0 <= r - value < w)
    if kind == "U":
        return frozenset((r, value - r) for r in range(h) if 0 <= value - r < w)
    raise AssertionError(f"unknown line kind: {kind}")


def _infer_lines(grid: np.ndarray, bg: int) -> dict[int, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    lines: dict[int, dict[str, object]] = {}
    used_signatures: set[tuple[str, int]] = set()

    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        assert coords.shape[0] >= 2
        rows = coords[:, 0]
        cols = coords[:, 1]

        candidates: list[tuple[str, int]] = []
        if np.all(rows == rows[0]):
            candidates.append(("H", int(rows[0])))
        if np.all(cols == cols[0]):
            candidates.append(("V", int(cols[0])))
        down = rows - cols
        if np.all(down == down[0]):
            candidates.append(("D", int(down[0])))
        up = rows + cols
        if np.all(up == up[0]):
            candidates.append(("U", int(up[0])))

        assert len(candidates) == 1
        signature = candidates[0]
        assert signature not in used_signatures
        used_signatures.add(signature)

        cells = _line_cells(g.shape, *signature)
        assert set(map(tuple, coords)).issubset(cells)
        lines[int(color)] = {"kind": signature[0], "value": signature[1], "cells": cells}

    assert lines
    return lines


def _covering_colors(
    lines: dict[int, dict[str, object]], r: int, c: int
) -> list[int]:
    point = (int(r), int(c))
    return [color for color, item in lines.items() if point in item["cells"]]


def _below_edges(grid: np.ndarray, lines: dict[int, dict[str, object]], bg: int) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    edges: set[tuple[int, int]] = set()
    saw_three_way = False

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            covering = _covering_colors(lines, r, c)
            visible = int(g[r, c])
            if not covering:
                assert visible == bg
                continue

            assert visible != bg
            assert visible in covering
            if len(covering) >= 3:
                saw_three_way = True
            if len(covering) >= 2:
                for color in covering:
                    if color != visible:
                        edges.add((color, visible))

    assert saw_three_way
    return edges


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    adjacency = {int(color): set() for color in colors}
    indegree = {int(color): 0 for color in colors}

    for below, above in edges:
        below = int(below)
        above = int(above)
        if above not in adjacency[below]:
            adjacency[below].add(above)
            indegree[above] += 1

    available = sorted(color for color in colors if indegree[color] == 0)
    order: list[int] = []
    while available:
        assert len(available) == 1
        color = available.pop()
        order.append(color)
        unlocked = []
        for nxt in sorted(adjacency[color]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                unlocked.append(nxt)
        available.extend(sorted(unlocked))

    assert len(order) == len(colors)
    return order


def _reverse_redraw(
    shape: tuple[int, int],
    lines: dict[int, dict[str, object]],
    order: list[int],
    bg: int,
) -> np.ndarray:
    rank = {color: i for i, color in enumerate(order)}
    out = np.full(shape, int(bg), dtype=int)

    for r in range(shape[0]):
        for c in range(shape[1]):
            covering = _covering_colors(lines, r, c)
            if not covering:
                continue
            out[r, c] = min(covering, key=lambda color: rank[color])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the unique global line stack from visible crossings, then redraw all full lines with the stack reversed.

    Input structure:
    - The background is the dominant color.
    - Each foreground color is the visible part of one complete one-cell-thick straight line: horizontal, vertical, or 45-degree diagonal.
    - Some crossings hide lower lines, and at least one cell is covered by three or more lines.

    Rule:
    - Infer each full line from its visible collinear pixels.
    - At every multi-line cell, the visible color is above every other line passing through that cell; combine all such constraints into the unique bottom-to-top order.
    - Reverse that full order and redraw every full line on the same canvas; therefore each crossed cell shows the color of the original lowest line through that cell.

    Key insight:
    - A local per-crossing swap is insufficient at three-way crossings; the visible colors must first be integrated into one unique global layer order, and only then can the whole stack be reversed consistently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    lines = _infer_lines(g, bg)
    colors = sorted(lines)
    order = _unique_topological_order(colors, _below_edges(g, lines, bg))
    return _reverse_redraw(g.shape, lines, order, bg)


_BG_CHOICES = tuple(range(10))
_MIN_SIDE = 11
_MAX_SIDE = 18
_MIN_VISIBLE_PIXELS = 2
_MIN_LINE_LENGTH = 4
_MAX_ATTEMPTS = 400
_SYMMETRIES = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti",
)


def _apply_symmetry(grid: np.ndarray, symmetry: str) -> np.ndarray:
    if symmetry == "id":
        return grid.copy()
    if symmetry == "rot90":
        return np.rot90(grid, 1).copy()
    if symmetry == "rot180":
        return np.rot90(grid, 2).copy()
    if symmetry == "rot270":
        return np.rot90(grid, 3).copy()
    if symmetry == "flip_lr":
        return np.fliplr(grid).copy()
    if symmetry == "flip_ud":
        return np.flipud(grid).copy()
    if symmetry == "transpose":
        return grid.T.copy()
    if symmetry == "anti":
        return np.fliplr(np.rot90(grid, 1)).copy()
    raise AssertionError(f"unknown symmetry: {symmetry}")


def _paint_lines(
    shape: tuple[int, int],
    line_specs: dict[int, tuple[str, int]],
    order: list[int],
    bg: int,
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for color in order:
        cells = _line_cells(shape, *line_specs[color])
        rows, cols = zip(*sorted(cells))
        grid[np.array(rows), np.array(cols)] = int(color)
    return grid


def _line_length(shape: tuple[int, int], kind: str, value: int) -> int:
    return len(_line_cells(shape, kind, value))


def _candidate_sum_values(
    h: int,
    w: int,
    rt: int,
    ct: int,
    r2: int,
) -> list[int]:
    diff = rt - ct
    candidates = []
    for total in range(h + w - 1):
        if total in {rt + ct, r2 + ct, 2 * r2 - rt + ct}:
            continue
        cab = total - rt
        if not (0 <= cab < w) or cab == ct:
            continue
        if (total + diff) % 2 != 0:
            continue
        rbc = (total + diff) // 2
        cbc = (total - diff) // 2
        if not (0 <= rbc < h and 0 <= cbc < w):
            continue
        if rbc in {rt, r2} or cbc == ct:
            continue
        if _line_length((h, w), "U", total) < _MIN_LINE_LENGTH:
            continue
        candidates.append(int(total))
    return candidates


def _candidate_top_sum_values(
    h: int,
    w: int,
    rt: int,
    ct: int,
    r2: int,
    lower_total: int,
) -> list[int]:
    diff = rt - ct
    candidates = []
    for total in range(h + w - 1):
        if total in {lower_total, rt + ct, r2 + ct, 2 * r2 - rt + ct}:
            continue
        row = total - ct
        if not (0 <= row < h) or row in {rt, r2}:
            continue
        if _line_length((h, w), "U", total) < _MIN_LINE_LENGTH:
            continue
        # Avoid covering the B/C intersection.
        if total == lower_total:
            continue
        # Avoid covering the C/D pair intersection.
        if total == 2 * r2 - diff:
            continue
        candidates.append(int(total))
    return candidates


def _spec_to_cells(
    shape: tuple[int, int], line_specs: dict[int, tuple[str, int]]
) -> dict[int, frozenset[tuple[int, int]]]:
    return {color: _line_cells(shape, *spec) for color, spec in line_specs.items()}


def _visible_counts(grid: np.ndarray, bg: int) -> dict[int, int]:
    return {color: int(np.sum(grid == color)) for color in colors_present(grid, exclude={bg})}


def _make_base_instance() -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    h = random.randint(_MIN_SIDE, _MAX_SIDE)
    w = random.randint(_MIN_SIDE, _MAX_SIDE)
    bg = random.choice(_BG_CHOICES)

    rt = random.randint(2, h - 3)
    ct = random.randint(2, w - 3)
    r2_choices = [r for r in range(2, h - 2) if r != rt]
    random.shuffle(r2_choices)

    for r2 in r2_choices:
        c_cd = r2 - rt + ct
        if not (0 <= c_cd < w):
            continue
        if _line_length((h, w), "D", rt - ct) < _MIN_LINE_LENGTH:
            continue

        lower_sums = _candidate_sum_values(h, w, rt, ct, r2)
        if not lower_sums:
            continue
        random.shuffle(lower_sums)

        for total in lower_sums:
            use_top_u = random.random() < 0.5
            top_sums = _candidate_top_sum_values(h, w, rt, ct, r2, total) if use_top_u else []
            if use_top_u and not top_sums:
                continue

            line_count = 6 if top_sums else 5
            colors = random.sample([c for c in range(10) if c != bg], line_count)

            order = colors[:]
            line_specs: dict[int, tuple[str, int]] = {
                colors[0]: ("H", rt),
                colors[1]: ("U", total),
                colors[2]: ("D", rt - ct),
                colors[3]: ("H", r2),
                colors[4]: ("V", ct),
            }
            if top_sums:
                line_specs[colors[5]] = ("U", random.choice(top_sums))

            input_grid = _paint_lines((h, w), line_specs, order, bg)
            if any(count < _MIN_VISIBLE_PIXELS for count in _visible_counts(input_grid, bg).values()):
                continue

            output_grid = _paint_lines((h, w), line_specs, list(reversed(order)), bg)
            if np.array_equal(input_grid, output_grid):
                continue

            symmetry = random.choice(_SYMMETRIES)
            input_grid = _apply_symmetry(input_grid, symmetry)
            output_grid = _apply_symmetry(output_grid, symmetry)
            return input_grid, output_grid, order

    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        sample = _make_base_instance()
        if sample is None:
            continue

        inp, expected, intended_order = sample
        bg = get_bg(inp)
        try:
            lines = _infer_lines(inp, bg)
            order = _unique_topological_order(sorted(lines), _below_edges(inp, lines, bg))
        except Exception:
            continue

        if order != intended_order:
            continue
        if not np.array_equal(main(inp), expected):
            continue
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue

        visible = _visible_counts(inp, bg)
        if any(count < _MIN_VISIBLE_PIXELS for count in visible.values()):
            continue
        return inp

    raise RuntimeError("failed to generate a valid layered-line reversal input")
