from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
INPUT_HEIGHT = 3
INPUT_WIDTHS = (4, 5)
FOREGROUND_COLORS = (1, 2, 3)
FOREGROUND_COUNT = 4
GENERATE_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected4(cells: tuple[tuple[int, int], ...]) -> bool:
    remaining = set(cells)
    stack = [next(iter(remaining))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in remaining and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(remaining)


def _crop_binary(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _is_nonsymmetric(cells: tuple[tuple[int, int], ...]) -> bool:
    mask = _crop_binary(cells)
    return not (
        np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _diagonal_groups(cells: tuple[tuple[int, int], ...]) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for cell in sorted(cells):
        groups.setdefault(cell[0] - cell[1], []).append(cell)
    return groups


def _enumerate_shapes() -> dict[int, dict[str, list[tuple[tuple[int, int], ...]]]]:
    shape_bank: dict[int, dict[str, list[tuple[tuple[int, int], ...]]]] = {}
    for width in INPUT_WIDTHS:
        by_kind = {"conflict": [], "plain": []}
        universe = [(r, c) for r in range(INPUT_HEIGHT) for c in range(width)]
        for cells in itertools.combinations(universe, FOREGROUND_COUNT):
            rows_used = {r for r, _ in cells}
            if rows_used != set(range(INPUT_HEIGHT)):
                continue
            if not _is_connected4(cells):
                continue
            if not _is_nonsymmetric(cells):
                continue
            groups = _diagonal_groups(cells)
            kind = "conflict" if any(len(group) > 1 for group in groups.values()) else "plain"
            by_kind[kind].append(tuple(sorted(cells)))
        shape_bank[width] = by_kind
    return shape_bank


SHAPE_BANK = _enumerate_shapes()


def _make_texture(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    texture = np.zeros((2 * height, 2 * width), dtype=int)
    foreground = sorted(
        ((int(r), int(c), int(g[r, c])) for r, c in np.argwhere(g != BACKGROUND)),
        key=lambda item: (item[0], item[1]),
    )

    for r, c, color in foreground:
        rr, cc = int(r), int(c)
        while rr < texture.shape[0] and cc < texture.shape[1]:
            current = int(texture[rr, cc])
            if current == BACKGROUND or current == color:
                texture[rr, cc] = color
            rr += 1
            cc += 1
    return texture


def _stack_texture(texture: np.ndarray) -> np.ndarray:
    unit = np.asarray(texture, dtype=int)
    unit_height, unit_width = map(int, unit.shape)
    step = unit_height - 1
    out = np.zeros((4 * unit_height - 3, unit_width), dtype=int)

    for idx in range(4):
        patch = unit if idx % 2 == 0 else np.flipud(unit)
        r0 = idx * step
        for local_r in range(unit_height):
            row = patch[local_r]
            if not np.any(row != BACKGROUND):
                continue
            dst = out[r0 + local_r]
            non_bg = row != BACKGROUND
            if idx % 2 == 0:
                dst[non_bg] = row[non_bg]
            else:
                write_mask = non_bg & (dst == BACKGROUND)
                dst[write_mask] = row[write_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a sparse motif along down-right diagonals on a doubled canvas, then stack alternating vertical flips.

    Input structure:
    - The input is a small black-background grid with four colored foreground pixels.
    - Foreground colors come from blue, red, and green, and the pixels form one 4-connected asymmetric motif.
    - The motif touches both the top and bottom input rows so the derived diagonal texture has content on its first and last rows.

    Rule:
    - Build an intermediate canvas with doubled height and width, filled with black.
    - For each foreground pixel at (r, c), copy its color to every in-bounds cell (r+k, c+k) on the down-right diagonal.
    - If multiple traces hit the same cell, keep the color from the source pixel with the smaller input row index.
    - Treat that intermediate texture as one wave unit, then stack four copies vertically with one-row overlaps, using the original texture for copies 1 and 3 and its vertical flip for copies 2 and 4.
    - In an overlap row, keep the odd-numbered copy when both copies contribute a non-black pixel.

    Key insight:
    - The answer is not the diagonal stamping alone: that stamped texture becomes a reusable wave tile whose top and bottom border rows are sewn together by alternating vertical flips.
    """
    texture = _make_texture(np.asarray(grid, dtype=int))
    return _stack_texture(texture)


def _assign_colors(
    cells: tuple[tuple[int, int], ...], want_conflict: bool
) -> dict[tuple[int, int], int]:
    groups = _diagonal_groups(cells)
    fixed: dict[tuple[int, int], int] = {}

    if want_conflict:
        repeated = [group for group in groups.values() if len(group) > 1]
        chosen = random.choice(repeated)
        winner = min(chosen)
        hidden = random.choice([cell for cell in chosen if cell != winner])
        winner_color = random.choice(FOREGROUND_COLORS)
        hidden_color = random.choice([c for c in FOREGROUND_COLORS if c != winner_color])
        fixed[winner] = winner_color
        fixed[hidden] = hidden_color

    while True:
        colors = dict(fixed)
        for cell in cells:
            if cell not in colors:
                colors[cell] = random.choice(FOREGROUND_COLORS)
        if len(set(colors.values())) >= 2:
            return colors


def _build_input(width: int, kind: str) -> np.ndarray:
    cells = random.choice(SHAPE_BANK[width][kind])
    color_map = _assign_colors(cells, want_conflict=(kind == "conflict"))
    grid = np.zeros((INPUT_HEIGHT, width), dtype=int)
    for r, c in cells:
        grid[r, c] = int(color_map[(r, c)])
    return grid


def _validate_generated_case(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.shape[0] == INPUT_HEIGHT
    assert g.shape[1] in INPUT_WIDTHS
    assert g.ndim == 2

    foreground_cells = [tuple(map(int, rc)) for rc in np.argwhere(g != BACKGROUND)]
    assert len(foreground_cells) == FOREGROUND_COUNT
    assert set(map(int, g[g != BACKGROUND])).issubset(set(FOREGROUND_COLORS))
    assert _is_connected4(tuple(foreground_cells))
    assert _is_nonsymmetric(tuple(foreground_cells))
    assert np.any(g[0] != BACKGROUND)
    assert np.any(g[-1] != BACKGROUND)

    texture = _make_texture(g)
    out = main(g)
    assert texture.shape == (2 * g.shape[0], 2 * g.shape[1])
    assert out.shape == (4 * texture.shape[0] - 3, texture.shape[1])
    assert texture[0].any() and texture[-1].any()
    assert np.array_equal(out[: texture.shape[0]], texture)

    diag_groups = _diagonal_groups(tuple(sorted(foreground_cells)))
    repeated = [group for group in diag_groups.values() if len(group) > 1]
    if repeated:
        assert any(len({int(g[r, c]) for r, c in group}) > 1 for group in repeated)
        for group in repeated:
            winner = min(group)
            winner_color = int(g[winner])
            for r, c in group:
                rr, cc = int(r), int(c)
                while rr < texture.shape[0] and cc < texture.shape[1]:
                    assert int(texture[rr, cc]) == winner_color
                    rr += 1
                    cc += 1
    else:
        for r, c in foreground_cells:
            assert int(texture[r, c]) == int(g[r, c])

    second_start = texture.shape[0] - 1
    assert np.array_equal(out[second_start], texture[-1])
    third_start = 2 * (texture.shape[0] - 1)
    assert np.array_equal(out[third_start], texture[0])


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        width = random.choice(INPUT_WIDTHS)
        kind = random.choice(["conflict", "plain"])
        grid = _build_input(width, kind)
        _validate_generated_case(grid)
        return grid
    raise RuntimeError("failed to generate a valid ARC input")
