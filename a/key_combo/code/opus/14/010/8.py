from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
CONTENT_COLORS = (1, 2, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_green_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    green_mask = grid == GREEN
    assert np.any(green_mask), "expected a green frame"
    r0, c0, r1, c1 = _bbox(green_mask)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h >= 5 and w >= 6, "frame must have a non-trivial interior"

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    assert np.all(patch[border] == GREEN), "expected a 1-cell-thick green rectangular border"
    assert not np.any(patch[~border] == GREEN), "green must only appear on the frame border"
    return r0, c0, r1, c1


def _expand_uniform_rows(interior: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for row in interior:
        rows.append(row.copy())
        if np.all(row == row[0]):
            rows.append(row.copy())
    return np.stack(rows, axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect uniform rows inside a green frame and duplicate only those rows before rebuilding the frame.

    Input structure:
    - The grid has a black background and one green 1-cell-thick rectangular frame.
    - The frame interior is completely filled by a non-black, non-green color pattern.
    - Some interior rows are uniform single-color rows, while others contain multiple colors.

    Rule:
    - Locate the green frame and extract its interior rectangle.
    - Inspect the interior row by row: duplicate each uniform row once vertically, and keep each non-uniform row once.
    - Return only the rebuilt framed pattern: a new green border wrapped around the expanded interior.

    Key insight:
    - The output height is not fixed by the input frame alone; it depends on first classifying every interior row as uniform or non-uniform.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _find_green_frame(g)
    framed = g[r0 : r1 + 1, c0 : c1 + 1]
    interior = framed[1:-1, 1:-1]
    assert interior.size > 0, "expected a non-empty frame interior"

    expanded = _expand_uniform_rows(interior)
    out = np.full((expanded.shape[0] + 2, expanded.shape[1] + 2), GREEN, dtype=int)
    out[1:-1, 1:-1] = expanded
    return out


def _make_uniform_row(width: int, palette: list[int]) -> np.ndarray:
    return np.full(width, random.choice(palette), dtype=int)


def _make_nonuniform_row(width: int, palette: list[int]) -> np.ndarray:
    if width == 4:
        a, b = random.sample(palette, 2)
        split = random.randint(1, width - 1)
        row = np.array([a] * split + [b] * (width - split), dtype=int)
    else:
        mode = random.choice(("two_blocks", "three_blocks", "alternating"))
        if mode == "two_blocks":
            a, b = random.sample(palette, 2)
            split = random.randint(1, width - 1)
            row = np.array([a] * split + [b] * (width - split), dtype=int)
        elif mode == "three_blocks":
            a, b, c = random.sample(palette, 3)
            cut1 = random.randint(1, width - 2)
            cut2 = random.randint(cut1 + 1, width - 1)
            row = np.array([a] * cut1 + [b] * (cut2 - cut1) + [c] * (width - cut2), dtype=int)
        else:
            a, b = random.sample(palette, 2)
            row = np.array([a if i % 2 == 0 else b for i in range(width)], dtype=int)

    assert np.unique(row).size >= 2, "non-uniform row must contain multiple colors"
    return row


def _sample_interior(inner_h: int, inner_w: int) -> np.ndarray:
    palette_size = random.randint(3, min(5, len(CONTENT_COLORS)))
    palette = random.sample(list(CONTENT_COLORS), palette_size)

    uniform_count = random.randint(1, inner_h - 1)
    uniform_rows = set(random.sample(range(inner_h), uniform_count))

    rows: list[np.ndarray] = []
    for r in range(inner_h):
        if r in uniform_rows:
            rows.append(_make_uniform_row(inner_w, palette))
        else:
            rows.append(_make_nonuniform_row(inner_w, palette))

    interior = np.stack(rows, axis=0)
    assert not np.any(interior == BLACK)
    assert not np.any(interior == GREEN)
    assert any(np.all(row == row[0]) for row in interior)
    assert any(np.unique(row).size >= 2 for row in interior)
    return interior


def _build_input(interior: np.ndarray) -> np.ndarray:
    inner_h, inner_w = interior.shape
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    extra_h = random.randint(0, min(8, 30 - frame_h))
    extra_w = random.randint(0, min(8, 30 - frame_w))
    grid_h = frame_h + extra_h
    grid_w = frame_w + extra_w

    top = random.randint(0, grid_h - frame_h)
    left = random.randint(0, grid_w - frame_w)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[top : top + frame_h, left : left + frame_w] = GREEN
    grid[top + 1 : top + frame_h - 1, left + 1 : left + frame_w - 1] = interior
    return grid


def _expected_output(interior: np.ndarray) -> np.ndarray:
    expanded = _expand_uniform_rows(interior)
    out = np.full((expanded.shape[0] + 2, expanded.shape[1] + 2), GREEN, dtype=int)
    out[1:-1, 1:-1] = expanded
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inner_h = random.randint(3, 12)
        inner_w = random.randint(4, 14)

        interior = _sample_interior(inner_h, inner_w)
        inp = _build_input(interior)
        expected = _expected_output(interior)

        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        if np.array_equal(main(inp), expected):
            return inp

    raise RuntimeError("failed to generate a valid input for puzzle 8")
