import math
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
TEMPLATE_FILL = 4
GREY = 5
DRAW_COLORS = (3, 6, 7, 8, 9)
GEN_RETRIES = 400


def _edge_markers(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    top = [(int(c), int(grid[0, c])) for c in range(1, grid.shape[1]) if int(grid[0, c]) != BG]
    left = [(int(r), int(grid[r, 0])) for r in range(1, grid.shape[0]) if int(grid[r, 0]) != BG]
    return left, top


def _infer_step(row_positions: list[int], col_positions: list[int]) -> int:
    diffs = []
    diffs.extend(abs(b - a) for a, b in zip(row_positions, row_positions[1:]) if b != a)
    diffs.extend(abs(b - a) for a, b in zip(col_positions, col_positions[1:]) if b != a)
    assert diffs, "need at least two row or column markers to infer the cell step"
    step = diffs[0]
    for value in diffs[1:]:
        step = math.gcd(step, value)
    assert step >= 2, "step must leave room for a non-empty interior template"
    return int(step)


def _find_unique_pixel(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert coords.shape == (1, 2), f"expected exactly one pixel of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _template_mask(grid: np.ndarray, marker: tuple[int, int], block: int) -> np.ndarray:
    r, c = marker
    h, w = grid.shape
    assert r + block + 1 < h and c + block + 1 < w, "template box runs out of bounds"

    top = grid[r, c + 1 : c + block + 2]
    left = grid[r + 1 : r + block + 2, c]
    bottom = grid[r + block + 1, c : c + block + 2]
    right = grid[r : r + block + 2, c + block + 1]
    assert np.all(top == GREY)
    assert np.all(left == GREY)
    assert np.all(bottom == GREY)
    assert np.all(right == GREY)

    inner = grid[r + 1 : r + 1 + block, c + 1 : c + 1 + block]
    return inner != BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use edge color legends to stamp one of two template sprites into every row/column coordinate cell.

    Input structure:
    - The background is black.
    - The top row contains equally spaced colored column markers, and the left column contains equally spaced colored row markers, both using the same step size S.
    - Away from the legend area, two small grey-framed template boxes sit side by side; a red corner marker labels the "same-color" template and a blue corner marker labels the "different-color" template.

    Rule:
    - Infer S from the spacing of the edge markers, so each row-marker/column-marker pair defines the adjacent interior (S-1)x(S-1) cell block just below-right of that lattice crossing.
    - Extract the two template masks from the red-labeled and blue-labeled boxes.
    - For every row-marker color r and column-marker color c, stamp the same-color mask when r == c, otherwise stamp the different-color mask; every stamped non-background pixel is recolored to the row color, which equals the column color in the same-color case.
    - Keep the edge markers, grey frames, and template boxes unchanged.

    Key insight:
    - The edge pixels are an invisible coordinate grid, while the red/blue template tags decide which sprite mask each coordinate cell receives based on color equality.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    row_markers, col_markers = _edge_markers(g)
    assert len(row_markers) >= 2 and len(col_markers) >= 2, "need multiple markers on both edges"

    row_positions = [r for r, _color in row_markers]
    col_positions = [c for c, _color in col_markers]
    step = _infer_step(row_positions, col_positions)
    block = step - 1

    same_mask = _template_mask(g, _find_unique_pixel(g, RED), block)
    diff_mask = _template_mask(g, _find_unique_pixel(g, BLUE), block)
    assert same_mask.shape == (block, block) and diff_mask.shape == (block, block)

    for row_pos, row_color in row_markers:
        for col_pos, col_color in col_markers:
            mask = same_mask if row_color == col_color else diff_mask
            rr = slice(row_pos + 1, row_pos + step)
            cc = slice(col_pos + 1, col_pos + step)
            assert out[rr, cc].shape == (block, block), "legend cell must match inferred step"
            out[rr, cc][mask] = int(row_color)

    return out


def _balanced_colors(rng: random.Random, count: int, a: int, b: int) -> list[int]:
    colors = [a] * ((count + 1) // 2) + [b] * (count // 2)
    rng.shuffle(colors)
    return colors


def _grow_blob(size: int, rng: random.Random) -> np.ndarray:
    area = rng.randint(max(2, size), max(2, min(size * size - 1, size * size // 2 + size)))
    cells = {(rng.randrange(size), rng.randrange(size))}
    while len(cells) < area:
        r, c = rng.choice(tuple(cells))
        candidates = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                candidates.append((nr, nc))
        if not candidates:
            continue
        cells.add(rng.choice(candidates))
    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _pattern_library(size: int) -> list[np.ndarray]:
    center = size // 2
    masks = []

    diag = np.eye(size, dtype=bool)
    masks.append(diag)
    masks.append(np.fliplr(diag))

    cross = np.zeros((size, size), dtype=bool)
    cross[center, :] = True
    cross[:, center] = True
    masks.append(cross)

    tee = np.zeros((size, size), dtype=bool)
    tee[0, :] = True
    tee[:, center] = True
    masks.append(tee)

    corner = np.zeros((size, size), dtype=bool)
    corner[:, 0] = True
    corner[-1, :] = True
    masks.append(corner)

    checker = np.fromfunction(lambda r, c: (r + c) % 2 == 0, (size, size), dtype=int).astype(bool)
    masks.append(checker)

    ring = np.zeros((size, size), dtype=bool)
    ring[[0, -1], :] = True
    ring[:, [0, -1]] = True
    if size >= 4:
        masks.append(ring)

    diamond = np.zeros((size, size), dtype=bool)
    radius = max(1, center)
    for r in range(size):
        for c in range(size):
            if abs(r - center) + abs(c - center) <= radius:
                diamond[r, c] = True
    masks.append(diamond)

    return [mask for mask in masks if 1 < int(mask.sum()) < size * size]


def _random_mask(size: int, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.45:
        mask = _grow_blob(size, rng)
    else:
        mask = rng.choice(_pattern_library(size)).copy()
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    turns = rng.randrange(4)
    mask = np.rot90(mask, turns)
    if not (1 < int(mask.sum()) < size * size):
        return _random_mask(size, rng)
    return mask


def _draw_template_box(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    marker_color: int,
) -> None:
    size = mask.shape[0]
    box = size + 2

    grid[top, left : left + box] = GREY
    grid[top + box - 1, left : left + box] = GREY
    grid[top : top + box, left] = GREY
    grid[top : top + box, left + box - 1] = GREY
    grid[top, left] = marker_color
    grid[top + 1 : top + 1 + size, left + 1 : left + 1 + size][mask] = TEMPLATE_FILL


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    step = rng.choice((4, 5, 6))
    block = step - 1
    gap_rows = rng.randint(1, 2)

    max_rows = (29 - gap_rows) // step - 2
    max_cols = 30 // step - 1
    if max_rows < 2 or max_cols < 2:
        return None

    row_count = rng.randint(2, max_rows)
    col_count = rng.randint(2, max_cols)

    color_a, color_b = rng.sample(DRAW_COLORS, 2)
    row_colors = _balanced_colors(rng, row_count, color_a, color_b)
    col_colors = _balanced_colors(rng, col_count, color_a, color_b)
    if len(set(row_colors) & set(col_colors)) == 0 or len(set(row_colors + col_colors)) < 2:
        return None

    same_pairs = sum(int(r == c) for r in row_colors for c in col_colors)
    diff_pairs = row_count * col_count - same_pairs
    if same_pairs == 0 or diff_pairs == 0:
        return None

    same_mask = _random_mask(block, rng)
    diff_mask = _random_mask(block, rng)
    for _ in range(20):
        if not np.array_equal(same_mask, diff_mask):
            break
        diff_mask = _random_mask(block, rng)
    if np.array_equal(same_mask, diff_mask):
        return None

    arena_h = (row_count + 1) * step
    arena_w = (col_count + 1) * step
    pair_w = 2 * (step + 1)
    panel_c_min = max(2, arena_w // 2)
    base_w = max(arena_w, pair_w + panel_c_min)
    if base_w > 30:
        return None
    width = base_w + rng.randint(0, min(30 - base_w, 3))

    panel_r = arena_h + gap_rows
    base_h = panel_r + step + 1
    if base_h > 30:
        return None
    height = base_h + rng.randint(0, min(30 - base_h, 2))

    grid = np.zeros((height, width), dtype=int)

    for idx, color in enumerate(col_colors, start=1):
        grid[0, idx * step] = int(color)
    for idx, color in enumerate(row_colors, start=1):
        grid[idx * step, 0] = int(color)

    panel_c = width - pair_w
    _draw_template_box(grid, panel_r, panel_c, same_mask, RED)
    _draw_template_box(grid, panel_r, panel_c + step + 1, diff_mask, BLUE)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
