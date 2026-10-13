from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MAX_GEN_ATTEMPTS = 400
REGION_CHOICES = (2, 3, 4)
TEMPLATE_SIZES = ((2, 3), (3, 2), (3, 3))
TEMPLATE_PALETTE = (BLUE, GREEN, YELLOW)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            area = 0
            while q:
                rr, cc = q.popleft()
                area += 1
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            sizes.append(area)
    return sizes


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=sprite.dtype))


def _separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full-height grey separator column"
    return int(cols[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count black regions on the left of a grey divider and use that count as the pixel-upscale factor for the right template.

    Input structure:
    - The grid contains a unique full-height grey column that splits it into a left panel and a right panel.
    - The left panel has black background plus a 1-pixel-wide 4-connected red barrier network.
    - The right panel has a small blue/green/yellow template on black background, with empty padding around it.

    Rule:
    - Find the grey separator column and split the grid into left and right panels.
    - In the left panel, count the number of 4-connected black background regions when red pixels are treated as barriers; call this number N.
    - In the right panel, take the minimum axis-aligned bounding box of all non-black pixels as the template.
    - Output only that template, with every pixel expanded into an N x N solid block of the same color, including black pixels inside the template box.

    Key insight:
    - The left panel encodes an integer through barrier topology, and that integer parameterizes the right panel's upscale; neither panel alone determines the final output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_col(g)
    assert 0 < sep < g.shape[1] - 1, "separator must leave non-empty panels on both sides"

    left = g[:, :sep]
    right = g[:, sep + 1 :]

    scale = _count_components(left == BLACK)
    assert scale >= 1, "left panel must contain at least one black region"

    template_bbox = _bbox(right != BLACK)
    assert template_bbox is not None, "right panel must contain a non-black template"
    r0, c0, r1, c1 = template_bbox
    template = right[r0 : r1 + 1, c0 : c1 + 1]

    return _scale_sprite(template, scale)


def _build_trunk(h: int, left_w: int, rng: random.Random) -> tuple[set[tuple[int, int]], dict[int, int], set[int]]:
    min_col = 2
    max_col = left_w - 3
    segment_count = rng.choices([1, 2, 3], weights=[2, 3, 2], k=1)[0]
    cols = [rng.randint(min_col, max_col)]
    for _ in range(segment_count - 1):
        candidates = [c for c in range(min_col, max_col + 1) if c != cols[-1]]
        cols.append(rng.choice(candidates))

    bend_rows: list[int] = []
    if segment_count > 1:
        available = list(range(2, h - 2))
        bend_rows = sorted(rng.sample(available, segment_count - 1))

    path: set[tuple[int, int]] = set()
    row_to_col: dict[int, int] = {}
    prev_row = 0
    cur_col = cols[0]
    for bend_idx, bend_row in enumerate(bend_rows):
        for r in range(prev_row, bend_row + 1):
            path.add((r, cur_col))
            row_to_col[r] = cur_col
        next_col = cols[bend_idx + 1]
        step = 1 if next_col >= cur_col else -1
        for c in range(cur_col, next_col + step, step):
            path.add((bend_row, c))
        row_to_col[bend_row] = next_col
        cur_col = next_col
        prev_row = bend_row

    for r in range(prev_row, h):
        path.add((r, cur_col))
        row_to_col[r] = cur_col

    return path, row_to_col, set(bend_rows)


def _branch_cells(row: int, col: int, left_w: int, side: str) -> set[tuple[int, int]]:
    if side == "left":
        return {(row, c) for c in range(0, col + 1)}
    return {(row, c) for c in range(col, left_w)}


def _template_masks(size: tuple[int, int]) -> list[np.ndarray]:
    h, w = size
    masks: list[np.ndarray] = []
    if (h, w) == (2, 3):
        masks.extend(
            [
                np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
                np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
                np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
                np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
            ]
        )
    elif (h, w) == (3, 2):
        for mask in _template_masks((2, 3)):
            masks.append(mask.T.copy())
    else:
        masks.extend(
            [
                np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
                np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
                np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
                np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool),
                np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=bool),
            ]
        )
    return masks


def _paint_template(size: tuple[int, int], rng: random.Random) -> np.ndarray:
    mask = rng.choice(_template_masks(size)).copy()
    h, w = size
    template = np.zeros((h, w), dtype=int)
    palette_size = rng.choice([2, 3])
    colors = list(rng.sample(TEMPLATE_PALETTE, palette_size))
    chosen = [rng.choice(colors) for _ in range(int(mask.sum()))]
    if len(set(chosen)) == 1:
        chosen[0] = colors[0]
        chosen[-1] = colors[-1]
    for (r, c), color in zip(np.argwhere(mask), chosen):
        template[int(r), int(c)] = int(color)
    assert len(set(int(v) for v in template[mask])) >= 2
    return template


def _gen_case(rng: random.Random) -> np.ndarray | None:
    target_regions = rng.choice(REGION_CHOICES)
    h = rng.randint(10, 16)
    left_w = rng.randint(7, 11)
    right_w = rng.randint(6, 10)
    if left_w + 1 + right_w > 30:
        return None

    grid = np.zeros((h, left_w + 1 + right_w), dtype=int)
    grid[:, left_w] = GREY

    trunk, row_to_col, bend_rows = _build_trunk(h, left_w, rng)
    barrier = set(trunk)

    branch_count = target_regions - 2
    if branch_count > 0:
        candidate_rows = [r for r in range(1, h - 1) if r not in bend_rows]
        if len(candidate_rows) < branch_count:
            return None
        branch_rows = sorted(rng.sample(candidate_rows, branch_count))
        if branch_count == 1:
            branch_sides = [rng.choice(["left", "right"])]
        else:
            branch_sides = rng.sample(["left", "right"], 2)
        for row, side in zip(branch_rows, branch_sides):
            col = row_to_col[row]
            if side == "left" and col < 2:
                return None
            if side == "right" and col > left_w - 3:
                return None
            barrier |= _branch_cells(row, col, left_w, side)

    for r, c in barrier:
        grid[r, c] = RED

    size = rng.choice(TEMPLATE_SIZES)
    template = _paint_template(size, rng)
    th, tw = template.shape
    max_row0 = min(3, h - th)
    max_col0 = min(3, right_w - tw)
    if max_row0 < 0 or max_col0 < 0:
        return None
    row0 = rng.randint(0, max_row0)
    col0 = rng.randint(0, max_col0)
    right_col0 = left_w + 1 + col0
    grid[row0 : row0 + th, right_col0 : right_col0 + tw] = template

    if not _validate_case(grid):
        return None
    return grid


def _validate_case(grid: np.ndarray) -> bool:
    try:
        sep = _separator_col(grid)
        if np.any(grid[:, :sep] == GREY) or np.any(grid[:, sep + 1 :] == GREY):
            return False

        left = grid[:, :sep]
        right = grid[:, sep + 1 :]
        if not np.all((left == BLACK) | (left == RED)):
            return False

        red_mask = left == RED
        if _count_components(red_mask) != 1:
            return False
        if any(int(block.sum()) == 4 for block in (red_mask[r : r + 2, c : c + 2] for r in range(left.shape[0] - 1) for c in range(left.shape[1] - 1))):
            return False

        region_count = _count_components(left == BLACK)
        if region_count not in REGION_CHOICES:
            return False

        bbox = _bbox(right != BLACK)
        if bbox is None:
            return False
        r0, c0, r1, c1 = bbox
        template = right[r0 : r1 + 1, c0 : c1 + 1]
        if template.shape not in TEMPLATE_SIZES:
            return False
        nz = template[template != BLACK]
        if len(nz) < 3 or len(set(int(v) for v in nz)) < 2:
            return False

        expected = _scale_sprite(template, region_count)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False

        whole_right_scaled = _scale_sprite(right, region_count)
        if actual.shape == whole_right_scaled.shape and np.array_equal(actual, whole_right_scaled):
            return False

        for wrong in REGION_CHOICES:
            if wrong == region_count:
                continue
            wrong_out = _scale_sprite(template, wrong)
            if wrong_out.shape == actual.shape and np.array_equal(wrong_out, actual):
                return False

        return max(actual.shape) <= 30
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random()
        grid = _gen_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
