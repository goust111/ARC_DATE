from __future__ import annotations

from collections import deque
import random

import numpy as np


ARC_MAX = 30
BG = 0
COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 400
OBJECT_COUNT_RANGE = (2, 5)
SPAN_RANGE = (2, 5)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _extract_ruler_colors(values: np.ndarray, bg: int) -> list[int]:
    return [int(v) for v in values if int(v) != int(bg)]


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == int(bg):
                continue

            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return components


def _offsets(sizes: list[int]) -> list[int]:
    cur = 0
    result: list[int] = []
    for size in sizes:
        result.append(int(cur))
        cur += int(size) + 1
    return result


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False

    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    start_r, start_c = map(int, pts[0])
    q: deque[tuple[int, int]] = deque([(start_r, start_c)])
    seen[start_r, start_c] = True
    visited = 0

    while q:
        r, c = q.popleft()
        visited += 1
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))

    return int(visited) == int(mask.sum())


def _bbox_is_full(mask: np.ndarray) -> bool:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return False
    return bool(
        int(rows.min()) == 0
        and int(rows.max()) == int(mask.shape[0] - 1)
        and int(cols.min()) == 0
        and int(cols.max()) == int(mask.shape[1] - 1)
    )


def _make_irregular_mask(height: int, width: int) -> np.ndarray:
    area = int(height * width)
    min_area = max(int(height + width - 1), int(np.ceil(area * 0.5)))
    max_area = area - 1

    for _ in range(200):
        mask = np.ones((height, width), dtype=bool)
        target = random.randint(int(min_area), int(max_area))
        candidates = [(r, c) for r in range(height) for c in range(width)]

        for _ in range(600):
            if int(mask.sum()) <= int(target):
                break
            r, c = random.choice(candidates)
            if not mask[r, c]:
                continue
            mask[r, c] = False
            if not _bbox_is_full(mask) or not _connected(mask):
                mask[r, c] = True

        if int(mask.sum()) < area and _bbox_is_full(mask) and _connected(mask):
            return mask

    mask = np.ones((height, width), dtype=bool)
    mask[0, 0] = False
    return mask


def _sample_spaced_positions(length: int, count: int) -> list[int] | None:
    if count < 1 or int(length) < int(2 * count - 1):
        return None
    base = sorted(random.sample(range(1, int(length - count + 2)), count))
    return [int(v + i) for i, v in enumerate(base)]


def _place_bboxes(inner_h: int, inner_w: int, masks: dict[int, np.ndarray]) -> dict[int, tuple[int, int]] | None:
    items = sorted(
        masks.items(),
        key=lambda item: (-int(item[1].shape[0] * item[1].shape[1]), -int(item[1].shape[0]), -int(item[1].shape[1])),
    )
    placements: dict[int, tuple[int, int]] = {}
    placed_boxes: list[tuple[int, int, int, int]] = []

    def overlaps_with_margin(r0: int, c0: int, h: int, w: int) -> bool:
        for pr0, pc0, ph, pw in placed_boxes:
            if not (r0 > pr0 + ph or pr0 > r0 + h or c0 > pc0 + pw or pc0 > c0 + w):
                return True
        return False

    def backtrack(i: int) -> bool:
        if i == len(items):
            return True

        color, mask = items[i]
        h, w = map(int, mask.shape)
        positions = [(r, c) for r in range(inner_h - h + 1) for c in range(inner_w - w + 1)]
        random.shuffle(positions)

        for r0, c0 in positions:
            if overlaps_with_margin(int(r0), int(c0), int(h), int(w)):
                continue
            placements[int(color)] = (int(r0), int(c0))
            placed_boxes.append((int(r0), int(c0), int(h), int(w)))
            if backtrack(i + 1):
                return True
            placed_boxes.pop()
            del placements[int(color)]

        return False

    return placements if backtrack(0) else None


def _build_expected_output(
    colors: list[int],
    row_order: list[int],
    col_order: list[int],
    dims_by_color: dict[int, tuple[int, int]],
) -> np.ndarray:
    row_heights = [int(dims_by_color[color][0]) for color in row_order]
    col_widths = [int(dims_by_color[color][1]) for color in col_order]
    out_h = int(sum(row_heights) + len(row_heights) - 1)
    out_w = int(sum(col_widths) + len(col_widths) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)

    row_offsets = _offsets(row_heights)
    col_offsets = _offsets(col_widths)
    row_index = {int(color): i for i, color in enumerate(row_order)}
    col_index = {int(color): i for i, color in enumerate(col_order)}

    for color in colors:
        h, w = dims_by_color[int(color)]
        r0 = row_offsets[row_index[int(color)]]
        c0 = col_offsets[col_index[int(color)]]
        out[r0 : r0 + h, c0 : c0 + w] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder interior irregular regions into solid rectangles using the top and left color rulers.

    Input structure:
    - The dominant background is black.
    - The top row and the left column contain isolated single-pixel ruler marks, with black gaps between marks.
    - Inside the remaining area, each non-background color forms one monochrome 4-connected irregular region.

    Rule:
    - Read the left ruler from top to bottom and the top ruler from left to right to get each color's output row and column index.
    - For every interior region, measure its bounding-box height and width.
    - Build an output grid without the rulers and place a solid rectangle of the same color at that indexed row and column, preserving that measured height and width, with one black separator row and column between adjacent output rows and columns.

    Key insight:
    - The edge marks are coordinate indices rather than decorations, while each irregular region contributes only its color and bounding-box size before being normalized into a solid block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 2 and w >= 2, "grid must include rulers and an interior"

    bg = _mode_color(g)
    row_order = _extract_ruler_colors(g[1:, 0], bg)
    col_order = _extract_ruler_colors(g[0, 1:], bg)
    assert row_order and col_order, "expected non-empty rulers"
    assert len(set(row_order)) == len(row_order), "left ruler colors must be unique"
    assert len(set(col_order)) == len(col_order), "top ruler colors must be unique"

    inner = g[1:, 1:]
    components = _find_monochrome_components(inner, bg)
    assert components, "expected interior regions"

    dims_by_color: dict[int, tuple[int, int]] = {}
    for comp in components:
        color = int(comp["color"])
        assert color not in dims_by_color, "each region color should appear once"
        r0, c0, r1, c1 = comp["bbox"]
        dims_by_color[color] = (int(r1 - r0 + 1), int(c1 - c0 + 1))

    region_colors = set(dims_by_color)
    assert region_colors == set(row_order) == set(col_order), "rulers and regions must use the same colors"

    row_heights = [int(dims_by_color[color][0]) for color in row_order]
    col_widths = [int(dims_by_color[color][1]) for color in col_order]
    row_offsets = _offsets(row_heights)
    col_offsets = _offsets(col_widths)
    out_h = int(sum(row_heights) + len(row_heights) - 1)
    out_w = int(sum(col_widths) + len(col_widths) - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    row_index = {int(color): i for i, color in enumerate(row_order)}
    col_index = {int(color): i for i, color in enumerate(col_order)}
    for color, (rect_h, rect_w) in dims_by_color.items():
        r0 = row_offsets[row_index[int(color)]]
        c0 = col_offsets[col_index[int(color)]]
        out[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(color)

    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(COLORS, count)
    row_order = random.sample(colors, count)
    col_order = random.sample(colors, count)

    dims_by_color: dict[int, tuple[int, int]] = {}
    masks: dict[int, np.ndarray] = {}
    for color in colors:
        height = random.randint(*SPAN_RANGE)
        width = random.randint(*SPAN_RANGE)
        dims_by_color[int(color)] = (int(height), int(width))
        masks[int(color)] = _make_irregular_mask(int(height), int(width))

    expected = _build_expected_output(colors, row_order, col_order, dims_by_color)
    if max(expected.shape) > ARC_MAX:
        return None

    max_h = max(int(mask.shape[0]) for mask in masks.values())
    max_w = max(int(mask.shape[1]) for mask in masks.values())
    min_inner_h = max(int(2 * count - 1), int(max_h + 1), 10)
    min_inner_w = max(int(2 * count - 1), int(max_w + 1), 10)
    if min_inner_h > ARC_MAX - 1 or min_inner_w > ARC_MAX - 1:
        return None

    inner_h = random.randint(int(min_inner_h), ARC_MAX - 1)
    inner_w = random.randint(int(min_inner_w), ARC_MAX - 1)
    placements = _place_bboxes(int(inner_h), int(inner_w), masks)
    if placements is None:
        return None

    row_positions = _sample_spaced_positions(int(inner_h), count)
    col_positions = _sample_spaced_positions(int(inner_w), count)
    if row_positions is None or col_positions is None:
        return None

    grid = np.full((inner_h + 1, inner_w + 1), BG, dtype=int)
    for pos, color in zip(row_positions, row_order):
        grid[int(pos), 0] = int(color)
    for pos, color in zip(col_positions, col_order):
        grid[0, int(pos)] = int(color)

    for color, mask in masks.items():
        r0, c0 = placements[int(color)]
        coords = np.argwhere(mask)
        for dr, dc in coords:
            grid[1 + int(r0) + int(dr), 1 + int(c0) + int(dc)] = int(color)

    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
