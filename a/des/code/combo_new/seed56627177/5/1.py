from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
YELLOW = 4
GRAY = 5
MAROON = 9
LEGEND_COLORS = [1, 2, 3, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if seen[rr, cc] or not bool(mask[rr, cc]):
                        continue
                    seen[rr, cc] = True
                    q.append((rr, cc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_separator_column(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == GRAY))]
    assert len(candidates) == 1, "expected exactly one full-height gray separator column"
    sep = int(candidates[0])
    assert 0 < sep < grid.shape[1] - 1, "separator must split legend and operation zones"
    return sep


def _legend_mapping(left: np.ndarray) -> dict[int, int]:
    special = {BG, YELLOW, GRAY, MAROON}
    mask = np.isin(left, list(special), invert=True)
    comps = _components(mask)
    assert comps, "expected at least one legend color block"

    mapping: dict[int, int] = {}
    for cells in comps:
        r0, c0, r1, c1 = _bbox(cells)
        color = int(left[cells[0]])
        block = left[r0 : r1 + 1, c0 : c1 + 1]
        assert block.shape[0] == block.shape[1], "legend block must be square"
        assert bool(np.all(block == color)), "legend block must be a solid square"
        assert r1 + 1 < left.shape[0], "legend block must have a label row below it"
        count = int(np.sum(left[r1 + 1, c0 : c1 + 1] == YELLOW))
        assert count > 0, "legend label must contain at least one yellow dot"
        assert count not in mapping, "legend labels must be unique"
        mapping[count] = color

    return mapping


def _is_maroon(op: np.ndarray, r: int, c: int) -> bool:
    return 0 <= r < op.shape[0] and 0 <= c < op.shape[1] and int(op[r, c]) == MAROON


def _frame_specs(op: np.ndarray, allowed_counts: set[int]) -> list[tuple[int, int, int, int, int]]:
    top_lefts: set[tuple[int, int]] = set()
    top_rights: set[tuple[int, int]] = set()
    bottom_lefts: set[tuple[int, int]] = set()
    bottom_rights: set[tuple[int, int]] = set()

    for r in range(op.shape[0]):
        for c in range(op.shape[1]):
            if not _is_maroon(op, r, c):
                continue
            if _is_maroon(op, r, c + 1) and _is_maroon(op, r + 1, c) and not _is_maroon(op, r - 1, c) and not _is_maroon(op, r, c - 1):
                top_lefts.add((r, c))
            if _is_maroon(op, r, c - 1) and _is_maroon(op, r + 1, c) and not _is_maroon(op, r - 1, c) and not _is_maroon(op, r, c + 1):
                top_rights.add((r, c))
            if _is_maroon(op, r, c + 1) and _is_maroon(op, r - 1, c) and not _is_maroon(op, r + 1, c) and not _is_maroon(op, r, c - 1):
                bottom_lefts.add((r, c))
            if _is_maroon(op, r, c - 1) and _is_maroon(op, r - 1, c) and not _is_maroon(op, r + 1, c) and not _is_maroon(op, r, c + 1):
                bottom_rights.add((r, c))

    specs: set[tuple[int, int, int, int, int]] = set()
    for r0, c0 in top_lefts:
        for r1, c1 in bottom_rights:
            if r1 - r0 < 4 or c1 - c0 < 4:
                continue
            if (r0, c1) not in top_rights or (r1, c0) not in bottom_lefts:
                continue
            patch = op[r0 : r1 + 1, c0 : c1 + 1]
            height, width = patch.shape
            border = _frame_border_mask(height, width)
            if bool(np.any((patch == MAROON) & (~border))):
                continue
            if not bool(np.all(patch[1:-1, 1:-1] == BG)):
                continue
            hole_count = int(border.sum() - np.sum(patch == MAROON))
            if hole_count not in allowed_counts:
                continue
            specs.add((r0, c0, r1, c1, hole_count))

    assert specs, "expected at least one maroon frame"
    return sorted(specs)


def main(grid: np.ndarray) -> np.ndarray:
    """Select a legend color by hole count and fill each hollow frame's interior.

    Input structure:
    - The background is black.
    - A single full-height gray column separates a left legend area from a right operation area.
    - In the legend area, each entry is a solid colored square with one row of yellow dots directly below it.
    - In the operation area, each object is a maroon hollow rectangle whose 1-pixel border contains one or more 1-pixel holes.

    Rule:
    - Read each legend entry by counting the yellow dots below the colored square, producing a hole-count to color mapping.
    - Find every maroon hollow rectangle and count how many border pixels are missing from its rectangular outline.
    - Use that hole count to look up the matching legend color.
    - Repaint only the black interior of the rectangle with the matched color, keeping the legend, separator, and maroon border unchanged.

    Key insight:
    - The missing border pixels are not damage to repair; their count is the lookup key that selects the interior fill color from the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep = _find_separator_column(g)
    mapping = _legend_mapping(g[:, :sep])

    out = g.copy()
    op_in = g[:, sep + 1 :]
    op_out = out[:, sep + 1 :]
    for r0, c0, r1, c1, hole_count in _frame_specs(op_in, set(mapping)):
        assert hole_count in mapping, "every frame hole count must appear in the legend"
        fill_color = int(mapping[hole_count])
        interior = op_out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BG] = fill_color

    return out


def _layout_boxes(
    sizes: list[tuple[int, int]],
    cols: int,
    row_gap: int,
    col_gap: int,
) -> tuple[list[tuple[int, int]], int, int]:
    count = len(sizes)
    rows = (count + cols - 1) // cols
    row_heights = [0] * rows
    col_widths = [0] * cols
    for idx, (height, width) in enumerate(sizes):
        row = idx // cols
        col = idx % cols
        row_heights[row] = max(row_heights[row], int(height))
        col_widths[col] = max(col_widths[col], int(width))

    row_starts = []
    acc = 0
    for row in range(rows):
        row_starts.append(acc)
        acc += row_heights[row]
        if row != rows - 1:
            acc += row_gap
    total_height = acc

    col_starts = []
    acc = 0
    for col in range(cols):
        col_starts.append(acc)
        acc += col_widths[col]
        if col != cols - 1:
            acc += col_gap
    total_width = acc

    origins: list[tuple[int, int]] = []
    for idx, (height, width) in enumerate(sizes):
        row = idx // cols
        col = idx % cols
        slack_r = row_heights[row] - height
        slack_c = col_widths[col] - width
        origins.append(
            (
                row_starts[row] + random.randint(0, slack_r),
                col_starts[col] + random.randint(0, slack_c),
            )
        )

    return origins, total_height, total_width


def _choose_holes(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates: list[tuple[int, int]] = []
    for c in range(2, width - 2):
        candidates.append((0, c))
        candidates.append((height - 1, c))
    for r in range(2, height - 2):
        candidates.append((r, 0))
        candidates.append((r, width - 1))
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []

    def dfs(start: int) -> bool:
        if len(chosen) == count:
            return True
        need = count - len(chosen)
        if len(candidates) - start < need:
            return False
        for idx in range(start, len(candidates)):
            cell = candidates[idx]
            if any(abs(cell[0] - r) + abs(cell[1] - c) == 1 for r, c in chosen):
                continue
            chosen.append(cell)
            if dfs(idx + 1):
                return True
            chosen.pop()
        return False

    if dfs(0):
        return chosen.copy()
    return None


def _frame_with_holes(hole_count: int) -> tuple[int, int, list[tuple[int, int]]]:
    for _ in range(100):
        height = random.randint(5, 8)
        width = random.randint(5, 8)
        holes = _choose_holes(height, width, hole_count)
        if holes is not None:
            return height, width, holes
    raise RuntimeError("failed to build a frame with the requested hole count")


def _paint_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    holes: list[tuple[int, int]],
) -> None:
    grid[top, left : left + width] = MAROON
    grid[top + height - 1, left : left + width] = MAROON
    grid[top : top + height, left] = MAROON
    grid[top : top + height, left + width - 1] = MAROON
    for r, c in holes:
        grid[top + r, left + c] = BG


def _paint_legend_entry(
    grid: np.ndarray,
    top: int,
    left: int,
    size: int,
    color: int,
    count: int,
) -> None:
    grid[top : top + size, left : left + size] = color
    cols = random.sample(range(size), count)
    for offset in cols:
        grid[top + size, left + offset] = YELLOW


def _expected_output(
    grid: np.ndarray,
    sep: int,
    frames: list[dict[str, int]],
    mapping: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    op = out[:, sep + 1 :]
    for frame in frames:
        r0 = int(frame["top"])
        c0 = int(frame["left"])
        r1 = r0 + int(frame["height"]) - 1
        c1 = c0 + int(frame["width"]) - 1
        color = int(mapping[int(frame["count"])])
        interior = op[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BG] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    entry_count = random.randint(3, 5)
    legend_counts = random.sample([1, 2, 3, 4, 5], entry_count)
    legend_colors = random.sample(LEGEND_COLORS, entry_count)
    count_to_color = {count: color for count, color in zip(legend_counts, legend_colors)}

    square_size = random.randint(max(legend_counts), min(max(legend_counts) + 1, 6))
    if entry_count >= 5:
        legend_cols = 2
    else:
        legend_cols = random.choice([1, 2])
        if legend_cols > entry_count:
            legend_cols = 1
    legend_row_gap = random.randint(1, 2)
    legend_col_gap = random.randint(2, 3)
    legend_sizes = [(square_size + 1, square_size) for _ in range(entry_count)]
    legend_origins, legend_height, legend_width = _layout_boxes(
        legend_sizes,
        legend_cols,
        legend_row_gap,
        legend_col_gap,
    )

    frame_count = entry_count
    frame_counts = legend_counts.copy()
    random.shuffle(frame_counts)

    frame_specs = []
    for hole_count in frame_counts:
        height, width, holes = _frame_with_holes(hole_count)
        frame_specs.append({"count": hole_count, "height": height, "width": width, "holes": holes})

    if frame_count >= 4:
        op_cols = 2
    else:
        op_cols = random.choice([1, 2])
    if op_cols > frame_count:
        op_cols = 1
    op_row_gap = random.randint(1, 2)
    op_col_gap = random.randint(2, 3)
    op_sizes = [(int(spec["height"]), int(spec["width"])) for spec in frame_specs]
    op_origins, op_height, op_width = _layout_boxes(op_sizes, op_cols, op_row_gap, op_col_gap)

    left_margin = random.randint(1, 2)
    gap_before_sep = random.randint(1, 2)
    gap_after_sep = random.randint(1, 2)
    right_margin = random.randint(1, 2)

    width = left_margin + legend_width + gap_before_sep + 1 + gap_after_sep + op_width + right_margin
    height = max(legend_height, op_height) + random.randint(0, 3)
    if height > 30 or width > 30:
        return None

    legend_top = random.randint(0, height - legend_height)
    op_top = random.randint(0, height - op_height)
    legend_left = left_margin
    sep = legend_left + legend_width + gap_before_sep
    op_left = sep + 1 + gap_after_sep

    grid = np.full((height, width), BG, dtype=int)
    grid[:, sep] = GRAY

    legend_items = list(zip(legend_counts, legend_colors))
    random.shuffle(legend_items)
    for (count, color), (dr, dc) in zip(legend_items, legend_origins):
        _paint_legend_entry(grid, legend_top + dr, legend_left + dc, square_size, color, count)

    frames_for_expected: list[dict[str, int]] = []
    for spec, (dr, dc) in zip(frame_specs, op_origins):
        top = op_top + dr
        left = gap_after_sep + dc
        _paint_frame(grid, top, sep + 1 + left, int(spec["height"]), int(spec["width"]), spec["holes"])
        frames_for_expected.append(
            {
                "top": top,
                "left": left,
                "height": int(spec["height"]),
                "width": int(spec["width"]),
                "count": int(spec["count"]),
            }
        )

    expected = _expected_output(grid, sep, frames_for_expected, count_to_color)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
