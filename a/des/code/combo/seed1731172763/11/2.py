from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
BROWN = 9

BORDER_COLORS = [RED, BLUE, GREEN, YELLOW, MAGENTA, ORANGE, CYAN, BROWN]
PATTERN_COLORS = [MAGENTA, ORANGE, CYAN, BROWN]
GENERATION_RETRIES = 200


def _non_separator_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_separator in enumerate(mask.tolist()):
        if not is_separator and start is None:
            start = idx
        elif is_separator and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def _infer_separator_color(grid: np.ndarray) -> int:
    row_colors = [int(grid[r, 0]) for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0])]
    col_colors = [int(grid[0, c]) for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]
    common = set(row_colors) & set(col_colors)
    assert common, "expected at least one separator color spanning full rows and columns"

    row_count = Counter(row_colors)
    col_count = Counter(col_colors)
    ordered_candidates: list[int] = []
    seen: set[int] = set()
    for color in row_colors + col_colors:
        if color in common and color not in seen:
            ordered_candidates.append(color)
            seen.add(color)
    return max(ordered_candidates, key=lambda color: row_count[color] + col_count[color])


def _split_panels(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    separator_color = _infer_separator_color(grid)
    sep_rows = np.all(grid == separator_color, axis=1)
    sep_cols = np.all(grid == separator_color, axis=0)
    row_segments = _non_separator_segments(sep_rows)
    col_segments = _non_separator_segments(sep_cols)

    assert row_segments and col_segments, "expected at least one panel"
    row_heights = {r1 - r0 for r0, r1 in row_segments}
    col_widths = {c1 - c0 for c0, c1 in col_segments}
    assert len(row_heights) == 1 and len(col_widths) == 1, "panels must be equal-sized"
    assert next(iter(row_heights)) >= 3 and next(iter(col_widths)) >= 3, "panels must support a 1-cell border"

    return separator_color, row_segments, col_segments


def _group_panels(grid: np.ndarray) -> list[dict[str, object]]:
    _separator_color, row_segments, col_segments = _split_panels(grid)
    groups: dict[bytes, dict[str, object]] = {}
    panel_index = 0

    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            panel = grid[r0:r1, c0:c1]
            signature = panel.tobytes()
            if signature not in groups:
                groups[signature] = {"panels": [], "first_index": panel_index}
            groups[signature]["panels"].append((r0, r1, c0, c1))
            panel_index += 1

    return sorted(groups.values(), key=lambda item: (-len(item["panels"]), item["first_index"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group equal panels in a grey-separated panel grid, then color each group's
    panel border with a rank color based on group size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    groups = _group_panels(g)
    assert len(groups) >= 2, "expected at least two distinct panel groups"
    assert len(groups) <= len(BORDER_COLORS), "not enough border colors for all groups"

    out = g.copy()
    for border_color, group in zip(BORDER_COLORS, groups):
        for r0, r1, c0, c1 in group["panels"]:
            out[r0, c0:c1] = border_color
            out[r1 - 1, c0:c1] = border_color
            out[r0:r1, c0] = border_color
            out[r0:r1, c1 - 1] = border_color

    return out


def _random_blob_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    max_area = min(height * width - 1, max(3, (height * width * 2) // 3))
    target = rng.randint(2, max_area)
    cells = {(rng.randrange(height), rng.randrange(width))}

    while len(cells) < target:
        base_r, base_c = rng.choice(tuple(cells))
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = base_r + dr
            cc = base_c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                neighbors.append((rr, cc))
        if neighbors:
            cells.add(rng.choice(neighbors))

    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_shape_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    modes = ["rect", "line", "blob"]
    if height >= 2 and width >= 2:
        modes.append("el")
    if height >= 3 and width >= 3:
        modes.extend(["plus", "ring"])

    for _ in range(100):
        mode = rng.choice(modes)
        mask = np.zeros((height, width), dtype=bool)

        if mode == "rect":
            box_h = rng.randint(1, height)
            box_w = rng.randint(1, width)
            if box_h == height and box_w == width:
                continue
            r0 = rng.randint(0, height - box_h)
            c0 = rng.randint(0, width - box_w)
            mask[r0:r0 + box_h, c0:c0 + box_w] = True

        elif mode == "line":
            horizontal = rng.choice([False, True])
            if horizontal and width >= 2:
                line_w = rng.randint(2, width)
                r0 = rng.randrange(height)
                c0 = rng.randint(0, width - line_w)
                mask[r0, c0:c0 + line_w] = True
            elif not horizontal and height >= 2:
                line_h = rng.randint(2, height)
                r0 = rng.randint(0, height - line_h)
                c0 = rng.randrange(width)
                mask[r0:r0 + line_h, c0] = True
            else:
                continue

        elif mode == "el":
            box_h = rng.randint(2, height)
            box_w = rng.randint(2, width)
            r0 = rng.randint(0, height - box_h)
            c0 = rng.randint(0, width - box_w)
            mask[r0:r0 + box_h, c0] = True
            mask[r0 + box_h - 1, c0:c0 + box_w] = True
            if rng.choice([False, True]):
                mask = np.flipud(mask)
            if rng.choice([False, True]):
                mask = np.fliplr(mask)

        elif mode == "plus":
            center_r = rng.randrange(height)
            center_c = rng.randrange(width)
            up = rng.randint(0, center_r)
            down = rng.randint(0, height - center_r - 1)
            left = rng.randint(0, center_c)
            right = rng.randint(0, width - center_c - 1)
            if up + down + left + right == 0:
                continue
            mask[center_r - up:center_r + down + 1, center_c] = True
            mask[center_r, center_c - left:center_c + right + 1] = True

        elif mode == "ring":
            box_h = rng.randint(3, height)
            box_w = rng.randint(3, width)
            if box_h == height and box_w == width and height * width <= 9:
                continue
            r0 = rng.randint(0, height - box_h)
            c0 = rng.randint(0, width - box_w)
            mask[r0, c0:c0 + box_w] = True
            mask[r0 + box_h - 1, c0:c0 + box_w] = True
            mask[r0:r0 + box_h, c0] = True
            mask[r0:r0 + box_h, c0 + box_w - 1] = True

        else:
            mask = _random_blob_mask(height, width, rng)

        area = int(mask.sum())
        if 2 <= area < height * width:
            return mask

    raise RuntimeError("failed to create a non-degenerate panel pattern")


def _make_panel_prototype(panel_h: int, panel_w: int, rng: random.Random) -> np.ndarray:
    panel = np.full((panel_h, panel_w), BLACK, dtype=int)
    inner = panel[1:-1, 1:-1]
    inner[_random_shape_mask(panel_h - 2, panel_w - 2, rng)] = rng.choice(PATTERN_COLORS)
    return panel


def _sample_group_sizes(total_panels: int, num_groups: int, rng: random.Random) -> list[int]:
    want_tie = num_groups >= 2 and (total_panels - num_groups) >= 2 and rng.choice([False, True])
    for _ in range(200):
        counts = [1] * num_groups
        for _ in range(total_panels - num_groups):
            counts[rng.randrange(num_groups)] += 1
        ranked = sorted(counts, reverse=True)
        if ranked[0] < 2:
            continue
        if want_tie and ranked[0] == ranked[1]:
            return counts
        if not want_tie and ranked[0] > ranked[1]:
            return counts

    raise RuntimeError("failed to sample valid group sizes")


def _layout_starts(count: int, panel_size: int) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for idx in range(count):
        starts.append(cursor)
        cursor += panel_size
        if idx < count - 1:
            cursor += 1
    return starts


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    panel_rows = rng.randint(2, 4)
    panel_cols = rng.randint(2, 4)
    total_panels = panel_rows * panel_cols
    num_groups = rng.randint(2, min(4, total_panels - 1))
    panel_h = rng.randint(5, 8)
    panel_w = rng.randint(5, 8)

    try:
        group_sizes = _sample_group_sizes(total_panels, num_groups, rng)
    except RuntimeError:
        return None

    prototypes: list[np.ndarray] = []
    seen_signatures: set[bytes] = set()
    for _ in range(num_groups):
        for _ in range(200):
            panel = _make_panel_prototype(panel_h, panel_w, rng)
            signature = panel.tobytes()
            if signature not in seen_signatures:
                prototypes.append(panel)
                seen_signatures.add(signature)
                break
        else:
            return None

    labels: list[int] = []
    for group_id, count in enumerate(group_sizes):
        labels.extend([group_id] * count)
    rng.shuffle(labels)

    grid_h = panel_rows * panel_h + (panel_rows - 1)
    grid_w = panel_cols * panel_w + (panel_cols - 1)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    row_starts = _layout_starts(panel_rows, panel_h)
    col_starts = _layout_starts(panel_cols, panel_w)
    for r0 in row_starts[1:]:
        grid[r0 - 1, :] = GREY
    for c0 in col_starts[1:]:
        grid[:, c0 - 1] = GREY

    label_idx = 0
    for r0 in row_starts:
        for c0 in col_starts:
            grid[r0:r0 + panel_h, c0:c0 + panel_w] = prototypes[labels[label_idx]]
            label_idx += 1

    try:
        groups = _group_panels(grid)
        actual_sizes = sorted((len(group["panels"]) for group in groups), reverse=True)
        assert len(groups) == num_groups
        assert actual_sizes == sorted(group_sizes, reverse=True)
        out = main(grid)
        assert out.shape == grid.shape
        assert not np.array_equal(out, grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-grouping task input")
