from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEP = 5
PAYLOAD_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

PANEL_ROW_RANGE = (2, 4)
PANEL_COL_RANGE = (2, 4)
PANEL_HEIGHT_RANGE = (4, 7)
PANEL_WIDTH_RANGE = (4, 7)
GENERATION_ATTEMPTS = 300

QUADRANTS = ("tl", "tr", "bl", "br")
DIAGONAL_PAIRS = (("tl", "br"), ("tr", "bl"))


def _separator_color(grid: np.ndarray) -> int:
    row_counts: Counter[int] = Counter()
    col_counts: Counter[int] = Counter()
    height, width = grid.shape

    for r in range(height):
        if np.all(grid[r, :] == grid[r, 0]):
            row_counts[int(grid[r, 0])] += 1
    for c in range(width):
        if np.all(grid[:, c] == grid[0, c]):
            col_counts[int(grid[0, c])] += 1

    candidates = set(row_counts) & set(col_counts)
    assert candidates, "expected a separator color spanning full rows and columns"

    ranked = sorted(
        candidates,
        key=lambda color: (row_counts[color] + col_counts[color], row_counts[color], col_counts[color], color),
        reverse=True,
    )
    return int(ranked[0])


def _contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, keep in enumerate(mask):
        if bool(keep):
            if start is None:
                start = int(idx)
        elif start is not None:
            spans.append((int(start), int(idx)))
            start = None
    if start is not None:
        spans.append((int(start), int(len(mask))))
    return spans


def _panel_spans(grid: np.ndarray, sep_color: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans = _contiguous_spans(np.array([not np.all(grid[r, :] == sep_color) for r in range(grid.shape[0])], dtype=bool))
    col_spans = _contiguous_spans(np.array([not np.all(grid[:, c] == sep_color) for c in range(grid.shape[1])], dtype=bool))

    assert row_spans and col_spans, "expected at least one panel in each axis"
    row_heights = {r1 - r0 for r0, r1 in row_spans}
    col_widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(row_heights) == 1, "panel rows must all have the same height"
    assert len(col_widths) == 1, "panel cols must all have the same width"
    return row_spans, col_spans


def _strict_majority(block: np.ndarray) -> int:
    counts = Counter(map(int, block.reshape(-1)))
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "panel majority color must be unique"
    return int(winners[0])


def _mark_targets(local_r: int, local_c: int, height: int, width: int) -> list[tuple[int, int]]:
    targets: list[tuple[int, int]] = []

    row_score = 2 * int(local_r) + 1
    if row_score < int(height):
        targets.append((0, 1))
    elif row_score > int(height):
        targets.append((2, 1))

    col_score = 2 * int(local_c) + 1
    if col_score < int(width):
        targets.append((1, 0))
    elif col_score > int(width):
        targets.append((1, 2))

    return targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each panel's majority color into a coarse grid and map minority-pixel positions to directional marks.

    Input structure:
    - Black background and grey separator rows/cols partition the grid into equal rectangular panels.
    - Each panel is mostly one fill color plus 0-2 minority pixels.

    Rule:
    - Each panel becomes one 3x3 output cell.
    - The center uses the panel's majority color.
    - A minority pixel in the top/bottom half paints the top/bottom center.
    - A minority pixel in the left/right half paints the left/right center.
    - Unused output pixels stay black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_color = _separator_color(g)
    row_spans, col_spans = _panel_spans(g, sep_color)

    panel_height = row_spans[0][1] - row_spans[0][0]
    panel_width = col_spans[0][1] - col_spans[0][0]

    out = np.full((len(row_spans) * 3, len(col_spans) * 3), BG, dtype=int)

    for pr, (r0, r1) in enumerate(row_spans):
        for pc, (c0, c1) in enumerate(col_spans):
            block = g[r0:r1, c0:c1]
            major = _strict_majority(block)

            base_r = pr * 3
            base_c = pc * 3
            out[base_r + 1, base_c + 1] = int(major)

            for local_r, local_c in np.argwhere(block != major):
                color = int(block[int(local_r), int(local_c)])
                for dr, dc in _mark_targets(int(local_r), int(local_c), panel_height, panel_width):
                    rr = base_r + dr
                    cc = base_c + dc
                    if int(out[rr, cc]) not in (BG, color):
                        raise AssertionError("minority markers map to the same output slot with different colors")
                    out[rr, cc] = int(color)

    return out


def _build_axis(count: int, panel_size: int, bordered: bool) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    separators: list[int] = []
    cursor = 0

    if bordered:
        separators.append(0)
        cursor = 1

    for idx in range(count):
        spans.append((int(cursor), int(cursor + panel_size)))
        cursor += int(panel_size)
        if idx != count - 1 or bordered:
            separators.append(int(cursor))
            cursor += 1

    return spans, separators, int(cursor)


def _quadrant_choices(height: int, width: int) -> dict[str, tuple[list[int], list[int]]]:
    top_rows = [r for r in range(height) if 2 * r + 1 < height]
    bottom_rows = [r for r in range(height) if 2 * r + 1 > height]
    left_cols = [c for c in range(width) if 2 * c + 1 < width]
    right_cols = [c for c in range(width) if 2 * c + 1 > width]

    assert top_rows and bottom_rows and left_cols and right_cols, "panels must support all four quadrants"

    return {
        "tl": (top_rows, left_cols),
        "tr": (top_rows, right_cols),
        "bl": (bottom_rows, left_cols),
        "br": (bottom_rows, right_cols),
    }


def _pick_position(height: int, width: int, quadrant: str) -> tuple[int, int]:
    row_choices, col_choices = _quadrant_choices(height, width)[quadrant]
    return int(random.choice(row_choices)), int(random.choice(col_choices))


def _render_expected(scene: list[list[dict]], panel_height: int, panel_width: int) -> np.ndarray:
    panel_rows = len(scene)
    panel_cols = len(scene[0])
    out = np.full((panel_rows * 3, panel_cols * 3), BG, dtype=int)

    for pr in range(panel_rows):
        for pc in range(panel_cols):
            panel = scene[pr][pc]
            base_r = pr * 3
            base_c = pc * 3
            out[base_r + 1, base_c + 1] = int(panel["major"])
            for color, local_r, local_c in panel["markers"]:
                for dr, dc in _mark_targets(int(local_r), int(local_c), panel_height, panel_width):
                    out[base_r + dr, base_c + dc] = int(color)

    return out


def _panel_marker_counts(panel_total: int) -> list[int]:
    counts = [idx % 3 for idx in range(panel_total)]
    random.shuffle(counts)
    while sum(counts) < 4:
        bumpable = [idx for idx, count in enumerate(counts) if count < 2]
        counts[random.choice(bumpable)] += 1
    return counts


def _assign_quadrants(marker_counts: list[int]) -> list[list[str]]:
    assignments: list[list[str]] = [[] for _ in marker_counts]
    uncovered = set(QUADRANTS)

    panel_order = list(range(len(marker_counts)))
    random.shuffle(panel_order)

    for idx in panel_order:
        count = marker_counts[idx]
        if count == 2:
            best_pair = None
            best_gain = -1
            for pair in DIAGONAL_PAIRS:
                gain = sum(1 for quadrant in pair if quadrant in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_pair = pair
            assert best_pair is not None
            pair_list = list(best_pair)
            random.shuffle(pair_list)
            assignments[idx] = pair_list
            uncovered -= set(pair_list)

    for idx in panel_order:
        if marker_counts[idx] != 1:
            continue
        if uncovered:
            quadrant = random.choice(sorted(uncovered))
            uncovered.remove(quadrant)
        else:
            quadrant = random.choice(QUADRANTS)
        assignments[idx] = [quadrant]

    for idx in panel_order:
        if marker_counts[idx] == 0:
            assignments[idx] = []
        elif marker_counts[idx] == 1 and not assignments[idx]:
            assignments[idx] = [random.choice(QUADRANTS)]
        elif marker_counts[idx] == 2 and not assignments[idx]:
            pair = list(random.choice(DIAGONAL_PAIRS))
            random.shuffle(pair)
            assignments[idx] = pair

    return assignments


def _sample_scene() -> tuple[np.ndarray, np.ndarray] | None:
    panel_rows = random.randint(*PANEL_ROW_RANGE)
    panel_cols = random.randint(*PANEL_COL_RANGE)
    panel_height = random.randint(*PANEL_HEIGHT_RANGE)
    panel_width = random.randint(*PANEL_WIDTH_RANGE)
    bordered = bool(random.getrandbits(1))

    row_spans, row_seps, height = _build_axis(panel_rows, panel_height, bordered)
    col_spans, col_seps, width = _build_axis(panel_cols, panel_width, bordered)

    grid = np.full((height, width), BG, dtype=int)
    if row_seps:
        grid[row_seps, :] = SEP
    if col_seps:
        grid[:, col_seps] = SEP

    panel_total = panel_rows * panel_cols
    marker_counts = _panel_marker_counts(panel_total)
    quadrant_assignments = _assign_quadrants(marker_counts)

    major_colors = [random.choice(PAYLOAD_COLORS) for _ in range(panel_total)]
    if len(set(major_colors)) == 1:
        swap_idx = random.randrange(panel_total)
        alternatives = [color for color in PAYLOAD_COLORS if color != major_colors[0]]
        major_colors[swap_idx] = random.choice(alternatives)

    scene: list[list[dict]] = []
    flat_idx = 0
    for pr, (r0, r1) in enumerate(row_spans):
        row_scene: list[dict] = []
        for pc, (c0, c1) in enumerate(col_spans):
            major = int(major_colors[flat_idx])
            count = int(marker_counts[flat_idx])
            quadrants = quadrant_assignments[flat_idx]

            panel = np.full((panel_height, panel_width), major, dtype=int)
            available = [color for color in PAYLOAD_COLORS if color != major]
            marker_colors = random.sample(available, count)

            markers: list[tuple[int, int, int]] = []
            used_positions: set[tuple[int, int]] = set()
            for marker_color, quadrant in zip(marker_colors, quadrants):
                for _ in range(30):
                    local_r, local_c = _pick_position(panel_height, panel_width, quadrant)
                    if (local_r, local_c) not in used_positions:
                        break
                else:
                    return None
                used_positions.add((local_r, local_c))
                panel[local_r, local_c] = int(marker_color)
                markers.append((int(marker_color), int(local_r), int(local_c)))

            grid[r0:r1, c0:c1] = panel
            row_scene.append({"major": major, "markers": markers})
            flat_idx += 1
        scene.append(row_scene)

    expected = _render_expected(scene, panel_height, panel_width)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        sample = _sample_scene()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid panel-direction ARC sample")
