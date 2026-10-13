from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _separator_info(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    best: tuple[tuple[int, int, int, int], int, np.ndarray, np.ndarray] | None = None
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        full_rows = np.all(grid == color, axis=1)
        full_cols = np.all(grid == color, axis=0)
        score = (
            int(full_rows.sum() + full_cols.sum()),
            int(full_rows.sum()),
            int(full_cols.sum()),
            int((grid == color).sum()),
        )
        if score[0] == 0:
            continue
        if best is None or score > best[0]:
            best = (score, int(color), full_rows, full_cols)

    assert best is not None, "expected one color to form the partition lines"
    _score, color, full_rows, full_cols = best
    return int(color), np.flatnonzero(full_rows).tolist(), np.flatnonzero(full_cols).tolist()


def _spans(size: int, lines: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(int(x) for x in lines) + [int(size)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:], strict=False):
        if right - left > 1:
            spans.append((int(left + 1), int(right - 1)))
    return spans


def _region_stats(region: np.ndarray, bg: int, separator: int) -> list[dict[str, object]]:
    stats: list[dict[str, object]] = []
    for color in map(int, np.unique(region)):
        if color in (bg, separator):
            continue
        coords = np.argwhere(region == color)
        if coords.size == 0:
            continue
        stats.append(
            {
                "color": int(color),
                "count": int(coords.shape[0]),
                "first": (int(coords[0, 0]), int(coords[0, 1])),
                "coords": coords,
            }
        )
    assert stats, "every region must contain at least one non-background color"
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each partitioned region with its most frequent color while preserving
    the original pixels of its least frequent color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    separator, row_lines, col_lines = _separator_info(g, bg)
    row_spans = _spans(g.shape[0], row_lines)
    col_spans = _spans(g.shape[1], col_lines)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            stats = _region_stats(region, bg, separator)
            dominant = min(
                stats,
                key=lambda item: (-int(item["count"]), tuple(item["first"])),
            )
            rare = min(
                stats,
                key=lambda item: (int(item["count"]), tuple(item["first"])),
            )

            out[r0 : r1 + 1, c0 : c1 + 1] = int(dominant["color"])
            rare_coords = np.asarray(rare["coords"], dtype=int)
            out[r0 + rare_coords[:, 0], c0 + rare_coords[:, 1]] = int(rare["color"])

    return out


# --------------- generator ---------------

_BG = 0
_SEPARATOR = 5
_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
_LAYOUTS = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3))
_REGION_SIZE_RANGE = (5, 8)
_GENERATE_RETRIES = 400

_PATTERN_TEMPLATES: dict[str, dict[int, tuple[tuple[int, ...], ...]]] = {
    "unique": {
        2: ((3, 1), (4, 1), (4, 2)),
        3: ((3, 2, 1), (4, 2, 1), (4, 3, 1)),
        4: ((3, 2, 2, 1), (4, 2, 2, 1)),
    },
    "max_tie": {
        3: ((2, 2, 1), (3, 3, 1), (3, 3, 2)),
        4: ((2, 2, 2, 1), (3, 3, 2, 1)),
    },
    "min_tie": {
        3: ((3, 1, 1), (4, 1, 1), (4, 2, 2)),
        4: ((3, 2, 1, 1), (4, 2, 1, 1), (4, 3, 1, 1)),
    },
    "both_tie": {
        4: ((2, 2, 1, 1), (3, 3, 1, 1)),
    },
}


def _line_positions(lengths: list[int]) -> list[int]:
    pos = 0
    lines = [0]
    for length in lengths:
        pos += int(length) + 1
        lines.append(pos)
    return lines


def _pattern_sequence(count: int) -> list[str]:
    base = list(_PATTERN_TEMPLATES)
    random.shuffle(base)
    seq: list[str] = []
    while len(seq) < count:
        seq.extend(base)
        random.shuffle(base)
    random.shuffle(seq)
    return seq[:count]


def _slot_options(height: int, width: int) -> list[list[tuple[int, int]]]:
    options: list[list[tuple[int, int]]] = []
    for row_offset in (0, 1):
        rows = list(range(row_offset, height, 2))
        if not rows:
            continue
        for col_offset in (0, 1):
            cols = list(range(col_offset, width, 2))
            if not cols:
                continue
            options.append([(r, c) for r in rows for c in cols])
    return options


def _choose_counts(pattern: str, capacity: int) -> tuple[int, list[int]]:
    valid: list[tuple[int, tuple[int, ...]]] = []
    for color_count, templates in _PATTERN_TEMPLATES[pattern].items():
        for counts in templates:
            if sum(counts) <= capacity:
                valid.append((int(color_count), counts))
    assert valid, "region must be large enough for its count pattern"
    color_count, counts = random.choice(valid)
    return int(color_count), list(counts)


def _paint_region(
    grid: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    pattern: str,
) -> None:
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    slot_options = _slot_options(height, width)
    capacity = max(len(option) for option in slot_options)

    color_count, counts = _choose_counts(pattern, capacity)
    total = sum(counts)
    valid_slots = [option[:] for option in slot_options if len(option) >= total]
    assert valid_slots, "expected at least one sparse placement layout"

    slots = random.choice(valid_slots)
    random.shuffle(slots)
    colors = random.sample(_PALETTE, color_count)
    color_specs = list(zip(colors, counts, strict=False))
    random.shuffle(color_specs)

    cursor = 0
    for color, count in color_specs:
        for _ in range(int(count)):
            rr, cc = slots[cursor]
            cursor += 1
            grid[r0 + rr, c0 + cc] = int(color)


def _region_boxes(
    row_lengths: list[int],
    col_lengths: list[int],
) -> list[tuple[int, int, int, int]]:
    row_lines = _line_positions(row_lengths)
    col_lines = _line_positions(col_lengths)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(len(row_lines) - 1):
        for j in range(len(col_lines) - 1):
            boxes.append(
                (
                    int(row_lines[i] + 1),
                    int(row_lines[i + 1] - 1),
                    int(col_lines[j] + 1),
                    int(col_lines[j + 1] - 1),
                )
            )
    return boxes


def _is_sparse_region(region: np.ndarray) -> bool:
    h, w = region.shape
    colored = np.argwhere(region != _BG)
    for r, c in colored:
        for rr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                if (rr, cc) == (int(r), int(c)):
                    continue
                if int(region[rr, cc]) != _BG:
                    return False
    return True


def _validate_generated(grid: np.ndarray) -> bool:
    bg = _most_common_color(grid)
    if bg != _BG:
        return False

    separator, row_lines, col_lines = _separator_info(grid, bg)
    if separator != _SEPARATOR:
        return False

    row_spans = _spans(grid.shape[0], row_lines)
    col_spans = _spans(grid.shape[1], col_lines)
    if len(row_spans) * len(col_spans) < 4:
        return False

    tie_seen = False
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            stats = _region_stats(region, bg, separator)
            if not (2 <= len(stats) <= 4):
                return False
            counts = [int(item["count"]) for item in stats]
            if len(set(counts)) < len(counts):
                tie_seen = True
            dominant = min(stats, key=lambda item: (-int(item["count"]), tuple(item["first"])))
            rare = min(stats, key=lambda item: (int(item["count"]), tuple(item["first"])))
            if int(dominant["color"]) == int(rare["color"]):
                return False
            if not _is_sparse_region(region):
                return False

    out = main(grid)
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            if np.any(out[r0 : r1 + 1, c0 : c1 + 1] == bg):
                return False
    return tie_seen


def generate_input() -> np.ndarray:
    for _ in range(_GENERATE_RETRIES):
        row_count, col_count = random.choice(_LAYOUTS)
        row_lengths = [random.randint(*_REGION_SIZE_RANGE) for _ in range(row_count)]
        col_lengths = [random.randint(*_REGION_SIZE_RANGE) for _ in range(col_count)]

        height = sum(row_lengths) + row_count + 1
        width = sum(col_lengths) + col_count + 1
        grid = np.full((height, width), _BG, dtype=int)

        row_lines = _line_positions(row_lengths)
        col_lines = _line_positions(col_lengths)
        grid[row_lines, :] = _SEPARATOR
        grid[:, col_lines] = _SEPARATOR

        boxes = _region_boxes(row_lengths, col_lengths)
        patterns = _pattern_sequence(len(boxes))
        for (r0, r1, c0, c1), pattern in zip(boxes, patterns, strict=False):
            _paint_region(grid, r0, r1, c0, c1, pattern)

        if _validate_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid partition-color puzzle input")
