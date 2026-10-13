from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
PINK = 6
PAIR_COLORS = (1, 2, 3, 4, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400
GRID_SIZE_MIN = 16
GRID_SIZE_MAX = 30
PAIR_COUNT_RANGE = (2, 4)
BAR_HEIGHT_CHOICES = (2, 4, 6, 8)
MIN_BAR_GAP = 3
INPUT_PADDING = 1


@dataclass(frozen=True)
class VerticalRun:
    color: int
    col: int
    r0: int
    r1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1


@dataclass(frozen=True)
class PairSpec:
    color: int
    r0: int
    height: int
    left_col: int
    right_col: int
    direction: str
    marked_bar: str

    @property
    def r1(self) -> int:
        return self.r0 + self.height - 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _vertical_runs(grid: np.ndarray, color: int) -> list[VerticalRun]:
    runs: list[VerticalRun] = []
    height, width = map(int, grid.shape)
    for col in range(width):
        row = 0
        while row < height:
            if int(grid[row, col]) != color:
                row += 1
                continue
            start = row
            while row + 1 < height and int(grid[row + 1, col]) == color:
                row += 1
            runs.append(VerticalRun(color=color, col=col, r0=start, r1=row))
            row += 1
    return runs


def _parse_pairs(grid: np.ndarray) -> tuple[int, list[PairSpec]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    colors = sorted(set(map(int, np.unique(g))) - {bg})
    assert PINK not in colors, "pink is reserved for output overlaps"
    grey_runs = _vertical_runs(g, GREY)
    pair_colors = [color for color in colors if color != GREY]
    assert pair_colors, "expected at least one colored bar pair"

    used_grey: set[int] = set()
    pairs: list[PairSpec] = []

    for color in pair_colors:
        runs = _vertical_runs(g, color)
        assert len(runs) == 2, f"expected two visible runs for color {color}, got {len(runs)}"
        runs = sorted(runs, key=lambda run: (run.height, run.col))
        half_run, full_run = runs
        assert full_run.height == 2 * half_run.height, "marked bar half must be exactly half of the full bar height"

        match: tuple[int, str, int, int] | None = None
        for idx, grey_run in enumerate(grey_runs):
            if idx in used_grey:
                continue
            if grey_run.col != half_run.col or grey_run.height != half_run.height:
                continue
            if grey_run.r1 + 1 == half_run.r0:
                match = (idx, "up", grey_run.r0, half_run.r1)
                break
            if half_run.r1 + 1 == grey_run.r0:
                match = (idx, "down", half_run.r0, grey_run.r1)
                break

        assert match is not None, f"expected one grey half matching color {color}"
        grey_idx, direction, total_r0, total_r1 = match
        used_grey.add(grey_idx)

        assert full_run.r0 == total_r0 and full_run.r1 == total_r1, "paired bars must share the same row interval"
        left_col, right_col = sorted((half_run.col, full_run.col))
        pairs.append(
            PairSpec(
                color=color,
                r0=full_run.r0,
                height=full_run.height,
                left_col=left_col,
                right_col=right_col,
                direction=direction,
                marked_bar="left",
            )
        )

    assert len(used_grey) == len(grey_runs), "every grey marker must belong to exactly one pair"
    return bg, pairs


def _add_segment(
    coverage: dict[tuple[int, int], set[int]],
    color: int,
    seg_r0: int,
    seg_r1: int,
    col: int,
    shape: tuple[int, int],
) -> None:
    height, width = map(int, shape)
    if not (0 <= col < width):
        return
    row0 = max(0, seg_r0)
    row1 = min(height - 1, seg_r1)
    if row0 > row1:
        return
    for row in range(row0, row1 + 1):
        coverage.setdefault((row, col), set()).add(color)


def _render_output(shape: tuple[int, int], pairs: list[PairSpec], bg: int = BLACK) -> np.ndarray:
    height, width = map(int, shape)
    coverage: dict[tuple[int, int], set[int]] = {}

    for pair in pairs:
        step = 0
        while True:
            if pair.direction == "up":
                seg_r0 = pair.r0 - step * pair.height
                seg_r1 = pair.r1 - step * pair.height
            else:
                seg_r0 = pair.r0 + step * pair.height
                seg_r1 = pair.r1 + step * pair.height

            left_col = pair.left_col - step
            right_col = pair.right_col + step
            if seg_r1 < 0 or seg_r0 >= height:
                break
            if left_col < 0 and right_col >= width:
                break

            _add_segment(coverage, pair.color, seg_r0, seg_r1, left_col, shape)
            _add_segment(coverage, pair.color, seg_r0, seg_r1, right_col, shape)
            step += 1

    out = np.full(shape, bg, dtype=int)
    for (row, col), colors in coverage.items():
        out[row, col] = PINK if len(colors) >= 2 else next(iter(colors))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each marked vertical-bar pair and extend only the one-sided staircase rays chosen by the grey half-mark.

    Input structure:
    - A black background contains several same-color vertical bar pairs; in each pair the two bars have equal height, share the same row interval, and sit in different columns.
    - For each pair, exactly one bar has its top half or bottom half recolored grey.
    - Different pairs use different non-grey colors, and pink is absent from the input.

    Rule:
    - Recolor every grey half-marker back to the color of its bar.
    - For each pair, use the marked half as a direction switch: a top-half marker means both bars extend only upward, and a bottom-half marker means both bars extend only downward.
    - The original bar interval is the first segment; every additional segment has the same height, while the left ray shifts one column farther left per segment and the right ray shifts one column farther right per segment. Cells hit by rays of two or more different colors become pink.

    Key insight:
    - The grey half is not an object to keep; it encodes both the unique growth direction and the staircase period, because the bar height itself is the segment length.
    """
    bg, pairs = _parse_pairs(np.asarray(grid, dtype=int))
    return _render_output(grid.shape, pairs, bg=bg)


def _pair_bbox(pair: PairSpec, pad: int = INPUT_PADDING) -> tuple[int, int, int, int]:
    return (
        pair.r0 - pad,
        pair.left_col - pad,
        pair.r1 + pad,
        pair.right_col + pad,
    )


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _can_place(pair: PairSpec, reserved: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    if not (0 <= pair.r0 <= pair.r1 < height):
        return False
    if not (0 <= pair.left_col < pair.right_col < width):
        return False
    bbox = _pair_bbox(pair)
    return all(not _bboxes_overlap(bbox, other) for other in reserved)


def _sample_columns(width: int) -> tuple[int, int] | None:
    if width < MIN_BAR_GAP + 4:
        return None
    left = random.randint(1, width - MIN_BAR_GAP - 3)
    right = random.randint(left + MIN_BAR_GAP, width - 2)
    return left, right


def _sample_independent_pair(color: int, shape: tuple[int, int], reserved: list[tuple[int, int, int, int]]) -> PairSpec | None:
    height, width = map(int, shape)
    feasible_heights = [bar_h for bar_h in BAR_HEIGHT_CHOICES if 2 * bar_h <= height]
    if not feasible_heights:
        return None

    for _ in range(200):
        bar_h = random.choice(feasible_heights)
        direction = random.choice(("up", "down"))
        if direction == "up":
            r0_min, r0_max = bar_h, height - bar_h
        else:
            r0_min, r0_max = 0, height - 2 * bar_h
        if r0_min > r0_max:
            continue

        cols = _sample_columns(width)
        if cols is None:
            return None
        left_col, right_col = cols
        pair = PairSpec(
            color=color,
            r0=random.randint(r0_min, r0_max),
            height=bar_h,
            left_col=left_col,
            right_col=right_col,
            direction=direction,
            marked_bar=random.choice(("left", "right")),
        )
        if _can_place(pair, reserved, shape):
            return pair
    return None


def _sample_overlap_duo(
    colors: tuple[int, int], shape: tuple[int, int], reserved: list[tuple[int, int, int, int]]
) -> list[PairSpec] | None:
    height, width = map(int, shape)
    feasible_heights = [bar_h for bar_h in BAR_HEIGHT_CHOICES if 3 * bar_h <= height]
    if not feasible_heights:
        return None

    for _ in range(200):
        bar_h = random.choice(feasible_heights)
        cols = _sample_columns(width)
        if cols is None:
            return None
        left_col, right_col = cols
        top_r0 = random.randint(0, height - 3 * bar_h)
        top = PairSpec(
            color=colors[0],
            r0=top_r0,
            height=bar_h,
            left_col=left_col,
            right_col=right_col,
            direction="down",
            marked_bar=random.choice(("left", "right")),
        )
        bottom = PairSpec(
            color=colors[1],
            r0=top_r0 + 2 * bar_h,
            height=bar_h,
            left_col=left_col,
            right_col=right_col,
            direction="up",
            marked_bar=random.choice(("left", "right")),
        )
        duo_reserved = reserved + [_pair_bbox(top)]
        if _can_place(top, reserved, shape) and _can_place(bottom, duo_reserved, shape):
            return [top, bottom]
    return None


def _render_input(shape: tuple[int, int], pairs: list[PairSpec]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for pair in pairs:
        grid[pair.r0 : pair.r1 + 1, pair.left_col] = pair.color
        grid[pair.r0 : pair.r1 + 1, pair.right_col] = pair.color
        marked_col = pair.left_col if pair.marked_bar == "left" else pair.right_col
        half = pair.height // 2
        if pair.direction == "up":
            grid[pair.r0 : pair.r0 + half, marked_col] = GREY
        else:
            grid[pair.r1 - half + 1 : pair.r1 + 1, marked_col] = GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX),
        random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX),
    )
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(PAIR_COLORS, pair_count)

    reserved: list[tuple[int, int, int, int]] = []
    pairs: list[PairSpec] = []
    want_overlap = pair_count >= 2 and random.random() < 0.65

    color_index = 0
    if want_overlap:
        duo = _sample_overlap_duo((colors[0], colors[1]), shape, reserved)
        if duo is None:
            return None
        pairs.extend(duo)
        reserved.extend(_pair_bbox(pair) for pair in duo)
        color_index = 2

    while color_index < pair_count:
        pair = _sample_independent_pair(colors[color_index], shape, reserved)
        if pair is None:
            return None
        pairs.append(pair)
        reserved.append(_pair_bbox(pair))
        color_index += 1

    inp = _render_input(shape, pairs)
    expected = _render_output(shape, pairs, bg=BLACK)

    if want_overlap and not np.any(expected == PINK):
        return None
    if np.sum(expected == PINK) > (shape[0] * shape[1]) // 7:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
