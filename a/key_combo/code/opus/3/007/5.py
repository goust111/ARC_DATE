from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BG = Color.BLACK
SRC = Color.BLUE
SEP = Color.YELLOW
DIST = Color.ORANGE
GENERATION_ATTEMPTS = 400


def _find_separator(g: np.ndarray, bg: int) -> tuple[int, int]:
    rows: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        vals = np.unique(g[r, :])
        if vals.size == 1 and int(vals[0]) != int(bg):
            rows.append((int(r), int(vals[0])))
    assert len(rows) == 1, "expected exactly one full-width separator row"
    return rows[0]


def _is_solid_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = map(int, component["bbox"])
    return int(component["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_source_rectangles(
    g: np.ndarray,
    bg: int,
    sep_row: int,
    sep_color: int,
) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg), int(sep_color)}):
        components = find_components(
            g,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[color],
        )
        if not components:
            continue
        if all(
            int(component["area"]) > 1
            and _is_solid_rectangle(component)
            and int(component["bbox"][2]) < int(sep_row)
            for component in components
        ):
            bboxes = [tuple(map(int, component["bbox"])) for component in components]
            candidates.append((int(color), sorted(bboxes)))
    assert len(candidates) == 1, "expected one source color made of upper solid rectangles"
    return candidates[0]


def _aligned_visible_cols(target_right: int, width: int, grid_w: int) -> tuple[int, int]:
    left = int(target_right) - int(width) + 1
    return max(0, left), min(int(grid_w) - 1, int(target_right))


def _render_scene(
    shape: tuple[int, int],
    sep_row: int,
    sep_color: int,
    source_color: int,
    rects: list[tuple[int, int, int, int]],
    target_right: int,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[int(sep_row), :] = int(sep_color)

    for r0, c0, r1, c1 in rects:
        width = int(c1) - int(c0) + 1
        vis_c0, vis_c1 = _aligned_visible_cols(target_right, width, shape[1])
        mirror_r0 = 2 * int(sep_row) - int(r1)
        mirror_r1 = 2 * int(sep_row) - int(r0)
        assert 0 <= mirror_r0 <= mirror_r1 < shape[0], "reflected rectangle leaves the grid"

        gap_top = int(r1) + 1
        gap_bottom = int(mirror_r0) - 1
        if gap_top <= gap_bottom:
            out[gap_top : gap_bottom + 1, vis_c0 : vis_c1 + 1] = int(sep_color)

        out[int(r0) : int(r1) + 1, vis_c0 : vis_c1 + 1] = int(source_color)
        out[int(mirror_r0) : int(mirror_r1) + 1, vis_c0 : vis_c1 + 1] = int(source_color)
    return out


def _render_without_alignment(
    shape: tuple[int, int],
    sep_row: int,
    sep_color: int,
    source_color: int,
    rects: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[int(sep_row), :] = int(sep_color)

    for r0, c0, r1, c1 in rects:
        mirror_r0 = 2 * int(sep_row) - int(r1)
        mirror_r1 = 2 * int(sep_row) - int(r0)
        gap_top = int(r1) + 1
        gap_bottom = int(mirror_r0) - 1
        if gap_top <= gap_bottom:
            out[gap_top : gap_bottom + 1, int(c0) : int(c1) + 1] = int(sep_color)

        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(source_color)
        out[int(mirror_r0) : int(mirror_r1) + 1, int(c0) : int(c1) + 1] = int(source_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the upper solid rectangles by their right edges, then reflect them across the separator row and fill each resulting corridor with the separator color.

    Input structure:
    - A black background contains exactly one full-width horizontal separator row in a non-background color.
    - One other color forms several solid rectangles entirely above that separator row.
    - Remaining non-background pixels are isolated distractors that do not belong to the rectangles.

    Rule:
    - Ignore the distractors completely.
    - Find the minimum right-edge column among the upper solid rectangles.
    - Shift every rectangle horizontally so its right edge lands on that column, clipping only if the shifted left edge would cross column 0.
    - Reflect each aligned rectangle across the separator row and color the vertical strip between the aligned source and the reflection with the separator color.

    Key insight:
    - The true source objects are the only solid upper rectangles, and the crucial first step is the shared right-edge alignment: that alignment changes both the mirrored blue geometry and the yellow gap widths.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    sep_row, sep_color = _find_separator(g, bg)
    source_color, rects = _find_source_rectangles(g, bg, sep_row, sep_color)
    target_right = min(int(c1) for _r0, _c0, _r1, c1 in rects)
    return _render_scene(g.shape, sep_row, sep_color, source_color, rects, target_right)


def _composition(total: int, parts: int) -> list[int]:
    if total == 0:
        return [0] * parts
    return list(np.random.multinomial(int(total), [1.0 / parts] * parts))


def _pack_intervals(length: int, sizes: list[int], min_gap: int) -> list[tuple[int, int]] | None:
    extra = int(length) - sum(int(size) for size in sizes) - int(min_gap) * (len(sizes) - 1)
    if extra < 0:
        return None
    gaps = _composition(extra, len(sizes) + 1)
    intervals: list[tuple[int, int]] = []
    pos = int(gaps[0])
    for i, size in enumerate(sizes):
        intervals.append((pos, pos + int(size) - 1))
        pos += int(size)
        if i + 1 < len(sizes):
            pos += int(min_gap) + int(gaps[i + 1])
    return intervals


def _sample_heights(num_rects: int) -> tuple[int, list[int], list[tuple[int, int]]]:
    half_choices = range(8, 10) if num_rects == 4 else range(6, 10)
    for _ in range(200):
        half_h = random.choice(list(half_choices))
        usable_rows = half_h - 1
        feasible = [
            list(pattern)
            for pattern in itertools.product((1, 2), repeat=num_rects)
            if sum(pattern) + (num_rects - 1) <= usable_rows
        ]
        assert feasible
        heights = random.choice(feasible)
        random.shuffle(heights)
        intervals = _pack_intervals(usable_rows, heights, min_gap=1)
        if intervals is not None:
            return half_h, heights, intervals
    raise RuntimeError("failed to sample rectangle row spans")


def _sample_widths_and_columns(num_rects: int, clip_branch: bool) -> tuple[list[int], list[int], int, int]:
    for _ in range(200):
        if clip_branch:
            widths = [random.choice((2, 3, 4)) for _ in range(num_rects)]
            if 2 not in widths or 4 not in widths:
                continue
            target_right = 2
            anchor_candidates = [i for i, width in enumerate(widths) if width == 2]
        else:
            widths = [random.randint(2, 4) for _ in range(num_rects)]
            if len(set(widths)) < 2:
                continue
            target_right = random.randint(max(widths) - 1, 8)
            anchor_candidates = list(range(num_rects))

        visible_cols = {_aligned_visible_cols(target_right, width, 30) for width in widths}
        if len(visible_cols) < 2:
            continue

        rights = [-1] * num_rects
        anchor = random.choice(anchor_candidates)
        rights[anchor] = target_right
        moved = False
        for i, width in enumerate(widths):
            if i == anchor:
                continue
            low = max(target_right + 1, width - 1)
            high = low + 5
            rights[i] = random.randint(low, high)
            moved = True
        if not moved:
            continue

        max_right = max(rights)
        grid_w = max(10, max_right + 1 + random.randint(1, 4))
        if grid_w > 18:
            continue
        if any(right >= grid_w for right in rights):
            continue
        if clip_branch and not any(target_right - width + 1 < 0 for width in widths):
            continue
        if not clip_branch and any(target_right - width + 1 < 0 for width in widths):
            continue
        if max(rights) == target_right:
            continue
        return widths, rights, target_right, grid_w
    raise RuntimeError("failed to sample rectangle widths and columns")


def _forbidden_distractor_cells(
    shape: tuple[int, int],
    sep_row: int,
    rects: list[tuple[int, int, int, int]],
) -> set[tuple[int, int]]:
    forbidden: set[tuple[int, int]] = {(int(sep_row), c) for c in range(shape[1])}
    for r0, c0, r1, c1 in rects:
        for r in range(max(0, int(r0) - 1), min(shape[0], int(r1) + 2)):
            for c in range(max(0, int(c0) - 1), min(shape[1], int(c1) + 2)):
                forbidden.add((r, c))
    return forbidden


def _pick_isolated_cells(
    candidates: list[tuple[int, int]],
    count: int,
    taken: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    occupied = set(taken)
    for cell in candidates:
        r, c = cell
        if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in occupied):
            continue
        chosen.append(cell)
        occupied.add(cell)
        if len(chosen) == count:
            return chosen
    return None


def _sample_distractors(
    shape: tuple[int, int],
    sep_row: int,
    rects: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]] | None:
    forbidden = _forbidden_distractor_cells(shape, sep_row, rects)
    upper = [(r, c) for r in range(sep_row) for c in range(shape[1]) if (r, c) not in forbidden]
    lower = [
        (r, c)
        for r in range(sep_row + 1, shape[0])
        for c in range(shape[1])
        if (r, c) not in forbidden
    ]
    if not upper or not lower:
        return None

    chosen: set[tuple[int, int]] = set()
    top = _pick_isolated_cells(upper[:], 1, chosen)
    if top is None:
        return None
    chosen.update(top)

    bottom = _pick_isolated_cells(lower[:], 1, chosen)
    if bottom is None:
        return None
    chosen.update(bottom)

    total = random.randint(3, 5)
    remaining = total - len(chosen)
    if remaining > 0:
        others = _pick_isolated_cells(upper + lower, remaining, chosen)
        if others is None:
            return None
        chosen.update(others)
    return sorted(chosen)


def _render_input(
    shape: tuple[int, int],
    sep_row: int,
    rects: list[tuple[int, int, int, int]],
    distractors: list[tuple[int, int]],
) -> np.ndarray:
    g = np.full(shape, BG, dtype=int)
    g[int(sep_row), :] = SEP
    for r0, c0, r1, c1 in rects:
        g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = SRC
    for r, c in distractors:
        g[int(r), int(c)] = DIST
    return g


def _validate_scene(
    grid: np.ndarray,
    expected: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    target_right: int,
    clip_branch: bool,
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    rights = [int(c1) for _r0, _c0, _r1, c1 in rects]
    widths = [int(c1) - int(c0) + 1 for _r0, c0, _r1, c1 in rects]
    if min(rights) != int(target_right) or max(rights) == min(rights):
        return False
    if len(set(widths)) < 2:
        return False

    visible_cols = {_aligned_visible_cols(target_right, width, grid.shape[1]) for width in widths}
    if len(visible_cols) < 2:
        return False

    shortcut = _render_without_alignment(grid.shape, grid.shape[0] // 2, SEP, SRC, rects)
    if np.array_equal(shortcut, expected):
        return False

    clipped = any(target_right - width + 1 < 0 for width in widths)
    if clipped != bool(clip_branch):
        return False

    if len(find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[DIST])) != int(
        np.count_nonzero(grid == DIST)
    ):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    num_rects = random.choice((3, 4))
    half_h, _heights, row_intervals = _sample_heights(num_rects)
    clip_branch = random.random() < 0.35
    widths, rights, target_right, grid_w = _sample_widths_and_columns(num_rects, clip_branch)

    rects = []
    for (r0, r1), width, right in zip(row_intervals, widths, rights):
        left = int(right) - int(width) + 1
        rects.append((int(r0), int(left), int(r1), int(right)))
    rects.sort()

    shape = (2 * int(half_h) + 1, int(grid_w))
    expected = _render_scene(shape, half_h, SEP, SRC, rects, target_right)
    distractors = _sample_distractors(shape, half_h, rects)
    if distractors is None:
        return None

    grid = _render_input(shape, half_h, rects, distractors)
    if not _validate_scene(grid, expected, rects, target_right, clip_branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
