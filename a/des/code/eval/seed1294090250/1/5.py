from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_REGIONS = 3
MAX_REGIONS = 5
MIN_REGION_WIDTH = 5
MAX_REGION_WIDTH = 8
MIN_BODY_HEIGHT = 8
MAX_BODY_HEIGHT = 15
MIN_SHAPE_HEIGHT = 3
MAX_SHAPE_HEIGHT = 6
GENERATION_ATTEMPTS = 400


def _pair_key(a: int, b: int) -> frozenset[int]:
    return frozenset((int(a), int(b)))


def _separator_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _region_spans(width: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + separators + [width]
    spans = []
    for i in range(len(bounds) - 1):
        c0 = bounds[i] + 1
        c1 = bounds[i + 1] - 1
        assert c0 <= c1, "each stripe region must contain at least one column"
        spans.append((int(c0), int(c1)))
    return spans


def _legend_mapping(bottom_row: np.ndarray) -> dict[frozenset[int], int]:
    legend: dict[frozenset[int], int] = {}
    c = 0
    while c < bottom_row.size:
        color = int(bottom_row[c])
        if color in (BLACK, GREY):
            c += 1
            continue

        start = c
        while c < bottom_row.size and int(bottom_row[c]) not in (BLACK, GREY):
            c += 1
        run = [int(v) for v in bottom_row[start:c]]
        assert len(run) >= 3 and len(run) % 3 == 0, "legend must be built from 3-pixel groups"

        for i in range(0, len(run), 3):
            a, mixed, b = run[i : i + 3]
            key = _pair_key(a, b)
            if key in legend:
                assert legend[key] == mixed, "legend entries for the same pair must agree"
            else:
                legend[key] = int(mixed)

    assert legend, "expected at least one bottom-row legend entry"
    return legend


def _stripe_shape_info(grid: np.ndarray, c0: int, c1: int) -> tuple[int, set[int]]:
    body = grid[:-1, c0 : c1 + 1]
    mask = (body != BLACK) & (body != GREY)
    colors = np.unique(body[mask])
    assert len(colors) == 1, "each stripe must contain exactly one non-legend shape color"
    color = int(colors[0])
    rows = {int(r) for r in np.flatnonzero(np.any(body == color, axis=1))}
    assert rows, "each stripe shape must occupy at least one row"
    return color, rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw mixed-color bridges on grey separators where neighboring stripe shapes overlap in height.

    Input structure:
    - Black background.
    - 3 to 5 stripe regions separated by full-height grey vertical lines.
    - Each stripe contains one connected monochrome shape above the bottom row.
    - The bottom row is a color-mixing legend made of 3-pixel groups [color_a, mix, color_b].

    Rule:
    - Read the unordered color-pair -> mixed-color mapping from the legend.
    - For every grey separator, find the rows where the left and right stripe shapes both appear.
    - Recolor those grey separator pixels to the mixed color for the two neighboring shape colors.
    - Keep all non-overlapping separator rows and all shapes unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 7, "expected a 2D ARC grid"
    assert int(g[0, 0]) == BLACK or np.count_nonzero(g == BLACK) > 0, "expected black background"

    separators = _separator_columns(g)
    assert MIN_REGIONS - 1 <= len(separators) <= MAX_REGIONS - 1, "expected 3 to 5 stripe regions"
    spans = _region_spans(g.shape[1], separators)

    legend = _legend_mapping(g[-1])
    stripe_info = [_stripe_shape_info(g, c0, c1) for c0, c1 in spans]

    out = g.copy()
    for i, sep_col in enumerate(separators):
        left_color, left_rows = stripe_info[i]
        right_color, right_rows = stripe_info[i + 1]
        key = _pair_key(left_color, right_color)
        assert key in legend, "missing legend entry for adjacent stripe colors"
        overlap_rows = left_rows & right_rows
        if overlap_rows:
            rr = np.fromiter(overlap_rows, dtype=int)
            out[rr, sep_col] = legend[key]

    return out


def _sample_relations(rng: np.random.Generator, count: int) -> list[bool]:
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return [False] * count
    if mode == 1:
        return [True] * count

    while True:
        relations = [bool(v) for v in rng.integers(0, 2, size=count)]
        if any(relations) and not all(relations):
            return relations


def _sample_intervals(
    rng: np.random.Generator,
    body_height: int,
    region_count: int,
    relations: list[bool],
) -> list[tuple[int, int]] | None:
    max_len = min(MAX_SHAPE_HEIGHT, body_height - 1)
    if max_len < MIN_SHAPE_HEIGHT:
        return None

    for _ in range(150):
        intervals: list[tuple[int, int]] = []

        first_len = int(rng.integers(MIN_SHAPE_HEIGHT, max_len + 1))
        first_start = int(rng.integers(0, body_height - first_len + 1))
        intervals.append((first_start, first_start + first_len - 1))

        ok = True
        for rel in relations:
            length = int(rng.integers(MIN_SHAPE_HEIGHT, max_len + 1))
            prev_start, prev_end = intervals[-1]

            if rel:
                lo = max(0, prev_start - length + 1)
                hi = min(body_height - length, prev_end)
                if lo > hi:
                    ok = False
                    break
                start = int(rng.integers(lo, hi + 1))
            else:
                choices: list[tuple[int, int]] = []
                above_hi = prev_start - length
                if above_hi >= 0:
                    choices.append((0, above_hi))
                below_lo = prev_end + 1
                below_hi = body_height - length
                if below_lo <= below_hi:
                    choices.append((below_lo, below_hi))
                if not choices:
                    ok = False
                    break
                lo, hi = choices[int(rng.integers(0, len(choices)))]
                start = int(rng.integers(lo, hi + 1))

            intervals.append((start, start + length - 1))

        if ok and len(intervals) == region_count:
            return intervals

    return None


def _build_shape_cells(
    rng: np.random.Generator,
    top: int,
    bottom: int,
    region_width: int,
) -> list[tuple[int, int]]:
    shape_height = bottom - top + 1
    usable_left = 1
    usable_right = region_width - 2
    usable_width = usable_right - usable_left + 1
    assert usable_width >= 2, "shape generator needs side padding inside the stripe"

    max_seg_width = min(4, usable_width)
    segments: list[tuple[int, int]] = []
    mask = np.zeros((shape_height, region_width), dtype=bool)

    for row in range(shape_height):
        width = int(rng.integers(1, max_seg_width + 1))
        max_start = usable_right - width + 1
        if row == 0:
            start = int(rng.integers(usable_left, max_start + 1))
        else:
            prev_a, prev_b = segments[-1]
            anchor = int(rng.integers(prev_a, prev_b + 1))
            lo = max(usable_left, anchor - width + 1)
            hi = min(max_start, anchor)
            start = int(rng.integers(lo, hi + 1))
        end = start + width - 1
        segments.append((start, end))
        mask[row, start : end + 1] = True

    extra_steps = int(rng.integers(0, shape_height + 1))
    for _ in range(extra_steps):
        row = int(rng.integers(0, shape_height))
        left, right = segments[row]
        if rng.random() < 0.5 and left > usable_left:
            mask[row, left - 1] = True
            segments[row] = (left - 1, right)
        elif right < usable_right:
            mask[row, right + 1] = True
            segments[row] = (left, right + 1)

    starts = {seg[0] for seg in segments}
    widths = {seg[1] - seg[0] + 1 for seg in segments}
    if len(starts) == 1 and len(widths) == 1:
        row = int(rng.integers(0, shape_height))
        left, right = segments[row]
        if right < usable_right:
            mask[row, right + 1] = True
        elif left > usable_left:
            mask[row, left - 1] = True

    coords = [(top + int(r), int(c)) for r, c in np.argwhere(mask)]
    return coords


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _all_pairs(colors: list[int]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            pairs.append((int(colors[i]), int(colors[j])))
    return pairs


def _legend_groups(
    rng: np.random.Generator,
    shape_colors: list[int],
) -> tuple[dict[frozenset[int], int], list[tuple[int, int, int]]]:
    adjacent_pairs = [(int(shape_colors[i]), int(shape_colors[i + 1])) for i in range(len(shape_colors) - 1)]
    legend_pairs = list(adjacent_pairs)

    unused_pairs = [pair for pair in _all_pairs(shape_colors) if pair not in set(adjacent_pairs)]
    if unused_pairs and rng.random() < 0.5:
        legend_pairs.append(unused_pairs[int(rng.integers(0, len(unused_pairs)))])

    mapping: dict[frozenset[int], int] = {}
    groups: list[tuple[int, int, int]] = []
    for a, b in legend_pairs:
        mix_choices = [color for color in PALETTE if color not in (a, b)]
        mixed = int(mix_choices[int(rng.integers(0, len(mix_choices)))])
        mapping[_pair_key(a, b)] = mixed
        if rng.random() < 0.5:
            groups.append((a, mixed, b))
        else:
            groups.append((b, mixed, a))

    rng.shuffle(groups)
    return mapping, groups


def _draw_legend(
    rng: np.random.Generator,
    grid: np.ndarray,
    spans: list[tuple[int, int]],
    groups: list[tuple[int, int, int]],
) -> None:
    region_ids = rng.permutation(len(spans))[: len(groups)]
    for region_id, group in zip(region_ids, groups):
        c0, c1 = spans[int(region_id)]
        width = c1 - c0 + 1
        start = int(rng.integers(0, width - 3 + 1))
        grid[-1, c0 + start : c0 + start + 3] = np.array(group, dtype=int)


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    region_count = int(rng.integers(MIN_REGIONS, MAX_REGIONS + 1))
    region_width = int(rng.integers(MIN_REGION_WIDTH, MAX_REGION_WIDTH + 1))
    body_height = int(rng.integers(MIN_BODY_HEIGHT, MAX_BODY_HEIGHT + 1))
    height = body_height + 1
    width = region_count * region_width + (region_count - 1)

    relations = _sample_relations(rng, region_count - 1)
    intervals = _sample_intervals(rng, body_height, region_count, relations)
    if intervals is None:
        return None

    shape_colors = [int(c) for c in rng.choice(np.array(PALETTE, dtype=int), size=region_count, replace=False)]
    legend_mapping, groups = _legend_groups(rng, shape_colors)

    grid = np.full((height, width), BLACK, dtype=int)
    separators = []
    spans = []
    for idx in range(region_count):
        c0 = idx * (region_width + 1)
        c1 = c0 + region_width - 1
        spans.append((c0, c1))
        if idx < region_count - 1:
            sep_col = c1 + 1
            separators.append(sep_col)
            grid[:, sep_col] = GREY

    for (c0, c1), (top, bottom), color in zip(spans, intervals, shape_colors):
        coords = _build_shape_cells(rng, top, bottom, c1 - c0 + 1)
        local_mask = np.zeros((body_height, c1 - c0 + 1), dtype=bool)
        for r, local_c in coords:
            grid[r, c0 + local_c] = color
            local_mask[r, local_c] = True
        if not _is_connected(local_mask):
            return None

    _draw_legend(rng, grid, spans, groups)

    for i, sep_col in enumerate(separators):
        left_color = shape_colors[i]
        right_color = shape_colors[i + 1]
        if _pair_key(left_color, right_color) not in legend_mapping:
            return None
        if not np.all(grid[:, sep_col] == GREY):
            return None

    if not _validate_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        candidate = _generate_attempt(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
