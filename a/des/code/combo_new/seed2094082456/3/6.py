from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
DIVIDER_COLORS = (GREY, MAROON)
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8)

GEN_RETRIES = 400
SIZE_RETRIES = 80
LAYOUT_RETRIES = 80


def _separator_rows(grid: np.ndarray) -> list[int]:
    return [
        int(r)
        for r in range(grid.shape[0])
        if bool(np.isin(grid[r, :], DIVIDER_COLORS).all())
    ]


def _separator_cols(grid: np.ndarray) -> list[int]:
    return [
        int(c)
        for c in range(grid.shape[1])
        if bool(np.isin(grid[:, c], DIVIDER_COLORS).all())
    ]


def _majority_divider(values: np.ndarray) -> int:
    grey_count = int(np.count_nonzero(values == GREY))
    maroon_count = int(np.count_nonzero(values == MAROON))
    assert grey_count != maroon_count, "divider line color must have a unique majority"
    return GREY if grey_count > maroon_count else MAROON


def _regions(
    shape: tuple[int, int],
    row_lines: list[int],
    col_lines: list[int],
) -> list[tuple[int, int, int, int, int | None, int | None, int | None, int | None]]:
    h, w = map(int, shape)
    row_cuts = [-1] + [int(r) for r in row_lines] + [h]
    col_cuts = [-1] + [int(c) for c in col_lines] + [w]

    regions = []
    for i in range(len(row_cuts) - 1):
        r0 = int(row_cuts[i] + 1)
        r1 = int(row_cuts[i + 1] - 1)
        if r0 > r1:
            continue
        for j in range(len(col_cuts) - 1):
            c0 = int(col_cuts[j] + 1)
            c1 = int(col_cuts[j + 1] - 1)
            if c0 > c1:
                continue
            top = None if row_cuts[i] == -1 else int(row_cuts[i])
            bottom = None if row_cuts[i + 1] == h else int(row_cuts[i + 1])
            left = None if col_cuts[j] == -1 else int(col_cuts[j])
            right = None if col_cuts[j + 1] == w else int(col_cuts[j + 1])
            regions.append((r0, r1, c0, c1, top, bottom, left, right))
    return regions


def _foreground_stats(region: np.ndarray) -> list[tuple[int, int, tuple[int, int]]]:
    stats = []
    for color in sorted(int(v) for v in np.unique(region)):
        if color in (BG, GREY, MAROON):
            continue
        coords = np.argwhere(region == color)
        assert coords.size > 0
        anchor = min((int(r), int(c)) for r, c in coords)
        stats.append((color, int(coords.shape[0]), anchor))
    assert len(stats) == 2, "each region must contain exactly two foreground colors"
    return stats


def _winner_color(stats: list[tuple[int, int, tuple[int, int]]], use_majority: bool) -> int:
    color_a, count_a, anchor_a = stats[0]
    color_b, count_b, anchor_b = stats[1]

    if count_a == count_b:
        return color_a if anchor_a < anchor_b else color_b
    if use_majority:
        return color_a if count_a > count_b else color_b
    return color_a if count_a < count_b else color_b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each divider-bounded region with the foreground color that wins its local vote.

    Input structure:
    - The background is black.
    - One-cell-wide grey and maroon divider lines run across whole rows and/or columns, partitioning the grid into rectangular regions.
    - Inside every region, black background surrounds scattered pixels of exactly two other colors.

    Rule:
    - Keep all divider pixels unchanged and process each rectangular region independently.
    - Look at the divider colors on the region's present sides; the unique majority divider color selects the vote rule for that region.
    - Grey means the region uses the foreground majority color, while maroon means it uses the foreground minority color.
    - If the two foreground colors have equal counts, choose the one whose nearest pixel is closest to the region's top-left corner.
    - Repaint the entire region interior with the winning foreground color.

    Key insight:
    - The colored pixels inside a region are only votes; the surrounding divider color pattern secretly decides whether that region rewards the local majority or the local minority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_lines = _separator_rows(g)
    col_lines = _separator_cols(g)
    assert row_lines or col_lines, "expected at least one divider line"

    row_colors = {int(r): _majority_divider(g[int(r), :]) for r in row_lines}
    col_colors = {int(c): _majority_divider(g[:, int(c)]) for c in col_lines}

    out = g.copy()
    for r0, r1, c0, c1, top, bottom, left, right in _regions(g.shape, row_lines, col_lines):
        sides = []
        if top is not None:
            sides.append(row_colors[top])
        if bottom is not None:
            sides.append(row_colors[bottom])
        if left is not None:
            sides.append(col_colors[left])
        if right is not None:
            sides.append(col_colors[right])
        assert sides, "each region must touch at least one divider"

        grey_count = sum(color == GREY for color in sides)
        maroon_count = sum(color == MAROON for color in sides)
        assert grey_count != maroon_count, "region divider colors must have a unique majority"

        use_majority = grey_count > maroon_count
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        winner = _winner_color(_foreground_stats(region), use_majority=use_majority)
        out[r0 : r1 + 1, c0 : c1 + 1] = winner

    return out


def _sample_sizes(count: int, lo: int, hi: int, limit: int) -> list[int] | None:
    for _ in range(SIZE_RETRIES):
        sizes = [random.randint(lo, hi) for _ in range(count)]
        if sum(sizes) + (count - 1) <= limit:
            return sizes
    return None


def _long_separator_pattern(count: int, outer_color: int, inner_color: int) -> list[int]:
    assert count >= 5
    seps = [outer_color] * (count - 1)
    run_len = random.choice([2, 3]) if count >= 6 else 2
    run_start = random.randint(1, len(seps) - run_len - 1)
    for idx in range(run_start, run_start + run_len):
        seps[idx] = inner_color
    return seps


def _count_options(area: int, relation: str) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    if relation == "tie":
        for n in (2, 3, 4):
            if 2 * n <= area - 2:
                options.append((n, n))
    elif relation == "a":
        for ca, cb in ((3, 2), (4, 2), (4, 3), (5, 3)):
            if ca + cb <= area - 2:
                options.append((ca, cb))
    else:
        assert relation == "b"
        for ca, cb in ((2, 3), (2, 4), (3, 4), (3, 5)):
            if ca + cb <= area - 2:
                options.append((ca, cb))
    assert options, f"region area {area} is too small for relation {relation}"
    return options


def _paint_region_votes(
    patch: np.ndarray,
    color_a: int,
    color_b: int,
    relation: str,
) -> None:
    h, w = map(int, patch.shape)
    area = int(h * w)
    count_a, count_b = random.choice(_count_options(area, relation))
    all_cells = [(r, c) for r in range(h) for c in range(w)]

    if relation == "tie":
        tie_winner = random.choice([color_a, color_b])
        earliest_idx = random.randint(0, area - (count_a + count_b))
        earliest = all_cells[earliest_idx]
        remaining = random.sample(all_cells[earliest_idx + 1 :], count_a + count_b - 1)
        if tie_winner == color_a:
            a_cells = [earliest] + random.sample(remaining, count_a - 1)
            b_cells = [cell for cell in remaining if cell not in set(a_cells[1:])]
        else:
            b_cells = [earliest] + random.sample(remaining, count_b - 1)
            a_cells = [cell for cell in remaining if cell not in set(b_cells[1:])]
    else:
        chosen = random.sample(all_cells, count_a + count_b)
        random.shuffle(chosen)
        a_cells = chosen[:count_a]
        b_cells = chosen[count_a:]

    for r, c in a_cells:
        patch[r, c] = color_a
    for r, c in b_cells:
        patch[r, c] = color_b


def _region_relations(count: int) -> list[str]:
    relations = (["tie", "a", "b"] * ((count + 2) // 3))[:count]
    random.shuffle(relations)
    return relations


def _build_candidate() -> np.ndarray | None:
    orientation = random.choice(["wide", "tall"])
    outer_color = random.choice(DIVIDER_COLORS)
    inner_color = MAROON if outer_color == GREY else GREY

    if orientation == "wide":
        row_heights = _sample_sizes(2, 4, 8, 30)
        col_count = random.choice([5, 6])
        col_widths = _sample_sizes(col_count, 3, 5, 30)
        if row_heights is None or col_widths is None:
            return None
        row_seps = [outer_color]
        col_seps = _long_separator_pattern(col_count, outer_color, inner_color)
    else:
        col_widths = _sample_sizes(2, 4, 8, 30)
        row_count = random.choice([5, 6])
        row_heights = _sample_sizes(row_count, 3, 5, 30)
        if row_heights is None or col_widths is None:
            return None
        row_seps = _long_separator_pattern(row_count, outer_color, inner_color)
        col_seps = [outer_color]

    height = sum(row_heights) + len(row_seps)
    width = sum(col_widths) + len(col_seps)
    assert height <= 30 and width <= 30

    grid = np.full((height, width), BG, dtype=int)

    row_line_idxs: list[int] = []
    row_starts: list[int] = []
    pos = 0
    for i, size in enumerate(row_heights):
        row_starts.append(pos)
        pos += size
        if i < len(row_seps):
            grid[pos, :] = row_seps[i]
            row_line_idxs.append(pos)
            pos += 1

    col_line_idxs: list[int] = []
    col_starts: list[int] = []
    pos = 0
    for i, size in enumerate(col_widths):
        col_starts.append(pos)
        pos += size
        if i < len(col_seps):
            grid[:, pos] = col_seps[i]
            col_line_idxs.append(pos)
            pos += 1

    # Paint columns last so intersections inherit the column color; main recovers each line color by majority.
    for idx, color in zip(col_line_idxs, col_seps, strict=False):
        grid[:, idx] = color

    region_boxes = []
    for r0, rh in zip(row_starts, row_heights, strict=False):
        for c0, cw in zip(col_starts, col_widths, strict=False):
            region_boxes.append((r0, r0 + rh - 1, c0, c0 + cw - 1))

    relations = _region_relations(len(region_boxes))
    for relation, (r0, r1, c0, c1) in zip(relations, region_boxes, strict=False):
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        color_a, color_b = random.sample(FILL_COLORS, 2)
        _paint_region_votes(patch, color_a, color_b, relation)

    try:
        row_lines = _separator_rows(grid)
        col_lines = _separator_cols(grid)
        assert len(row_lines) == len(row_seps)
        assert len(col_lines) == len(col_seps)

        winners = set()
        for r0, r1, c0, c1, top, bottom, left, right in _regions(grid.shape, row_lines, col_lines):
            sides = []
            if top is not None:
                sides.append(_majority_divider(grid[top, :]))
            if bottom is not None:
                sides.append(_majority_divider(grid[bottom, :]))
            if left is not None:
                sides.append(_majority_divider(grid[:, left]))
            if right is not None:
                sides.append(_majority_divider(grid[:, right]))
            grey_count = sum(color == GREY for color in sides)
            maroon_count = sum(color == MAROON for color in sides)
            assert grey_count != maroon_count
            winners.add(GREY if grey_count > maroon_count else MAROON)
            stats = _foreground_stats(grid[r0 : r1 + 1, c0 : c1 + 1])
            assert len(stats) == 2
        assert winners == {GREY, MAROON}
        main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
