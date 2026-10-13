from __future__ import annotations

import random
from collections import Counter

import numpy as np


COLORS = tuple(range(10))
LINE_COLOR_HINT = 5
GEN_RETRIES = 400
LAYOUT_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3))
REGION_SIZE_RANGE = (4, 7)
VOTE_COLOR_COUNT = 5


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _line_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    best_score = None
    best_color = None
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        row_hits = sum(int(np.count_nonzero(grid[r] == color) > w // 2) for r in range(h))
        col_hits = sum(int(np.count_nonzero(grid[:, c] == color) > h // 2) for c in range(w))
        score = (int(row_hits > 0 and col_hits > 0), row_hits + col_hits, np.count_nonzero(grid == color))
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
    assert best_score is not None and best_score[0] == 1, "expected grid separators in both axes"
    return int(best_color)


def _separator_indices(grid: np.ndarray, line_color: int) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.count_nonzero(grid[r] == line_color) > w // 2]
    sep_cols = [c for c in range(w) if np.count_nonzero(grid[:, c] == line_color) > h // 2]
    assert sep_rows and sep_cols, "expected at least one separator row and one separator column"
    return sep_rows, sep_cols


def _regions_from_grid(
    shape: tuple[int, int], sep_rows: list[int], sep_cols: list[int]
) -> tuple[list[dict[str, int]], np.ndarray]:
    h, w = shape
    row_bounds = [-1, *sep_rows, h]
    col_bounds = [-1, *sep_cols, w]
    regions: list[dict[str, int]] = []
    region_ids = -np.ones(shape, dtype=int)

    region_id = 0
    for r0_bound, r1_bound in zip(row_bounds, row_bounds[1:]):
        r0 = r0_bound + 1
        r1 = r1_bound - 1
        if r0 > r1:
            continue
        for c0_bound, c1_bound in zip(col_bounds, col_bounds[1:]):
            c0 = c0_bound + 1
            c1 = c1_bound - 1
            if c0 > c1:
                continue
            regions.append({"id": region_id, "r0": r0, "r1": r1, "c0": c0, "c1": c1})
            region_ids[r0 : r1 + 1, c0 : c1 + 1] = region_id
            region_id += 1

    assert regions, "expected at least one region"
    return regions, region_ids


def _collect_vetoes(
    grid: np.ndarray,
    bg: int,
    line_color: int,
    sep_rows: list[int],
    sep_cols: list[int],
    region_ids: np.ndarray,
    region_count: int,
) -> tuple[list[set[int]], list[dict[str, object]]]:
    h, w = grid.shape
    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)
    vetoes = [set() for _ in range(region_count)]
    markers: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg, line_color):
                continue
            on_row = r in sep_row_set
            on_col = c in sep_col_set
            if on_row and not on_col:
                upper = int(region_ids[r - 1, c]) if r > 0 else -1
                lower = int(region_ids[r + 1, c]) if r + 1 < h else -1
                if upper >= 0 and lower >= 0:
                    vetoes[upper].add(color)
                    vetoes[lower].add(color)
                    markers.append(
                        {"cell": (r, c), "orientation": "h", "color": color, "regions": (upper, lower)}
                    )
            elif on_col and not on_row:
                left = int(region_ids[r, c - 1]) if c > 0 else -1
                right = int(region_ids[r, c + 1]) if c + 1 < w else -1
                if left >= 0 and right >= 0:
                    vetoes[left].add(color)
                    vetoes[right].add(color)
                    markers.append(
                        {"cell": (r, c), "orientation": "v", "color": color, "regions": (left, right)}
                    )
    return vetoes, markers


def _region_vote_summary(
    grid: np.ndarray, region: dict[str, int], bg: int, line_color: int, veto_colors: set[int]
) -> tuple[Counter, dict[int, tuple[int, int]]]:
    counts: Counter = Counter()
    first_pos: dict[int, tuple[int, int]] = {}
    for r in range(region["r0"], region["r1"] + 1):
        for c in range(region["c0"], region["c1"] + 1):
            color = int(grid[r, c])
            if color == bg:
                continue
            assert color != line_color, "separator color may not appear inside a region"
            if color in veto_colors:
                continue
            counts[color] += 1
            first_pos.setdefault(color, (r, c))
    return counts, first_pos


def _region_winner(
    grid: np.ndarray, region: dict[str, int], bg: int, line_color: int, veto_colors: set[int]
) -> int:
    counts, first_pos = _region_vote_summary(grid, region, bg, line_color, veto_colors)
    assert counts, "each region must keep at least one non-vetoed vote"
    best_count = max(counts.values())
    candidates = [color for color, count in counts.items() if count == best_count]
    return min(candidates, key=lambda color: first_pos[color])


def _parse_grid(grid: np.ndarray) -> tuple[int, int, list[dict[str, int]], list[set[int]], list[dict[str, object]]]:
    bg = _mode_color(grid)
    line_color = _line_color(grid, bg)
    sep_rows, sep_cols = _separator_indices(grid, line_color)
    regions, region_ids = _regions_from_grid(grid.shape, sep_rows, sep_cols)
    vetoes, markers = _collect_vetoes(grid, bg, line_color, sep_rows, sep_cols, region_ids, len(regions))
    return bg, line_color, regions, vetoes, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each table region with its post-veto majority vote color.

    Input structure:
    - The most common color is the background.
    - Single-pixel-thick gray separator rows and columns split the grid into rectangular regions.
    - Each region contains scattered colored vote pixels on top of the background.
    - Some internal separator cells are replaced by a colored veto marker between two adjacent regions.

    Rule:
    - For each veto marker, ignore all votes of that marker color in the two regions touching that boundary segment.
    - Count the remaining votes inside each region; if several colors tie, choose the color whose surviving vote appears first in row-major order.
    - Recolor every background cell in the region with that chosen majority color.
    - Keep the original vote pixels, separator lines, and veto markers unchanged.

    Key insight: The boundary markers do not paint regions directly; they first cancel one color in two neighboring tallies, and only then does each region perform its own majority fill.
    """
    g = np.asarray(grid, dtype=int)
    bg, line_color, regions, vetoes, _markers = _parse_grid(g)

    out = g.copy()
    for region in regions:
        winner = _region_winner(g, region, bg, line_color, vetoes[region["id"]])
        patch = out[region["r0"] : region["r1"] + 1, region["c0"] : region["c1"] + 1]
        patch[g[region["r0"] : region["r1"] + 1, region["c0"] : region["c1"] + 1] == bg] = winner
    return out


def _choose_layout(rng: random.Random) -> tuple[list[int], list[int]]:
    region_rows, region_cols = rng.choice(LAYOUT_CHOICES)
    row_sizes = [rng.randint(*REGION_SIZE_RANGE) for _ in range(region_rows)]
    col_sizes = [rng.randint(*REGION_SIZE_RANGE) for _ in range(region_cols)]
    height = sum(row_sizes) + region_rows - 1
    width = sum(col_sizes) + region_cols - 1
    assert height <= 30 and width <= 30
    return row_sizes, col_sizes


def _build_layout(
    row_sizes: list[int], col_sizes: list[int]
) -> tuple[int, int, list[dict[str, int]], list[dict[str, object]], list[dict[str, object]]]:
    row_starts: list[int] = []
    row_ends: list[int] = []
    boundary_rows: list[int] = []
    cursor = 0
    for i, size in enumerate(row_sizes):
        row_starts.append(cursor)
        row_ends.append(cursor + size - 1)
        cursor += size
        if i < len(row_sizes) - 1:
            boundary_rows.append(cursor)
            cursor += 1
    height = cursor

    col_starts: list[int] = []
    col_ends: list[int] = []
    boundary_cols: list[int] = []
    cursor = 0
    for j, size in enumerate(col_sizes):
        col_starts.append(cursor)
        col_ends.append(cursor + size - 1)
        cursor += size
        if j < len(col_sizes) - 1:
            boundary_cols.append(cursor)
            cursor += 1
    width = cursor

    regions: list[dict[str, int]] = []
    region_lookup: dict[tuple[int, int], int] = {}
    region_id = 0
    for i, (r0, r1) in enumerate(zip(row_starts, row_ends)):
        for j, (c0, c1) in enumerate(zip(col_starts, col_ends)):
            regions.append({"id": region_id, "r0": r0, "r1": r1, "c0": c0, "c1": c1, "ri": i, "ci": j})
            region_lookup[(i, j)] = region_id
            region_id += 1

    h_segments: list[dict[str, object]] = []
    for i, boundary_row in enumerate(boundary_rows):
        for j, (c0, c1) in enumerate(zip(col_starts, col_ends)):
            cells = [(boundary_row, c) for c in range(c0, c1 + 1)]
            marker_cells = cells[1:-1] if len(cells) > 2 else cells
            h_segments.append(
                {
                    "key": ("h", i, j),
                    "orientation": "h",
                    "regions": (region_lookup[(i, j)], region_lookup[(i + 1, j)]),
                    "marker_cells": marker_cells,
                }
            )

    v_segments: list[dict[str, object]] = []
    for j, boundary_col in enumerate(boundary_cols):
        for i, (r0, r1) in enumerate(zip(row_starts, row_ends)):
            cells = [(r, boundary_col) for r in range(r0, r1 + 1)]
            marker_cells = cells[1:-1] if len(cells) > 2 else cells
            v_segments.append(
                {
                    "key": ("v", i, j),
                    "orientation": "v",
                    "regions": (region_lookup[(i, j)], region_lookup[(i, j + 1)]),
                    "marker_cells": marker_cells,
                }
            )

    return height, width, regions, h_segments, v_segments


def _choose_markers(
    rng: random.Random,
    h_segments: list[dict[str, object]],
    v_segments: list[dict[str, object]],
    vote_palette: list[int],
) -> tuple[list[dict[str, object]], tuple[int, int], int]:
    mode = rng.choice(("h", "v", "both"))
    all_segments = [*h_segments, *v_segments]

    if mode == "h":
        selected = [rng.choice(h_segments)]
    elif mode == "v":
        selected = [rng.choice(v_segments)]
    else:
        first = rng.choice(all_segments)
        opposite_pool = v_segments if first["orientation"] == "h" else h_segments
        selected = [first, rng.choice(opposite_pool)]

    used_keys = {segment["key"] for segment in selected}
    remaining = [segment for segment in all_segments if segment["key"] not in used_keys]
    rng.shuffle(remaining)
    extra_limit = 1
    extra_count = rng.randint(0, extra_limit)
    selected.extend(remaining[:extra_count])

    marker_colors = rng.sample(vote_palette, len(selected))
    markers: list[dict[str, object]] = []
    for segment, color in zip(selected, marker_colors):
        markers.append(
            {
                "cell": rng.choice(segment["marker_cells"]),
                "orientation": segment["orientation"],
                "regions": tuple(segment["regions"]),
                "color": int(color),
            }
        )
    active_regions = tuple(selected[0]["regions"])
    active_color = int(marker_colors[0])
    return markers, active_regions, active_color


def _plan_region_counts(
    rng: random.Random,
    palette: list[int],
    veto_colors: set[int],
    force_change_color: int | None,
    want_tie: bool,
) -> tuple[Counter, tuple[int, int] | None] | None:
    allowed = [color for color in palette if color not in veto_colors]
    if len(allowed) < 2:
        return None

    if want_tie:
        a, b = rng.sample(allowed, 2)
        preferred = rng.choice((a, b))
        other = b if preferred == a else a
        counts: Counter = Counter({a: 2, b: 2})
        spare = [color for color in allowed if color not in {a, b}]
        if spare and rng.random() < 0.5:
            counts[rng.choice(spare)] = 1
        if force_change_color is not None:
            assert force_change_color in veto_colors
            counts[force_change_color] = 4
        elif veto_colors and rng.random() < 0.5:
            counts[rng.choice(sorted(veto_colors))] = 1
        return counts, (preferred, other)

    winner, runner = rng.sample(allowed, 2)
    counts = Counter({winner: rng.randint(3, 4), runner: 2})
    spare = [color for color in allowed if color not in {winner, runner}]
    if spare and rng.random() < 0.5:
        counts[rng.choice(spare)] = 1
    if force_change_color is not None:
        assert force_change_color in veto_colors
        counts[force_change_color] = counts[winner] + 1
    elif veto_colors and rng.random() < 0.5:
        counts[rng.choice(sorted(veto_colors))] = 1
    return counts, None


def _paint_region_votes(
    rng: random.Random,
    grid: np.ndarray,
    region: dict[str, int],
    counts: Counter,
    tie_priority: tuple[int, int] | None,
) -> None:
    cells = [(r, c) for r in range(region["r0"], region["r1"] + 1) for c in range(region["c0"], region["c1"] + 1)]
    total_votes = sum(counts.values())
    assert total_votes <= len(cells) - 4, "regions must keep visible background"
    chosen = sorted(rng.sample(cells, total_votes))

    remaining = Counter(counts)
    assignments: list[int] = []
    if tie_priority is not None:
        first_color, second_color = tie_priority
        assignments.extend([first_color, second_color])
        remaining[first_color] -= 1
        remaining[second_color] -= 1
        if remaining[first_color] == 0:
            del remaining[first_color]
        if remaining.get(second_color, 0) == 0:
            remaining.pop(second_color, None)

    tokens: list[int] = []
    for color, count in remaining.items():
        tokens.extend([int(color)] * int(count))
    rng.shuffle(tokens)
    assignments.extend(tokens)

    for (r, c), color in zip(chosen, assignments):
        grid[r, c] = int(color)


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[bg] > max((count for color, count in counts.items() if color != bg), default=0)


def _winner_without_veto(grid: np.ndarray, region: dict[str, int], bg: int, line_color: int) -> int:
    return _region_winner(grid, region, bg, line_color, set())


def _active_veto_changes_some_region(
    grid: np.ndarray, active_regions: tuple[int, int]
) -> bool:
    bg, line_color, regions, vetoes, _markers = _parse_grid(grid)
    for region_id in active_regions:
        region = regions[region_id]
        before = _winner_without_veto(grid, region, bg, line_color)
        after = _region_winner(grid, region, bg, line_color, vetoes[region_id])
        if before != after:
            return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_sizes, col_sizes = _choose_layout(rng)
    height, width, regions, h_segments, v_segments = _build_layout(row_sizes, col_sizes)

    bg = rng.choice([color for color in COLORS if color != LINE_COLOR_HINT])
    line_color = LINE_COLOR_HINT
    vote_palette = rng.sample([color for color in COLORS if color not in {bg, line_color}], VOTE_COLOR_COUNT)

    grid = np.full((height, width), bg, dtype=int)
    for segment in h_segments:
        r = segment["marker_cells"][0][0]
        grid[r, :] = line_color
    for segment in v_segments:
        c = segment["marker_cells"][0][1]
        grid[:, c] = line_color

    markers, active_regions, active_color = _choose_markers(rng, h_segments, v_segments, vote_palette)
    vetoes = [set() for _ in regions]
    for marker in markers:
        for region_id in marker["regions"]:
            vetoes[int(region_id)].add(int(marker["color"]))
        r, c = marker["cell"]
        grid[r, c] = int(marker["color"])

    tie_region = None
    if rng.random() < 0.5:
        region_ids = [region["id"] for region in regions if region["id"] not in active_regions]
        tie_region = rng.choice(region_ids or [region["id"] for region in regions])

    for region in regions:
        force_change = active_color if region["id"] in active_regions else None
        plan = _plan_region_counts(
            rng,
            vote_palette,
            vetoes[region["id"]],
            force_change,
            region["id"] == tie_region,
        )
        if plan is None:
            return None
        counts, tie_priority = plan
        _paint_region_votes(rng, grid, region, counts, tie_priority)

    if not _bg_is_dominant(grid, bg):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not _active_veto_changes_some_region(grid, active_regions):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
