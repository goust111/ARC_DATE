from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BG = Color.BLACK
PALETTE_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GRAY)
MARKER_COLOR = Color.ORANGE
POS_COLOR = Color.TEAL
NEG_COLOR = Color.RED

GRID_HEIGHT_RANGE = (15, 24)
GRID_WIDTH_RANGE = (15, 24)
MARKER_HEIGHT_RANGE = (2, 4)
BAR_HEIGHT_RANGE = (1, 6)
POS_BAR_COUNT_RANGE = (2, 4)
NEG_BAR_COUNT_RANGE = (1, 3)
GENERATION_ATTEMPTS = 600


def _find_gap_col(palette: np.ndarray) -> int:
    gap_cols = np.flatnonzero(np.asarray(palette, dtype=int) == int(BG))
    assert gap_cols.size == 1, "palette row must contain exactly one black gap"
    return int(gap_cols[0])


def _scan_marker_height(grid: np.ndarray, gap_col: int) -> int:
    height = 0
    for r in range(1, int(grid.shape[0])):
        if int(grid[r, gap_col]) != int(MARKER_COLOR):
            break
        height += 1
    assert MARKER_HEIGHT_RANGE[0] <= height <= MARKER_HEIGHT_RANGE[1]
    return int(height)


def _scan_bottom_bars(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = map(int, grid.shape)
    bars: list[tuple[int, int, int]] = []
    for c in range(w):
        color = int(grid[h - 1, c])
        if color not in (int(POS_COLOR), int(NEG_COLOR)):
            continue
        r = h - 1
        while r >= 1 and int(grid[r, c]) == color:
            r -= 1
        bars.append((int(c), int(color), int(h - 1 - r)))
    return bars


def _extract_input_spec(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    bg = int(get_bg(g))
    assert bg == int(BG), "background must be black"

    palette = g[0].copy()
    gap_col = _find_gap_col(palette)
    non_gap_mask = np.ones(w, dtype=bool)
    non_gap_mask[gap_col] = False
    assert np.all(np.isin(palette[non_gap_mask], PALETTE_COLORS))
    assert len({int(x) for x in palette[non_gap_mask]}) >= 3

    marker_height = _scan_marker_height(g, gap_col)

    lower = g[1:, :]
    components = find_components(
        lower,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[MARKER_COLOR, POS_COLOR, NEG_COLOR],
    )

    marker_components = [comp for comp in components if int(comp["color"]) == int(MARKER_COLOR)]
    assert len(marker_components) == 1
    marker_box = tuple(map(int, marker_components[0]["bbox"]))
    assert marker_box == (0, gap_col, marker_height - 1, gap_col)
    assert int(marker_components[0]["area"]) == int(marker_height)

    bar_components = [comp for comp in components if int(comp["color"]) in (int(POS_COLOR), int(NEG_COLOR))]
    assert bar_components, "at least one bottom bar is required"

    bars = _scan_bottom_bars(g)
    bars_from_components: list[tuple[int, int, int]] = []
    for comp in bar_components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert c0 == c1, "bars must be single-column"
        assert int(comp["area"]) == r1 - r0 + 1
        assert r1 == h - 2, "bars must touch the bottom edge"
        bars_from_components.append((c0, int(comp["color"]), int(comp["area"])))
    bars.sort()
    bars_from_components.sort()
    assert bars == bars_from_components

    cols = [col for col, _color, _height in bars]
    assert all(cols[i + 1] - cols[i] >= 2 for i in range(len(cols) - 1)), "bars may not be adjacent"

    pos_heights = [height for _col, color, height in bars if color == int(POS_COLOR)]
    neg_heights = [height for _col, color, height in bars if color == int(NEG_COLOR)]
    assert pos_heights and neg_heights

    bar_tops = [h - height for _col, _color, height in bars]
    marker_bottom = int(marker_height)
    assert min(bar_tops) - marker_bottom - 1 >= 3

    sum_pos = int(sum(pos_heights))
    sum_neg = int(sum(neg_heights))
    depth = int(sum_pos - sum_neg)
    available_rows = int(h - 1 - marker_height)
    assert 1 <= depth <= available_rows

    return {
        "shape": (h, w),
        "palette": palette,
        "gap_col": int(gap_col),
        "marker_height": int(marker_height),
        "bars": bars,
        "pos_heights": pos_heights,
        "neg_heights": neg_heights,
        "sum_pos": int(sum_pos),
        "sum_neg": int(sum_neg),
        "depth": int(depth),
        "available_rows": int(available_rows),
    }


def _render_wedge(shape: tuple[int, int], palette: np.ndarray, gap_col: int, depth: int) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), int(BG), dtype=int)
    out[0, :] = np.asarray(palette, dtype=int)
    for k in range(1, int(depth) + 1):
        out[k, gap_col] = int(MARKER_COLOR)
        left = max(0, gap_col - k)
        right = min(w, gap_col + k + 1)
        if left < gap_col:
            out[k, left:gap_col] = palette[left:gap_col]
        if gap_col + 1 < right:
            out[k, gap_col + 1 : right] = palette[gap_col + 1 : right]
    return out


def _output_matches_spec(output: np.ndarray, spec: dict) -> bool:
    out = np.asarray(output, dtype=int)
    h, w = spec["shape"]
    if out.shape != (h, w):
        return False
    if not np.array_equal(out[0], spec["palette"]):
        return False
    if spec["depth"] + 1 < h and np.any(out[spec["depth"] + 1 :, :] != int(BG)):
        return False
    for k in range(1, spec["depth"] + 1):
        expected = np.full(w, int(BG), dtype=int)
        expected[spec["gap_col"]] = int(MARKER_COLOR)
        left = max(0, spec["gap_col"] - k)
        right = min(w, spec["gap_col"] + k + 1)
        if left < spec["gap_col"]:
            expected[left : spec["gap_col"]] = spec["palette"][left : spec["gap_col"]]
        if spec["gap_col"] + 1 < right:
            expected[spec["gap_col"] + 1 : right] = spec["palette"][spec["gap_col"] + 1 : right]
        if not np.array_equal(out[k], expected):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the signed bottom-bar height arithmetic to control how many wedge rows expand under the palette gap.

    Input structure:
    - Black background with a full-width palette row at the top, using colors from blue/red/green/yellow/gray and containing exactly one black gap.
    - An orange vertical marker bar sits directly below the gap column.
    - After a blank separator, the lower part contains non-adjacent bottom-aligned single-column bars: teal bars contribute positively and red bars contribute negatively.

    Rule:
    - Sum the heights of all teal bars and all red bars, then compute the signed depth `H = S_pos - S_neg`.
    - Erase everything except the top palette row.
    - Starting on the next row, build exactly `H` wedge rows centered on the gap column: row `k` copies the nearest `k` palette columns on each side, clipped at the palette boundaries, and the gap column itself is always orange.
    - Leave every row below the `H`-th wedge row black.

    Key insight:
    - The lower bars are not redrawn; their signed height arithmetic is an implicit numeric parameter that determines the depth of an unrelated geometric wedge expansion under the top-row gap.
    """
    spec = _extract_input_spec(np.asarray(grid, dtype=int))
    return _render_wedge(spec["shape"], spec["palette"], spec["gap_col"], spec["depth"])


def _sample_gap_col(width: int) -> tuple[int, str]:
    candidates: dict[str, list[int]] = {
        "edge": [],
        "near": [],
        "interior": [],
    }
    for c in range(width):
        dist = min(c, width - 1 - c)
        if dist == 0:
            candidates["edge"].append(c)
        elif dist <= 2:
            candidates["near"].append(c)
        else:
            candidates["interior"].append(c)

    modes = ["interior"] * 5 + ["near"] * 3 + ["edge"] * 2
    modes = [mode for mode in modes if candidates[mode]]
    mode = random.choice(modes)
    return int(random.choice(candidates[mode])), mode


def _partition_positions(count: int, parts: int) -> list[int]:
    if parts == 1:
        return [count]
    cuts = sorted(random.sample(range(1, count), parts - 1))
    edges = [0] + cuts + [count]
    return [edges[i + 1] - edges[i] for i in range(parts)]


def _generate_palette_row(width: int, gap_col: int) -> np.ndarray:
    include_red = random.random() < 0.7
    if include_red:
        color_count = random.randint(3, 5)
        chosen = [int(NEG_COLOR)]
        remaining = [int(c) for c in PALETTE_COLORS if int(c) != int(NEG_COLOR)]
        chosen.extend(random.sample(remaining, color_count - 1))
    else:
        color_count = random.randint(3, 4)
        chosen = random.sample([int(c) for c in PALETTE_COLORS if int(c) != int(NEG_COLOR)], color_count)
    random.shuffle(chosen)

    non_gap_count = width - 1
    max_runs = min(8, non_gap_count)
    min_runs = min(max_runs, max(4, len(chosen)))
    run_count = random.randint(min_runs, max_runs)
    run_lengths = _partition_positions(non_gap_count, run_count)

    run_colors = chosen[:]
    random.shuffle(run_colors)
    while len(run_colors) < run_count:
        previous = run_colors[-1]
        options = [color for color in chosen if color != previous]
        run_colors.append(int(random.choice(options)))

    palette = np.full(width, int(BG), dtype=int)
    active_cols = [c for c in range(width) if c != gap_col]
    offset = 0
    for run_len, color in zip(run_lengths, run_colors):
        cols = active_cols[offset : offset + run_len]
        palette[cols] = int(color)
        offset += run_len

    unique_colors = {int(x) for x in palette if int(x) != int(BG)}
    assert len(unique_colors) >= 3
    return palette


def _sample_bar_columns(width: int, gap_col: int, total_bars: int) -> list[int] | None:
    candidates = [c for c in range(width) if c != gap_col]
    for _ in range(200):
        cols = sorted(random.sample(candidates, total_bars))
        if all(cols[i + 1] - cols[i] >= 2 for i in range(total_bars - 1)):
            random.shuffle(cols)
            return [int(c) for c in cols]
    return None


def _heights_are_non_degenerate(
    pos_heights: list[int],
    neg_heights: list[int],
    depth: int,
    marker_height: int,
) -> bool:
    if not pos_heights or not neg_heights:
        return False
    if depth <= 0:
        return False
    if len(pos_heights) < 2:
        return False
    shortcut_values = {
        len(pos_heights),
        len(neg_heights),
        len(pos_heights) + len(neg_heights),
        marker_height,
        max(pos_heights),
        max(neg_heights),
    }
    if depth in shortcut_values:
        return False
    if any(height == depth for height in pos_heights + neg_heights):
        return False
    if sum(pos_heights) <= sum(neg_heights):
        return False
    return True


def _sample_bar_heights(available_rows: int, marker_height: int) -> tuple[list[int], list[int], int] | None:
    for _ in range(300):
        pos_count = random.randint(*POS_BAR_COUNT_RANGE)
        neg_count = random.randint(*NEG_BAR_COUNT_RANGE)
        pos_heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(pos_count)]
        neg_heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(neg_count)]
        depth = int(sum(pos_heights) - sum(neg_heights))
        if not (1 <= depth <= available_rows):
            continue
        if depth == 1 and random.random() < 0.8:
            continue
        if not _heights_are_non_degenerate(pos_heights, neg_heights, depth, marker_height):
            continue
        return pos_heights, neg_heights, depth
    return None


def _spec_is_informative(spec: dict, gap_mode: str) -> bool:
    depth = int(spec["depth"])
    gap_col = int(spec["gap_col"])
    width = int(spec["shape"][1])
    min_edge_distance = min(gap_col, width - 1 - gap_col)
    if gap_mode in {"edge", "near"} and depth <= min_edge_distance:
        return False

    left = np.asarray(spec["palette"][:gap_col], dtype=int)
    right = np.asarray(spec["palette"][gap_col + 1 :], dtype=int)
    overlap = min(left.size, right.size)
    if overlap > 0 and np.array_equal(left[-overlap:], right[:overlap]):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    marker_height = random.randint(*MARKER_HEIGHT_RANGE)
    gap_col, gap_mode = _sample_gap_col(w)
    palette = _generate_palette_row(w, gap_col)
    available_rows = h - 1 - marker_height

    sampled = _sample_bar_heights(available_rows, marker_height)
    if sampled is None:
        return None
    pos_heights, neg_heights, _depth = sampled

    bar_columns = _sample_bar_columns(w, gap_col, len(pos_heights) + len(neg_heights))
    if bar_columns is None:
        return None

    bar_specs = [(int(POS_COLOR), height) for height in pos_heights]
    bar_specs.extend((int(NEG_COLOR), height) for height in neg_heights)
    random.shuffle(bar_specs)

    grid = np.full((h, w), int(BG), dtype=int)
    grid[0, :] = palette
    grid[1 : 1 + marker_height, gap_col] = int(MARKER_COLOR)
    for col, (color, height) in zip(bar_columns, bar_specs):
        grid[h - height :, col] = int(color)

    try:
        spec = _extract_input_spec(grid)
    except AssertionError:
        return None
    if not _spec_is_informative(spec, gap_mode):
        return None

    expected = _render_wedge(spec["shape"], spec["palette"], spec["gap_col"], spec["depth"])
    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    if not _output_matches_spec(produced, spec):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wedge-expansion arithmetic input")
