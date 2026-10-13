import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components


BAR_COUNT_RANGE = (5, 8)  # numpy high-exclusive: 5..7 bars
BAR_HEIGHT_CHOICES = (1, 2)
BAR_GAP_RANGE = (1, 3)  # 1..2 blank rows
GRID_MARGIN_RANGE = (1, 3)
WIDTH_EXTRA_RANGE = (2, 6)
GRID_WIDTH_PADDING_RANGE = (3, 9)
GENERATION_RETRIES = 500

_GEN_CALL_COUNT = 0


def _assert_solid_horizontal_bar(grid: np.ndarray, obj: dict) -> None:
    r0, c0, r1, c1 = map(int, obj["bbox"])
    color = int(obj["color"])
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(region == color), "each bar must be a solid rectangle"
    assert bbox_w(obj["bbox"]) >= 1, "bar width must be positive"
    assert bbox_h(obj["bbox"]) >= 1, "bar height must be positive"


def _ordered_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    allowed = {int(Color.BLACK), int(Color.BLUE), int(Color.GREEN)}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected colors present"

    bars = find_components(
        g,
        bg=int(Color.BLACK),
        connectivity=4,
        monochromatic=True,
        colors=[int(Color.BLUE), int(Color.GREEN)],
        return_format="dict",
    )
    assert len(bars) >= 2, "expected multiple bars"

    ordered = sorted(bars, key=lambda obj: (int(obj["bbox"][0]), int(obj["bbox"][1])))
    green_count = 0
    last_bottom = -1
    for obj in ordered:
        _assert_solid_horizontal_bar(g, obj)
        r0, _c0, r1, _c1 = map(int, obj["bbox"])
        assert r0 > last_bottom, "bars must be separated vertically"
        last_bottom = r1
        green_count += int(int(obj["color"]) == int(Color.GREEN))
    assert green_count == 1, "expected exactly one green start bar"
    return ordered


def _pointer_walk(lengths: list[int], start_rank0: int) -> tuple[list[int], str]:
    visited = {int(start_rank0)}
    order = [int(start_rank0)]
    current = int(start_rank0)

    while True:
        target_rank = int(lengths[current])
        if not (1 <= target_rank <= len(lengths)):
            return order, "invalid"

        target_rank0 = target_rank - 1
        if target_rank0 in visited:
            return order, "visited"

        visited.add(target_rank0)
        order.append(target_rank0)
        current = target_rank0


def _paint_red(output: np.ndarray, bars: list[dict], visited_order: list[int]) -> np.ndarray:
    out = output.copy()
    for rank0 in visited_order[1:]:
        coords = bars[int(rank0)]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(Color.RED)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the hidden length-pointer chain that starts at the green bar and recolor every reached target bar red.

    Input structure:
    - Black background surrounds several separated solid horizontal bars.
    - Exactly one bar is green; all other bars are blue.
    - Bars are ordered from top to bottom, and a bar's horizontal length is its pointer value.

    Rule:
    - Number the bars from top to bottom as 1, 2, 3, ...
    - Start at the green bar, read its length n, jump to bar number n, then continue from that bar using its own length as the next pointer.
    - Stop before moving whenever the pointer is out of range or would revisit a previously visited bar. Keep the green start bar green and recolor every later visited bar red.

    Key insight:
    - The bar widths are not a size pattern to compare; each width is a 1-based index into the top-to-bottom bar list, so the red bars trace a hidden pointer chase.
    """
    g = np.asarray(grid, dtype=int)
    bars = _ordered_bars(g)
    lengths = [int(bbox_w(obj["bbox"])) for obj in bars]
    start_rank0 = next(i for i, obj in enumerate(bars) if int(obj["color"]) == int(Color.GREEN))
    visited_order, _termination = _pointer_walk(lengths, start_rank0)
    return _paint_red(g, bars, visited_order)


def _pointer_walk_with_order(order: list[int], pointers: list[int], start_bar: int) -> list[int]:
    pos_of = {bar: pos for pos, bar in enumerate(order)}
    visited_positions = {int(pos_of[start_bar])}
    visited_bars = [int(start_bar)]
    current_pos = int(pos_of[start_bar])

    while True:
        current_bar = int(order[current_pos])
        target_rank = int(pointers[current_bar])
        if not (1 <= target_rank <= len(order)):
            return visited_bars

        target_pos = target_rank - 1
        if target_pos in visited_positions:
            return visited_bars

        visited_positions.add(target_pos)
        current_pos = target_pos
        visited_bars.append(int(order[current_pos]))


def _sample_path(n_bars: int, branch: int, rng: np.random.Generator) -> list[int] | None:
    path_len_min = 3 if branch != 2 else 4
    path_len_max = min(5, n_bars - 1)
    if path_len_min > path_len_max:
        return None

    for _ in range(300):
        path_len = int(rng.integers(path_len_min, path_len_max + 1))
        path = list(map(int, rng.choice(np.arange(n_bars), size=path_len, replace=False)))
        if path[0] in {0, n_bars - 1}:
            continue

        diffs = [path[i + 1] - path[i] for i in range(len(path) - 1)]
        if not any(abs(delta) > 1 for delta in diffs):
            continue
        signs = {1 if delta > 0 else -1 for delta in diffs}
        if len(signs) < 2:
            continue
        if not any(idx not in path and min(path) < idx < max(path) for idx in range(n_bars)):
            continue
        if branch == 3 and 0 not in path[1:]:
            continue
        return path

    return None


def _termination_pointer(path: list[int], branch: int, n_bars: int, rng: np.random.Generator) -> tuple[int, str]:
    if branch in {0, 3}:
        extra = int(rng.integers(WIDTH_EXTRA_RANGE[0], WIDTH_EXTRA_RANGE[1] + 1))
        return int(n_bars + extra), "invalid"

    if branch == 1:
        revisit_pool = path[:-1]
    else:
        revisit_pool = path[1:-1]
        if not revisit_pool:
            revisit_pool = path[:-1]
    revisit_target = int(revisit_pool[int(rng.integers(len(revisit_pool)))])
    return revisit_target + 1, "visited"


def _assign_lengths(n_bars: int, path: list[int], branch: int, rng: np.random.Generator) -> tuple[list[int], str]:
    max_width = n_bars + WIDTH_EXTRA_RANGE[1]
    lengths = [int(rng.integers(1, max_width + 1)) for _ in range(n_bars)]

    for current, nxt in zip(path, path[1:]):
        lengths[int(current)] = int(nxt) + 1

    lengths[int(path[-1])], termination = _termination_pointer(path, branch, n_bars, rng)

    unvisited = [idx for idx in range(n_bars) if idx not in path]
    rng.shuffle(unvisited)
    if unvisited:
        trap = int(unvisited[0])
        lengths[trap] = int(path[int(rng.integers(1, len(path)))]) + 1
    if len(unvisited) >= 2:
        trap = int(unvisited[1])
        donor = int(path[int(rng.integers(len(path) - 1))])
        lengths[trap] = int(lengths[donor])
    if len(unvisited) >= 3:
        trap = int(unvisited[2])
        lengths[trap] = 1 if branch == 3 else int(rng.integers(n_bars + 1, max_width + 1))

    return lengths, termination


def _sample_heights(n_bars: int, path: list[int], rng: np.random.Generator) -> list[int]:
    heights = [int(rng.choice(BAR_HEIGHT_CHOICES)) for _ in range(n_bars)]
    if not any(height == 2 for height in heights):
        heights[int(rng.integers(n_bars))] = 2
    if not any(heights[idx] == 2 for idx in path):
        heights[int(path[int(rng.integers(len(path)))])] = 2
    if not any(height == 1 for height in heights):
        heights[int(path[0])] = 1
    return heights


def _sample_layout(
    lengths: list[int],
    heights: list[int],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], list[int], list[int]] | None:
    n_bars = len(lengths)
    gaps = [int(rng.integers(BAR_GAP_RANGE[0], BAR_GAP_RANGE[1])) for _ in range(n_bars - 1)]
    top_margin = int(rng.integers(GRID_MARGIN_RANGE[0], GRID_MARGIN_RANGE[1]))
    bottom_margin = int(rng.integers(GRID_MARGIN_RANGE[0], GRID_MARGIN_RANGE[1]))

    tops: list[int] = []
    row = top_margin
    for idx in range(n_bars):
        tops.append(int(row))
        row += int(heights[idx])
        if idx < n_bars - 1:
            row += int(gaps[idx])
    height = row + bottom_margin
    if height > 30:
        return None

    min_width = max(max(lengths) + 2, 10)
    max_width = min(30, max(lengths) + int(rng.integers(GRID_WIDTH_PADDING_RANGE[0], GRID_WIDTH_PADDING_RANGE[1])))
    if min_width > max_width:
        return None
    width = int(rng.integers(min_width, max_width + 1))
    lefts = [int(rng.integers(0, width - lengths[idx] + 1)) for idx in range(n_bars)]
    return (int(height), width), tops, lefts


def _render_grid(
    shape: tuple[int, int],
    lengths: list[int],
    heights: list[int],
    tops: list[int],
    lefts: list[int],
    start_bar: int,
) -> np.ndarray:
    grid = np.full(shape, int(Color.BLACK), dtype=int)
    for idx, width in enumerate(lengths):
        color = int(Color.GREEN) if idx == start_bar else int(Color.BLUE)
        r0 = int(tops[idx])
        c0 = int(lefts[idx])
        h = int(heights[idx])
        grid[r0 : r0 + h, c0 : c0 + int(width)] = color
    return grid


def _expected_output(
    grid: np.ndarray,
    lengths: list[int],
    start_bar: int,
    visited_order: list[int],
) -> np.ndarray:
    bars = _ordered_bars(grid)
    assert [int(bbox_w(obj["bbox"])) for obj in bars] == [int(v) for v in lengths]
    assert next(i for i, obj in enumerate(bars) if int(obj["color"]) == int(Color.GREEN)) == int(start_bar)
    return _paint_red(grid, bars, visited_order)


def _semantic_checks(
    lengths: list[int],
    heights: list[int],
    lefts: list[int],
    path: list[int],
    termination: str,
) -> bool:
    n_bars = len(lengths)
    if len(path) < 3:
        return False
    if termination == "invalid" and int(lengths[path[-1]]) <= n_bars:
        return False
    if termination == "visited":
        pointer = int(lengths[path[-1]])
        if not (1 <= pointer <= n_bars and (pointer - 1) in path):
            return False

    if not any(idx not in path and 1 <= int(lengths[idx]) <= n_bars for idx in range(n_bars)):
        return False
    if not any(
        idx not in path and 1 <= int(lengths[idx]) <= n_bars and (int(lengths[idx]) - 1) in path
        for idx in range(n_bars)
    ):
        return False
    if not any(idx not in path and min(path) < idx < max(path) for idx in range(n_bars)):
        return False
    if not any(int(heights[idx]) == 2 for idx in path):
        return False
    if list(lefts) == sorted(lefts) or list(lefts) == sorted(lefts, reverse=True):
        return False
    return True


def _example_is_valid(
    grid: np.ndarray,
    lengths: list[int],
    heights: list[int],
    tops: list[int],
    lefts: list[int],
    start_bar: int,
    visited_order: list[int],
    termination: str,
) -> bool:
    if not _semantic_checks(lengths, heights, lefts, visited_order, termination):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False

    expected = _expected_output(grid, lengths, start_bar, visited_order)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    bars = _ordered_bars(grid)
    parsed_lengths = [int(bbox_w(obj["bbox"])) for obj in bars]
    parsed_heights = [int(bbox_h(obj["bbox"])) for obj in bars]
    parsed_lefts = [int(obj["bbox"][1]) for obj in bars]
    parsed_tops = [int(obj["bbox"][0]) for obj in bars]
    if parsed_lengths != [int(v) for v in lengths]:
        return False
    if parsed_heights != [int(v) for v in heights]:
        return False
    if parsed_lefts != [int(v) for v in lefts]:
        return False
    if parsed_tops != [int(v) for v in tops]:
        return False

    correct_red = set(map(int, visited_order[1:]))
    left_order = sorted(range(len(lengths)), key=lambda idx: (int(lefts[idx]), int(tops[idx])))
    left_red = set(_pointer_walk_with_order(left_order, lengths, start_bar)[1:])
    if left_red == correct_red:
        return False

    area_pointers = [int(lengths[idx]) * int(heights[idx]) for idx in range(len(lengths))]
    area_red = set(_pointer_walk_with_order(list(range(len(lengths))), area_pointers, start_bar)[1:])
    if area_red == correct_red:
        return False

    return True


def _gen_attempt(branch: int) -> np.ndarray | None:
    rng = np.random.default_rng()
    n_bars = int(rng.integers(BAR_COUNT_RANGE[0], BAR_COUNT_RANGE[1]))
    path = _sample_path(n_bars, branch, rng)
    if path is None:
        return None

    lengths, termination = _assign_lengths(n_bars, path, branch, rng)
    heights = _sample_heights(n_bars, path, rng)
    laid_out = _sample_layout(lengths, heights, rng)
    if laid_out is None:
        return None
    shape, tops, lefts = laid_out

    start_bar = int(path[0])
    visited_order, actual_termination = _pointer_walk(lengths, start_bar)
    if visited_order != path or actual_termination != termination:
        return None

    grid = _render_grid(shape, lengths, heights, tops, lefts, start_bar)
    if not _example_is_valid(grid, lengths, heights, tops, lefts, start_bar, visited_order, termination):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    preferred_branch = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1

    for branch in [preferred_branch, 0, 1, 2, 3]:
        for _ in range(GENERATION_RETRIES):
            grid = _gen_attempt(int(branch))
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid pointer-chain bars example")
