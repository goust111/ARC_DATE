import random
from collections import Counter

import numpy as np


BG = 0
BAR_COLOR = 3
MARKER_COLOR = 7
INNER_SIZE = 6
FRAME_SIZE = INNER_SIZE + 2
FRAME_COLORS = (1, 2, 4, 5, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _find_square_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    candidates = []
    for size in range(5, min(h, w) + 1):
        for r0 in range(h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(w - size + 1):
                c1 = c0 + size - 1
                color = int(g[r0, c0])
                if color == bg:
                    continue
                if (
                    int(g[r0, c1]) != color
                    or int(g[r1, c0]) != color
                    or int(g[r1, c1]) != color
                ):
                    continue
                if not (
                    np.all(g[r0, c0 : c1 + 1] == color)
                    and np.all(g[r1, c0 : c1 + 1] == color)
                    and np.all(g[r0 : r1 + 1, c0] == color)
                    and np.all(g[r0 : r1 + 1, c1] == color)
                ):
                    continue
                inner = g[r0 + 1 : r1, c0 + 1 : c1]
                if inner.size == 0 or not np.any(inner != color):
                    continue
                candidates.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "size": size,
                        "color": color,
                        "inner": inner.copy(),
                    }
                )
    candidates.sort(key=lambda item: (-item["size"], item["bbox"][0], item["bbox"][1]))
    frames = []
    for cand in candidates:
        if any(_overlap(cand["bbox"], other["bbox"]) for other in frames):
            continue
        frames.append(cand)
    return frames


def _extract_bars(inner: np.ndarray) -> tuple[list[dict], int, int]:
    inner = np.asarray(inner, dtype=int)
    h, w = inner.shape
    colors = [int(c) for c in np.unique(inner) if int(c) != BG]
    color_counts = sorted(((int(np.sum(inner == c)), c) for c in colors))
    marker_color = color_counts[0][1]
    bar_color = color_counts[-1][1]

    bars = []
    for col in range(w):
        rows = np.where(inner[:, col] != BG)[0]
        if rows.size == 0:
            continue
        if rows[-1] != h - 1:
            raise AssertionError("bars must be bottom-aligned")
        if not np.array_equal(rows, np.arange(rows[0], h)):
            raise AssertionError("bars must be contiguous")
        top = int(rows[0])
        pixels = inner[top:, col].copy()
        marker_rows = np.where(pixels == marker_color)[0]
        marker_half = None
        if marker_rows.size:
            marker_half = "top" if float(marker_rows.mean()) < (len(pixels) - 1) / 2 else "bottom"
        bars.append(
            {
                "col": col,
                "height": len(pixels),
                "pixels": pixels,
                "marker_half": marker_half,
            }
        )

    bars.sort(key=lambda bar: bar["col"])
    return bars, bar_color, marker_color


def _monotonic_direction(bars: list[dict]) -> str | None:
    heights = [bar["height"] for bar in sorted(bars, key=lambda bar: bar["col"])]
    if len(heights) < 2:
        return None
    asc = all(a <= b for a, b in zip(heights, heights[1:])) and any(
        a < b for a, b in zip(heights, heights[1:])
    )
    dec = all(a >= b for a, b in zip(heights, heights[1:])) and any(
        a > b for a, b in zip(heights, heights[1:])
    )
    if asc and not dec:
        return "asc"
    if dec and not asc:
        return "desc"
    return None


def _reorder_bars(shape: tuple[int, int], bars: list[dict], direction: str) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ordered_cols = sorted(bar["col"] for bar in bars)
    sorted_bars = sorted(
        bars,
        key=lambda bar: (bar["height"], bar["col"]),
        reverse=(direction == "desc"),
    )
    for col, bar in zip(ordered_cols, sorted_bars):
        height = bar["height"]
        out[shape[0] - height :, col] = bar["pixels"]
    return out


def _infer_half_mapping(training_frames: list[dict]) -> dict[str, str]:
    infos = []
    for frame in training_frames:
        bars, _bar_color, _marker_color = _extract_bars(frame["inner"])
        infos.append({"bars": bars, "direction": _monotonic_direction(bars)})

    sorted_infos = [info for info in infos if info["direction"] is not None]
    if len(sorted_infos) != 1:
        raise AssertionError("training pair must contain exactly one sorted frame")

    target = sorted_infos[0]
    marker_bars = [bar for bar in target["bars"] if bar["marker_half"] is not None]
    if len(marker_bars) != 1:
        raise AssertionError("expected exactly one marker bar in training target")

    seen_half = marker_bars[0]["marker_half"]
    seen_dir = target["direction"]
    other_half = "bottom" if seen_half == "top" else "top"
    other_dir = "desc" if seen_dir == "asc" else "asc"
    return {seen_half: seen_dir, other_half: other_dir}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the test frame's vertical bars by height using the training pair and the marker-half cue.

    Input structure:
    - The grid contains three non-overlapping hollow square frames on a black background.
    - Two frames share the same border color and form a training pair; the third border color marks the test frame.
    - Each frame interior holds 3-5 bottom-aligned vertical bars, and exactly one bar contains an orange marker segment in its top or bottom half.

    Rule:
    - Identify the training pair from the repeated border color and determine which of those two interiors is already sorted by bar height from left to right.
    - Read the marked bar's half-position in that sorted training frame to infer whether that half means ascending or descending order.
    - In the test frame, keep the same occupied columns, reorder the bars by height according to the test marker's half-position, and output only the reordered test interior.

    Key insight:
    - The repeated border color says which two frames teach the rule, while the marker half disambiguates whether "sort by height" means ascending or descending.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_square_frames(g)
    if len(frames) < 3:
        raise AssertionError("expected at least three square frames")

    by_color: dict[int, list[dict]] = {}
    for frame in frames:
        by_color.setdefault(frame["color"], []).append(frame)

    training_frames = None
    test_frame = None
    for color, group in by_color.items():
        if len(group) == 2:
            training_frames = group
        elif len(group) == 1:
            test_frame = group[0]

    if training_frames is None or test_frame is None:
        raise AssertionError("failed to separate training pair from test frame")

    half_to_direction = _infer_half_mapping(training_frames)

    test_bars, _bar_color, _marker_color = _extract_bars(test_frame["inner"])
    marker_bars = [bar for bar in test_bars if bar["marker_half"] is not None]
    if len(marker_bars) != 1:
        raise AssertionError("expected exactly one marked test bar")

    direction = half_to_direction[marker_bars[0]["marker_half"]]
    return _reorder_bars(test_frame["inner"].shape, test_bars, direction)


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _is_monotonic(seq: list[int]) -> bool:
    return seq == sorted(seq) or seq == sorted(seq, reverse=True)


def _sample_bar_spec() -> tuple[list[int], list[int], int]:
    count = random.randint(3, 5)
    while True:
        heights = random.sample(range(1, INNER_SIZE + 1), count)
        even_heights = [height for height in heights if height % 2 == 0]
        if even_heights:
            break
    columns = sorted(random.sample(range(INNER_SIZE), count))
    marker_height = random.choice(even_heights)
    return columns, heights, marker_height


def _sample_unsorted_order(heights: list[int]) -> list[int]:
    target_orders = {tuple(sorted(heights)), tuple(sorted(heights, reverse=True))}
    for _ in range(100):
        order = random.sample(heights, len(heights))
        if tuple(order) not in target_orders:
            return order
    raise RuntimeError("failed to sample a non-monotonic order")


def _build_interior(
    columns: list[int],
    heights_left_to_right: list[int],
    marker_height: int,
    marker_half: str,
    *,
    bar_color: int = BAR_COLOR,
    marker_color: int = MARKER_COLOR,
) -> np.ndarray:
    inner = np.zeros((INNER_SIZE, INNER_SIZE), dtype=int)
    for col, height in zip(columns, heights_left_to_right):
        top = INNER_SIZE - height
        inner[top:, col] = bar_color
        if height == marker_height:
            half = height // 2
            if marker_half == "top":
                inner[top : top + half, col] = marker_color
            else:
                inner[INNER_SIZE - half :, col] = marker_color
    return inner


def _draw_frame(grid: np.ndarray, top: int, left: int, border_color: int, inner: np.ndarray) -> None:
    bottom = top + FRAME_SIZE - 1
    right = left + FRAME_SIZE - 1
    grid[top, left : right + 1] = border_color
    grid[bottom, left : right + 1] = border_color
    grid[top : bottom + 1, left] = border_color
    grid[top : bottom + 1, right] = border_color
    grid[top + 1 : bottom, left + 1 : right] = inner


def _place_frame_boxes(height: int, width: int) -> list[tuple[int, int]]:
    free_width = width - 3 * FRAME_SIZE
    if free_width < 0:
        raise ValueError("grid too narrow for three frames")
    gaps = _random_composition(free_width, 4)
    left0 = gaps[0]
    left1 = left0 + FRAME_SIZE + gaps[1]
    left2 = left1 + FRAME_SIZE + gaps[2]
    rows = [random.randint(0, height - FRAME_SIZE) for _ in range(3)]
    return [(rows[0], left0), (rows[1], left1), (rows[2], left2)]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 18)
    width = random.randint(26, 30)
    marker_half = random.choice(("top", "bottom"))

    train_columns, train_heights, train_marker_height = _sample_bar_spec()
    test_columns, test_heights, test_marker_height = _sample_bar_spec()

    train_source_order = _sample_unsorted_order(train_heights)
    train_direction = "asc" if marker_half == "top" else "desc"
    train_target_order = sorted(train_heights, reverse=(train_direction == "desc"))
    test_order = _sample_unsorted_order(test_heights)
    test_output_order = sorted(test_heights, reverse=(train_direction == "desc"))

    train_source = _build_interior(train_columns, train_source_order, train_marker_height, marker_half)
    train_target = _build_interior(train_columns, train_target_order, train_marker_height, marker_half)
    test_input = _build_interior(test_columns, test_order, test_marker_height, marker_half)
    expected = _build_interior(test_columns, test_output_order, test_marker_height, marker_half)

    if np.array_equal(train_source, train_target) or np.array_equal(test_input, expected):
        return None

    grid = np.zeros((height, width), dtype=int)
    train_color, test_color = random.sample(FRAME_COLORS, 2)
    positions = _place_frame_boxes(height, width)
    frame_payloads = [
        (train_color, train_source),
        (train_color, train_target),
        (test_color, test_input),
    ]
    random.shuffle(frame_payloads)
    for (top, left), (border_color, inner) in zip(positions, frame_payloads):
        _draw_frame(grid, top, left, border_color, inner)

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    if len(_find_square_frames(grid)) != 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
