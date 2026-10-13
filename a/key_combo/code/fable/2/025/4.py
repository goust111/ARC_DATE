from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
PILLAR_COLORS = np.array([1, 3, 4, 5, 6, 7, 8, 9], dtype=int)

MIN_GAP = 2
MIN_HEIGHT = 2
MAX_HEIGHT = 8
MAX_GENERATION_ATTEMPTS = 400


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(int(lo), int(hi) + 1))


def _extract_pillars(grid: np.ndarray) -> list[dict[str, int]]:
    height, _width = grid.shape
    pillars: list[dict[str, int]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (BLACK, RED)):
        coords = np.argwhere(grid == int(color))
        assert coords.size > 0, "every pillar color must appear at least once"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        pillar_height = int(r1 - r0 + 1)
        assert int(c0) == int(c1), "each pillar must be one cell wide"
        assert int(r1) == height - 1, "each pillar must stand on the bottom row"
        assert int(coords.shape[0]) == pillar_height, "pillar cells must form one solid segment"
        assert np.array_equal(np.sort(coords[:, 0]), np.arange(int(r0), int(r1) + 1))
        pillars.append(
            {
                "color": int(color),
                "col": int(c0),
                "height": pillar_height,
                "top_row": int(r0),
            }
        )

    assert pillars, "expected at least one pillar"
    pillars.sort(key=lambda pillar: int(pillar["col"]))
    cols = [int(pillar["col"]) for pillar in pillars]
    assert len(cols) == len(set(cols)), "pillar base columns must be unique"
    assert all(cols[i + 1] - cols[i] >= MIN_GAP for i in range(len(cols) - 1)), "pillars must be separated"
    return pillars


def _find_trigger(grid: np.ndarray, pillars: list[dict[str, int]]) -> tuple[int, int]:
    marker_coords = np.argwhere(grid == RED)
    assert marker_coords.shape == (1, 2), "expected exactly one red marker"
    marker_row, marker_col = map(int, marker_coords[0])

    matches: list[tuple[int, int]] = []
    for index, pillar in enumerate(pillars):
        if marker_row != int(pillar["top_row"]):
            continue
        delta = marker_col - int(pillar["col"])
        if delta == -1:
            matches.append((index, 1))
        elif delta == 1:
            matches.append((index, -1))

    assert len(matches) == 1, "marker must touch exactly one pillar top from one side"
    return matches[0]


def _segment_bounds(col: int, pillar_height: int, direction: int, width: int) -> tuple[int, int]:
    if direction == 1:
        left = int(col)
        right = int(col + pillar_height - 1)
    else:
        left = int(col - pillar_height + 1)
        right = int(col)
    assert 0 <= left <= right < width, "fallen pillar must stay inside the grid"
    return left, right


def _chain_order(pillars: list[dict[str, int]], trigger_index: int, direction: int) -> list[int]:
    step = 1 if int(direction) == 1 else -1
    toppled = [int(trigger_index)]
    current = int(trigger_index)
    while 0 <= current + step < len(pillars):
        nxt = current + step
        gap = abs(int(pillars[nxt]["col"]) - int(pillars[current]["col"]))
        if int(pillars[current]["height"]) >= gap + 1:
            toppled.append(int(nxt))
            current = int(nxt)
        else:
            break
    return toppled


def _render_output(
    shape: tuple[int, int],
    pillars: list[dict[str, int]],
    toppled: list[int],
    direction: int,
) -> np.ndarray:
    height, width = map(int, shape)
    toppled_set = set(map(int, toppled))
    out = np.full((height, width), BLACK, dtype=int)

    for index, pillar in enumerate(pillars):
        if index in toppled_set:
            continue
        out[height - int(pillar["height"]) : height, int(pillar["col"])] = int(pillar["color"])

    for index in toppled:
        pillar = pillars[int(index)]
        left, right = _segment_bounds(int(pillar["col"]), int(pillar["height"]), int(direction), width)
        out[height - 1, left : right + 1] = int(pillar["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Topple the marked grounded pillar away from the red push marker and propagate the fall along the row.

    Input structure:
    - The background is black, with several one-cell-wide vertical pillars standing on the bottom row.
    - Every pillar has its own non-red color and occupies one contiguous column segment.
    - Exactly one red pixel touches one side of one pillar's top cell; that side is the push side.

    Rule:
    - The marked pillar falls to the side opposite the red marker.
    - A pillar of height h becomes a same-colored horizontal bar on the bottom row, starting at its base cell and extending h cells in the fall direction.
    - Moving pillar by pillar in that direction, if the current fallen bar reaches the next pillar's base column, that next pillar falls too; later fallen bars overwrite earlier overlaps, untouched pillars stay vertical, and the red marker disappears.

    Key insight:
    - Height is reach: each consecutive pillar-to-pillar gap must be covered in order, so one short middle pillar breaks the chain and everything beyond it survives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pillars = _extract_pillars(g)
    trigger_index, direction = _find_trigger(g, pillars)
    toppled = _chain_order(pillars, trigger_index, direction)
    return _render_output(g.shape, pillars, toppled, direction)


def _sample_columns(pillar_count: int, height_cap: int, direction: int) -> tuple[int, list[int]]:
    push_margin = _randint(1, 3)
    if direction == 1:
        left_margin = push_margin
        right_margin = height_cap
    else:
        left_margin = height_cap
        right_margin = push_margin

    gap_hi = min(height_cap - 1, (29 - left_margin - right_margin) // (pillar_count - 1))
    assert gap_hi >= MIN_GAP
    gaps = [_randint(MIN_GAP, gap_hi) for _ in range(pillar_count - 1)]
    span = sum(gaps)
    width_min = left_margin + right_margin + span + 1
    assert width_min <= 30
    width = _randint(width_min, min(30, width_min + 6))
    start = _randint(left_margin, width - right_margin - 1 - span)

    cols = [int(start)]
    for gap in gaps:
        cols.append(int(cols[-1] + gap))
    return width, cols


def _sample_trigger(pillar_count: int, direction: int) -> int:
    valid = list(range(0, pillar_count - 1)) if direction == 1 else list(range(1, pillar_count))
    interior = [index for index in valid if 0 < index < pillar_count - 1]
    if interior and np.random.rand() < 0.7:
        return int(np.random.choice(interior))
    return int(np.random.choice(valid))


def _sample_chain_length(reachable_count: int) -> int:
    if reachable_count == 2:
        return 1 if np.random.rand() < 0.5 else 2
    branch_roll = float(np.random.rand())
    if branch_roll < 0.3:
        return 1
    if branch_roll < 0.75:
        return _randint(2, reachable_count - 1)
    return int(reachable_count)


def _build_heights(
    cols: list[int],
    width: int,
    trigger_index: int,
    direction: int,
    chain_length: int,
    height_cap: int,
) -> tuple[list[int], list[int]]:
    step = 1 if direction == 1 else -1
    path = list(range(trigger_index, len(cols), step)) if step == 1 else list(range(trigger_index, -1, step))
    toppled = path[:chain_length]
    ahead = path[chain_length:]

    heights = [0] * len(cols)
    for order_index, pillar_index in enumerate(toppled):
        col = int(cols[pillar_index])
        edge_limit = width - col if direction == 1 else col + 1
        if order_index < len(toppled) - 1:
            next_col = int(cols[toppled[order_index + 1]])
            gap = abs(next_col - col)
            low = gap + 1
            high = min(edge_limit, height_cap, gap + 2)
            assert low <= high
            heights[pillar_index] = low if (low == high or np.random.rand() < 0.5) else high
        elif ahead:
            stop_col = int(cols[ahead[0]])
            gap = abs(stop_col - col)
            high = min(edge_limit, height_cap, gap)
            low = min(MIN_HEIGHT, high)
            assert low <= high
            heights[pillar_index] = high if (low == high or np.random.rand() < 0.65) else _randint(low, high)
        else:
            heights[pillar_index] = _randint(MIN_HEIGHT, min(edge_limit, height_cap))

    for pillar_index, col in enumerate(cols):
        if heights[pillar_index]:
            continue
        edge_limit = width - int(col) if direction == 1 else int(col) + 1
        heights[pillar_index] = _randint(MIN_HEIGHT, min(MAX_HEIGHT, height_cap, max(MIN_HEIGHT, edge_limit)))

    return heights, toppled


def _validate_scene(
    grid: np.ndarray,
    intended_direction: int | None = None,
    intended_toppled: list[int] | None = None,
) -> None:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    assert g.ndim == 2
    assert height <= 30 and width <= 30

    pillars = _extract_pillars(g)
    trigger_index, direction = _find_trigger(g, pillars)
    if intended_direction is not None:
        assert int(direction) == int(intended_direction)

    toppled = _chain_order(pillars, trigger_index, direction)
    if intended_toppled is not None:
        assert toppled == list(map(int, intended_toppled))

    step = 1 if direction == 1 else -1
    path = list(range(trigger_index, len(pillars), step)) if step == 1 else list(range(trigger_index, -1, step))

    for order_index, pillar_index in enumerate(toppled):
        pillar = pillars[pillar_index]
        left, right = _segment_bounds(int(pillar["col"]), int(pillar["height"]), int(direction), width)
        if direction == 1:
            assert left == int(pillar["col"])
        else:
            assert right == int(pillar["col"])

        if order_index + 1 < len(toppled):
            next_col = int(pillars[toppled[order_index + 1]]["col"])
            assert left <= next_col <= right, "every toppled pillar must reach the next one"
            if order_index + 2 < len(path):
                second_next_col = int(pillars[path[order_index + 2]]["col"])
                assert not (left <= second_next_col <= right), "generator must avoid multi-hit ambiguity"
        elif len(toppled) < len(path):
            next_col = int(pillars[path[len(toppled)]]["col"])
            assert not (left <= next_col <= right), "the chain must stop exactly at the first unhit pillar"

    out = main(g)
    assert out.shape == g.shape
    assert np.count_nonzero(out == RED) == 0
    assert not np.array_equal(out, g), "the transformation must change the grid"
    untouched = [index for index in range(len(pillars)) if index not in set(toppled)]
    for index in untouched:
        pillar = pillars[index]
        assert np.array_equal(
            out[height - int(pillar["height"]) : height, int(pillar["col"])],
            g[height - int(pillar["height"]) : height, int(pillar["col"])],
        ), "standing pillars must remain vertical"


def _generate_attempt() -> np.ndarray:
    pillar_count = _randint(4, 7)
    direction = 1 if np.random.rand() < 0.5 else -1
    height_cap = _randint(5, MAX_HEIGHT)

    width, cols = _sample_columns(pillar_count, height_cap, direction)
    trigger_index = _sample_trigger(pillar_count, direction)
    step = 1 if direction == 1 else -1
    path = list(range(trigger_index, pillar_count, step)) if step == 1 else list(range(trigger_index, -1, step))
    chain_length = _sample_chain_length(len(path))
    heights, toppled = _build_heights(cols, width, trigger_index, direction, chain_length, height_cap)

    grid_height = _randint(max(heights), min(30, max(heights) + 4))
    colors = [int(color) for color in np.random.choice(PILLAR_COLORS, size=pillar_count, replace=False)]
    grid = np.full((grid_height, width), BLACK, dtype=int)
    for col, pillar_height, color in zip(cols, heights, colors):
        grid[grid_height - int(pillar_height) : grid_height, int(col)] = int(color)

    trigger_col = int(cols[trigger_index])
    trigger_top = grid_height - int(heights[trigger_index])
    marker_col = trigger_col - 1 if direction == 1 else trigger_col + 1
    grid[trigger_top, marker_col] = RED

    _validate_scene(grid, intended_direction=direction, intended_toppled=toppled)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _generate_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid domino-topple input")
