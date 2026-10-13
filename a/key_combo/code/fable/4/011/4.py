from __future__ import annotations

import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _line_length(shape: tuple[int, int], side: str) -> int:
    height, width = shape
    return width if side in ("top", "bottom") else height


def _channel_length(shape: tuple[int, int], side: str) -> int:
    height, width = shape
    return height - 1 if side in ("top", "bottom") else width - 1


def _border_line(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return grid[0, :]
    if side == "bottom":
        return grid[-1, :]
    if side == "left":
        return grid[:, 0]
    if side == "right":
        return grid[:, -1]
    raise ValueError(f"unknown side: {side}")


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []

    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = prev = idx
    runs.append((start, prev))
    return runs


def _find_edge_bar(grid: np.ndarray, bg: int) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    for side in SIDES:
        line = _border_line(grid, side)
        color = int(line[0])
        if color != bg and np.all(line == color):
            candidates.append((side, color))
    assert len(candidates) == 1, "expected exactly one full non-background border bar"
    return candidates[0]


def _find_counter_segment(grid: np.ndarray, bg: int, side: str, edge_color: int) -> tuple[int, int, int]:
    line = _border_line(grid, OPPOSITE_SIDE[side])
    filled = np.flatnonzero(line != bg).tolist()
    runs = _contiguous_runs(filled)
    assert len(runs) == 1, "expected exactly one counter segment on the opposite border"
    start, end = runs[0]
    assert end - start + 1 < line.size, "counter segment must be shorter than the full border"
    segment = line[start : end + 1]
    color = int(segment[0])
    assert color != edge_color, "counter segment must use a color different from the edge bar"
    assert np.all(segment == color), "counter segment must be monochrome"
    return color, start, end


def _counter_mask(shape: tuple[int, int], side: str, start: int, end: int) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    opposite = OPPOSITE_SIDE[side]
    if opposite == "top":
        mask[0, start : end + 1] = True
    elif opposite == "bottom":
        mask[-1, start : end + 1] = True
    elif opposite == "left":
        mask[start : end + 1, 0] = True
    else:
        mask[start : end + 1, -1] = True
    return mask


def _channel_of(side: str, r: int, c: int) -> int:
    return int(c) if side in ("top", "bottom") else int(r)


def _index_from_coord(shape: tuple[int, int], side: str, r: int, c: int) -> int:
    height, width = shape
    if side == "top":
        return int(r) - 1
    if side == "bottom":
        return height - 2 - int(r)
    if side == "left":
        return int(c) - 1
    if side == "right":
        return width - 2 - int(c)
    raise ValueError(f"unknown side: {side}")


def _coord_from_index(shape: tuple[int, int], side: str, channel: int, index: int) -> tuple[int, int]:
    height, width = shape
    if side == "top":
        return int(index) + 1, int(channel)
    if side == "bottom":
        return height - 2 - int(index), int(channel)
    if side == "left":
        return int(channel), int(index) + 1
    if side == "right":
        return int(channel), width - 2 - int(index)
    raise ValueError(f"unknown side: {side}")


def _parse_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) >= 3, "grid must leave room for a border bar and walkable channel cells"

    bg = _dominant_color(g)
    values, counts = np.unique(g, return_counts=True)
    bg_count = int(counts[values == bg][0])
    assert all(int(count) < bg_count for value, count in zip(values, counts) if int(value) != bg), "background must dominate"

    side, edge_color = _find_edge_bar(g, bg)
    counter_color, counter_start, counter_end = _find_counter_segment(g, bg, side, edge_color)

    marker_mask = g != bg
    if side == "top":
        marker_mask[0, :] = False
    elif side == "bottom":
        marker_mask[-1, :] = False
    elif side == "left":
        marker_mask[:, 0] = False
    else:
        marker_mask[:, -1] = False
    marker_mask[_counter_mask(g.shape, side, counter_start, counter_end)] = False

    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(marker_mask)]
    assert marker_coords, "expected at least one marker pixel"

    marker_colors = {int(g[r, c]) for r, c in marker_coords}
    assert len(marker_colors) == 1, "all markers must share one color"
    marker_color = marker_colors.pop()
    assert marker_color not in {bg, edge_color, counter_color}, "marker color must be distinct from the bars"
    assert len(set(map(int, np.unique(g)))) == 4, "expected background, edge bar, counter bar, and marker colors only"

    counter_channels = set(range(counter_start, counter_end + 1))
    seen_channels: set[int] = set()
    walk_length = _channel_length(g.shape, side)
    for r, c in marker_coords:
        channel = _channel_of(side, r, c)
        assert channel not in counter_channels, "counter channels must not contain markers"
        assert channel not in seen_channels, "each channel may contain at most one marker"
        seen_channels.add(channel)

        index = _index_from_coord(g.shape, side, r, c)
        assert 0 <= index < walk_length, "marker must lie in its channel's walkable cells"

    return {
        "grid": g,
        "bg": bg,
        "side": side,
        "edge_color": edge_color,
        "counter_color": counter_color,
        "counter_start": counter_start,
        "counter_end": counter_end,
        "step_count": counter_end - counter_start + 1,
        "marker_color": marker_color,
        "marker_coords": marker_coords,
    }


def _apply_rule(spec: dict[str, object]) -> np.ndarray:
    grid = spec["grid"]
    assert isinstance(grid, np.ndarray)

    bg = int(spec["bg"])
    side = str(spec["side"])
    edge_color = int(spec["edge_color"])
    marker_color = int(spec["marker_color"])
    step_count = int(spec["step_count"])
    marker_coords = list(spec["marker_coords"])

    out = grid.copy()
    for r, c in marker_coords:
        out[r, c] = bg

    walk_length = _channel_length(grid.shape, side)
    for r, c in marker_coords:
        channel = _channel_of(side, r, c)
        index = _index_from_coord(grid.shape, side, r, c)
        new_index = (index - step_count) % walk_length
        wrapped = step_count > index
        rr, cc = _coord_from_index(grid.shape, side, channel, new_index)
        assert out[rr, cc] == bg, "markers must land on free cells"
        out[rr, cc] = edge_color if wrapped else marker_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marker toward the full edge bar by the counter-bar length, wrapping within its channel and recoloring wrapped markers.

    Input structure:
    - One outer border of the grid is completely filled by a one-cell-thick edge bar of one color.
    - On the opposite border there is exactly one shorter monochrome segment whose length encodes a shared step count.
    - Every remaining non-background foreground cell is an isolated marker, with at most one marker in each channel perpendicular to the edge bar.

    Rule:
    - Treat each row-or-column channel perpendicular to the full edge bar as a cyclic track whose walkable cells are all positions except the edge-bar cell itself.
    - Move every marker toward the edge bar by as many steps as the opposite-border segment length.
    - If a marker passes the cell nearest the edge bar, wrap it to the far end of the same channel and continue; wrapped markers change to the edge-bar color, while non-wrapped markers keep the marker color.

    Key insight:
    - The short border segment is not another moving object; it is a visible counter that tells every channel-aligned marker how far to advance around the same one-dimensional cycle.
    """
    return _apply_rule(_parse_grid(grid))


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[values == bg][0])
    return all(int(count) < bg_count for value, count in zip(values, counts) if int(value) != bg)


def _sample_shape(side: str) -> tuple[int, int]:
    if side in ("top", "bottom"):
        return random.randint(6, 12), random.randint(8, 20)
    return random.randint(8, 20), random.randint(6, 12)


def _paint_bars(grid: np.ndarray, side: str, edge_color: int, counter_color: int, start: int, steps: int) -> None:
    if side == "top":
        grid[0, :] = edge_color
        grid[-1, start : start + steps] = counter_color
    elif side == "bottom":
        grid[-1, :] = edge_color
        grid[0, start : start + steps] = counter_color
    elif side == "left":
        grid[:, 0] = edge_color
        grid[start : start + steps, -1] = counter_color
    else:
        grid[:, -1] = edge_color
        grid[start : start + steps, 0] = counter_color


def _sample_marker_channels(total_channels: int, blocked_channels: set[int], count: int) -> list[int]:
    candidates = [idx for idx in range(total_channels) if idx not in blocked_channels]
    random.shuffle(candidates)

    selected: list[int] = []
    for idx in candidates:
        if all(abs(idx - prev) > 1 for prev in selected):
            selected.append(idx)
            if len(selected) == count:
                return selected

    for idx in candidates:
        if idx not in selected:
            selected.append(idx)
            if len(selected) == count:
                return selected

    return selected


def _sample_marker_index(step_count: int, walk_length: int, should_wrap: bool) -> int:
    interior_max = walk_length - 2
    if should_wrap:
        return random.randint(0, step_count - 1)
    return random.randint(step_count, interior_max)


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(SIDES)
    height, width = _sample_shape(side)
    shape = (height, width)
    walk_length = _channel_length(shape, side)
    total_channels = _line_length(shape, side)
    if walk_length < 3 or total_channels < 4:
        return None

    step_max = min(total_channels - 2, walk_length - 2)
    if step_max < 1:
        return None

    bg, edge_color, marker_color, counter_color = random.sample(PALETTE, 4)
    step_count = random.randint(1, step_max)
    counter_start = random.randint(0, total_channels - step_count)
    blocked_channels = set(range(counter_start, counter_start + step_count))

    available_channels = total_channels - step_count
    if available_channels < 2:
        return None
    marker_count = random.randint(2, min(6, available_channels))
    marker_channels = _sample_marker_channels(total_channels, blocked_channels, marker_count)
    if len(marker_channels) != marker_count:
        return None

    grid = np.full(shape, bg, dtype=int)
    _paint_bars(grid, side, edge_color, counter_color, counter_start, step_count)

    for i, channel in enumerate(marker_channels):
        if i == 0:
            should_wrap = True
        elif i == 1:
            should_wrap = False
        else:
            should_wrap = random.random() < 0.5
        index = _sample_marker_index(step_count, walk_length, should_wrap)
        r, c = _coord_from_index(shape, side, channel, index)
        if grid[r, c] != bg:
            return None
        grid[r, c] = marker_color

    if not _background_is_dominant(grid, bg):
        return None

    try:
        spec = _parse_grid(grid)
        expected = _apply_rule(spec)
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid counting-bar wraparound input")
