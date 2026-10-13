from __future__ import annotations

import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    ORANGE = 7


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _channel_spans(width: int, n_channels: int, channel_width: int) -> list[tuple[int, int]]:
    expected_width = 1 + n_channels * channel_width + (n_channels - 1)
    assert width == expected_width
    spans: list[tuple[int, int]] = []
    c = 1
    for i in range(n_channels):
        spans.append((c, c + channel_width - 1))
        c += channel_width
        if i != n_channels - 1:
            c += 1
    return spans


def _separator_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == Color.GREY)]


def _extract_channels(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    sep_cols = _separator_columns(grid)
    assert 4 <= len(sep_cols) <= 6, "expected 5-7 channels separated by full grey columns"
    assert 0 not in sep_cols and w - 1 not in sep_cols, "grey separators may not occupy the outer border"

    channels: list[tuple[int, int]] = []
    prev = 0
    for sep in sep_cols:
        c0 = prev + 1
        c1 = sep - 1
        assert c0 <= c1, "empty channel"
        channels.append((c0, c1))
        prev = sep
    channels.append((prev + 1, w - 1))

    widths = {c1 - c0 + 1 for c0, c1 in channels}
    assert len(widths) == 1, "channels must have equal width"
    channel_width = next(iter(widths))
    assert 2 <= channel_width <= 5

    allowed = {Color.BLACK, Color.BLUE, Color.GREEN, Color.GREY, Color.RED}
    assert set(map(int, np.unique(grid))).issubset(allowed)
    assert h <= 30 and w <= 30
    return channels


def _extract_bar(grid: np.ndarray, c0: int, c1: int) -> tuple[int, int] | None:
    patch = grid[:, c0 : c1 + 1]
    assert set(map(int, np.unique(patch))).issubset({Color.BLACK, Color.BLUE, Color.GREEN})
    mask = np.isin(patch, [Color.BLUE, Color.GREEN])
    if not mask.any():
        return None

    rows = np.where(mask.any(axis=1))[0]
    assert len(rows) == 1, "a channel may contain at most one horizontal obstacle row"
    r = int(rows[0])
    colors = {int(v) for v in patch[r] if int(v) in (Color.BLUE, Color.GREEN)}
    assert len(colors) == 1
    color = next(iter(colors))
    assert np.all(patch[r] == color), "obstacle must cover the whole channel width"
    assert int(mask.sum()) == patch.shape[1]
    return r, color


def _parse_grid(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int] | None]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = _get_bg(g)
    assert bg == Color.BLACK

    red = np.argwhere(g == Color.RED)
    assert red.shape == (1, 2), "expected exactly one red start marker"
    start_r, start_c = map(int, red[0])
    assert start_c == 0, "red marker must be on the left border"
    assert np.count_nonzero(g[:, 0] == Color.RED) == 1
    assert np.count_nonzero(g[:, 0] == Color.GREY) == 0

    channels = _extract_channels(g)
    specs = [_extract_bar(g, c0, c1) for c0, c1 in channels]
    return start_r, channels, specs


def _delta_for_color(color: int) -> int:
    if color == Color.BLUE:
        return -1
    if color == Color.GREEN:
        return 1
    raise AssertionError("unexpected obstacle color")


def _simulate(start_r: int, specs: list[tuple[int, int] | None]) -> list[dict]:
    current = int(start_r)
    records: list[dict] = []
    for index, spec in enumerate(specs):
        entry_row = current
        hit = bool(spec is not None and spec[0] == current)
        if hit:
            current += _delta_for_color(spec[1])
        records.append(
            {
                "index": index,
                "entry_row": entry_row,
                "bar": spec,
                "hit": hit,
                "exit_row": current,
            }
        )
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each channel by whether the hidden path detours around its obstacle.

    Input structure:
    - The grid is black with one red start marker on the left border.
    - Full-height grey columns split the canvas into 5-7 equal-width vertical channels.
    - Each channel contains either no obstacle or one full-width horizontal blue/green bar.

    Rule:
    - Start a hidden horizontal path on the red marker row and carry a current row state across channels.
    - In each channel, collide only when that channel's obstacle row matches the current row; blue moves the state up by 1 and green moves it down by 1.
    - Fill black cells in hit channels orange and black cells in non-hit channels yellow, while keeping the red marker, grey separators, and blue/green bars unchanged.

    Key insight:
    - A channel's color depends on the accumulated row shifts caused by earlier collisions, so the channels must be simulated from left to right instead of classified independently.
    """
    g = np.asarray(grid, dtype=int)
    start_r, channels, specs = _parse_grid(g)
    records = _simulate(start_r, specs)

    out = g.copy()
    for (c0, c1), record in zip(channels, records, strict=True):
        fill = Color.ORANGE if record["hit"] else Color.YELLOW
        patch = out[:, c0 : c1 + 1]
        patch[patch == Color.BLACK] = fill
        out[:, c0 : c1 + 1] = patch
    return out


def _random_miss_row(current_row: int, height: int) -> int:
    candidates = [r for r in range(1, height - 1) if r != current_row]
    assert candidates
    near = [r for r in candidates if abs(r - current_row) == 1]
    return int(random.choice(near or candidates))


def _random_color() -> int:
    return int(random.choice([Color.BLUE, Color.GREEN]))


def _color_for_delta(delta: int) -> int:
    return Color.BLUE if delta < 0 else Color.GREEN


def _build_specs(height: int, n_channels: int, start_r: int) -> list[tuple[int, int] | None]:
    assert 5 <= n_channels <= 7
    assert 2 <= start_r <= height - 3

    extra = n_channels - 5
    prefix_len = random.randint(0, extra)
    suffix_len = extra - prefix_len
    specs: list[tuple[int, int] | None] = []

    inserted_empty = False
    current = start_r
    for i in range(prefix_len):
        if (not inserted_empty and (i == prefix_len - 1 or random.random() < 0.6)) or random.random() < 0.35:
            specs.append(None)
            inserted_empty = True
        else:
            specs.append((_random_miss_row(current, height), _random_color()))

    first_delta = random.choice([-1, 1])
    specs.append((current, _color_for_delta(first_delta)))
    current += first_delta

    specs.append((start_r, _random_color()))

    second_delta = -first_delta
    specs.append((current, _color_for_delta(second_delta)))
    current += second_delta

    specs.append((current, _color_for_delta(second_delta)))
    current += second_delta

    pre_last_row = current
    last_delta_choices = [delta for delta in (-1, 1) if 1 <= current + delta <= height - 2]
    assert last_delta_choices
    last_delta = random.choice(last_delta_choices)
    specs.append((current, _color_for_delta(last_delta)))
    current += last_delta

    if suffix_len >= 1:
        specs.append((pre_last_row, _random_color()))
        suffix_len -= 1

    for i in range(suffix_len):
        if not inserted_empty and i == suffix_len - 1:
            specs.append(None)
            inserted_empty = True
            continue

        roll = random.random()
        if roll < 0.35:
            specs.append(None)
            inserted_empty = True
        elif roll < 0.75:
            specs.append((_random_miss_row(current, height), _random_color()))
        else:
            hit_choices = [delta for delta in (-1, 1) if 1 <= current + delta <= height - 2]
            delta = random.choice(hit_choices)
            specs.append((current, _color_for_delta(delta)))
            current += delta

    assert len(specs) == n_channels
    return specs


def _render_grid(height: int, n_channels: int, channel_width: int, start_r: int, specs: list[tuple[int, int] | None]) -> np.ndarray:
    width = 1 + n_channels * channel_width + (n_channels - 1)
    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[start_r, 0] = Color.RED

    spans = _channel_spans(width, n_channels, channel_width)
    for i, (c0, c1) in enumerate(spans):
        if i != n_channels - 1:
            grid[:, c1 + 1] = Color.GREY
        spec = specs[i]
        if spec is not None:
            row, color = spec
            grid[row, c0 : c1 + 1] = color
    return grid


def _semantic_checks(grid: np.ndarray) -> None:
    start_r, channels, specs = _parse_grid(grid)
    records = _simulate(start_r, specs)

    assert len(channels) == len(specs)
    assert 5 <= len(specs) <= 7
    assert any(record["hit"] for record in records), "need at least one detour"
    assert any(not record["hit"] for record in records), "need at least one straight-through channel"
    assert any(record["bar"] is not None and not record["hit"] for record in records), "need an obstacle that is missed"
    assert any(record["bar"] is not None and record["bar"][0] == start_r and not record["hit"] for record in records), (
        "need a miss caused by moving away from the original row"
    )
    assert any(record["bar"] is not None and record["bar"][0] != start_r and record["hit"] for record in records), (
        "need a hit caused by accumulated displacement"
    )

    obstacle_colors = {bar[1] for bar in specs if bar is not None}
    assert obstacle_colors == {Color.BLUE, Color.GREEN}, "both obstacle colors must appear"

    influential_hits = 0
    for i, record in enumerate(records):
        if not record["hit"]:
            continue
        altered_specs = [None if j == i else spec for j, spec in enumerate(specs)]
        altered = _simulate(start_r, altered_specs)
        if any(records[j]["hit"] != altered[j]["hit"] for j in range(i + 1, len(specs))):
            influential_hits += 1
    assert influential_hits >= 2, "need multiple collisions to affect later channel classification"

    out = main(grid)
    fills = []
    for c0, c1 in channels:
        patch = out[:, c0 : c1 + 1]
        colors = {int(v) for v in np.unique(patch)}
        fills.append(Color.ORANGE in colors)
    assert any(fills) and not all(fills)


def generate_input() -> np.ndarray:
    for _ in range(400):
        n_channels = random.randint(5, 7)
        max_channel_width = min(4, 30 // n_channels - 1)
        channel_width = random.randint(2, max_channel_width)
        height = random.randint(10, 16)
        start_r = random.randint(2, height - 3)
        specs = _build_specs(height, n_channels, start_r)
        grid = _render_grid(height, n_channels, channel_width, start_r, specs)
        try:
            _semantic_checks(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid non-degenerate puzzle instance")
