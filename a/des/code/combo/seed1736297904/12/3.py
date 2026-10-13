from __future__ import annotations

import random

import numpy as np


BG = 0
SIGNAL_RED = 2
SIGNAL_GREEN = 3
SIGNAL_YELLOW = 4
GEN_SEPARATOR = 5
GEN_OBJECT_COLORS = (1, 6, 7, 8, 9)

GEN_WIDTH_RANGE = (13, 20)
GEN_CHANNEL_COUNT_RANGE = (3, 5)
GEN_CHANNEL_HEIGHT_RANGE = (4, 7)
GEN_OBJECT_COUNT_RANGE = (2, 4)
GEN_MAX_OBJECT_AREA = 6
GEN_SCENE_ATTEMPTS = 300
GEN_CHANNEL_ATTEMPTS = 120
GEN_OBJECT_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _separator_rows(g: np.ndarray) -> tuple[list[int], int]:
    rows: list[int] = []
    colors: list[int] = []
    for r in range(g.shape[0]):
        vals = np.unique(g[r, :])
        if vals.size == 1 and int(vals[0]) != BG:
            rows.append(int(r))
            colors.append(int(vals[0]))
    assert rows, "expected at least one full-width separator row"
    assert len(set(colors)) == 1, "all separator rows must share one color"
    return rows, colors[0]


def _channel_spans(height: int, separator_rows: list[int]) -> list[tuple[int, int]]:
    is_separator = np.zeros(height, dtype=bool)
    is_separator[separator_rows] = True
    spans: list[tuple[int, int]] = []
    start = None
    for r in range(height):
        if is_separator[r]:
            if start is not None:
                spans.append((start, r - 1))
                start = None
        elif start is None:
            start = r
    if start is not None:
        spans.append((start, height - 1))
    assert spans, "expected at least one non-separator channel"
    return spans


def _channel_signal(g: np.ndarray, r0: int, r1: int, separator_color: int) -> tuple[tuple[int, int], int]:
    signal_rows = [
        r
        for r in range(r0, r1 + 1)
        if int(g[r, 0]) not in (BG, int(separator_color))
    ]
    assert len(signal_rows) == 1, "each channel must have exactly one left-edge signal"
    signal_r = int(signal_rows[0])
    signal_color = int(g[signal_r, 0])
    assert signal_color in (SIGNAL_RED, SIGNAL_GREEN, SIGNAL_YELLOW), "unexpected signal color"
    return (signal_r, 0), signal_color


def _channel_components(
    g: np.ndarray,
    r0: int,
    r1: int,
    signal_rc: tuple[int, int],
    separator_color: int,
) -> list[dict]:
    seen = np.zeros((r1 - r0 + 1, g.shape[1]), dtype=bool)
    comps: list[dict] = []

    for r in range(r0, r1 + 1):
        for c in range(g.shape[1]):
            if (r, c) == signal_rc:
                continue
            color = int(g[r, c])
            if color in (BG, int(separator_color)):
                continue
            if seen[r - r0, c]:
                continue

            stack = [(r, c)]
            seen[r - r0, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (r0 <= nr <= r1 and 0 <= nc < g.shape[1]):
                        continue
                    if (nr, nc) == signal_rc:
                        continue
                    if seen[nr - r0, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr - r0, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "left": int(arr[:, 1].min()),
                    "top": int(arr[:, 0].min()),
                }
            )

    assert comps, "each channel must contain at least one foreground object"
    return comps


def _yellow_target_color(components: list[dict]) -> int:
    winner = min(
        components,
        key=lambda comp: (-int(comp["area"]), int(comp["left"]), int(comp["top"])),
    )
    return int(winner["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each horizontal channel contains a left-edge signal pixel.

    Green keeps every object in that channel unchanged, red erases every object,
    and yellow recolors every object to the color of the channel's largest object.
    Separator rows and the signal pixels themselves remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_rows, separator_color = _separator_rows(g)
    spans = _channel_spans(g.shape[0], separator_rows)

    out = g.copy()
    for r0, r1 in spans:
        signal_rc, signal_color = _channel_signal(g, r0, r1, separator_color)
        components = _channel_components(g, r0, r1, signal_rc, separator_color)

        if signal_color == SIGNAL_GREEN:
            continue
        if signal_color == SIGNAL_RED:
            for comp in components:
                coords = comp["coords"]
                out[coords[:, 0], coords[:, 1]] = BG
            continue

        target_color = _yellow_target_color(components)
        for comp in components:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = target_color

    return out


def _random_connected_mask(area: int, max_h: int, max_w: int) -> np.ndarray:
    candidates = [
        (h, w)
        for h in range(1, max_h + 1)
        for w in range(1, max_w + 1)
        if h * w >= area
    ]
    assert candidates, "at least one bounding box must fit the target area"
    slack_limit = random.choice((0, 1, 2, 3))
    tight = [(h, w) for h, w in candidates if h * w - area <= slack_limit]
    h, w = random.choice(tight or candidates)

    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _yellow_areas(n_objects: int) -> list[int]:
    if random.random() < 0.45:
        max_area = random.randint(3, GEN_MAX_OBJECT_AREA)
        areas = [max_area, max_area]
        while len(areas) < n_objects:
            areas.append(random.randint(1, max_area - 1))
    else:
        max_area = random.randint(3, GEN_MAX_OBJECT_AREA)
        areas = [max_area]
        while len(areas) < n_objects:
            areas.append(random.randint(1, max_area - 1))
    random.shuffle(areas)
    return areas


def _non_yellow_areas(n_objects: int) -> list[int]:
    areas = [random.randint(1, GEN_MAX_OBJECT_AREA) for _ in range(n_objects)]
    if max(areas) == 1:
        areas[random.randrange(n_objects)] = random.randint(2, GEN_MAX_OBJECT_AREA)
    return areas


def _sample_areas(signal_color: int, n_objects: int) -> list[int]:
    if signal_color == SIGNAL_YELLOW:
        return _yellow_areas(n_objects)
    return _non_yellow_areas(n_objects)


def _valid_yellow_layout(objects: list[dict]) -> bool:
    max_area = max(int(obj["area"]) for obj in objects)
    max_lefts = [int(obj["left"]) for obj in objects if int(obj["area"]) == max_area]
    return len(max_lefts) == len(set(max_lefts))


def _place_channel_objects(channel_h: int, width: int, signal_color: int) -> list[dict] | None:
    n_objects = random.randint(*GEN_OBJECT_COUNT_RANGE)
    colors = random.sample(GEN_OBJECT_COLORS, n_objects)
    areas = _sample_areas(signal_color, n_objects)
    specs = sorted(zip(areas, colors), key=lambda item: item[0], reverse=True)

    for _ in range(GEN_CHANNEL_ATTEMPTS):
        occupied = np.zeros((channel_h, width), dtype=bool)
        placed: list[dict] = []
        success = True

        for area, color in specs:
            found = None
            for _ in range(GEN_OBJECT_ATTEMPTS):
                mask = _random_connected_mask(area, min(3, channel_h), min(5, width - 2))
                h, w = mask.shape
                if h > channel_h or w > width - 2:
                    continue
                top = random.randint(0, channel_h - h)
                left = random.randint(2, width - w)
                trial = np.zeros_like(occupied)
                rr, cc = np.where(mask)
                trial[top + rr, left + cc] = True
                if np.any(_dilate8(occupied) & trial):
                    continue
                occupied |= trial
                coords = np.argwhere(trial)
                found = {
                    "color": int(color),
                    "coords": coords.copy(),
                    "area": int(area),
                    "left": int(coords[:, 1].min()),
                    "top": int(coords[:, 0].min()),
                }
                break
            if found is None:
                success = False
                break
            placed.append(found)

        if success:
            if signal_color == SIGNAL_YELLOW:
                if not _valid_yellow_layout(placed):
                    continue
                winner = _yellow_target_color(placed)
                if len({int(obj["color"]) for obj in placed}) < 2:
                    continue
                if all(int(obj["color"]) == winner for obj in placed):
                    continue
            return placed

    return None


def _render_scene(width: int, channel_heights: list[int], signals: list[int], channel_objects: list[list[dict]]) -> np.ndarray:
    total_h = sum(channel_heights) + len(channel_heights) - 1
    grid = np.full((total_h, width), BG, dtype=int)

    top = 0
    for idx, channel_h in enumerate(channel_heights):
        if idx > 0:
            grid[top - 1, :] = GEN_SEPARATOR
        signal_row = top + random.randrange(channel_h)
        grid[signal_row, 0] = int(signals[idx])

        for obj in channel_objects[idx]:
            coords = obj["coords"]
            abs_rows = top + coords[:, 0]
            abs_cols = coords[:, 1]
            grid[abs_rows, abs_cols] = int(obj["color"])

        top += channel_h + 1

    return grid


def _expected_output(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    separator_rows, separator_color = _separator_rows(inp)
    spans = _channel_spans(inp.shape[0], separator_rows)

    for r0, r1 in spans:
        signal_rc, signal_color = _channel_signal(inp, r0, r1, separator_color)
        components = _channel_components(inp, r0, r1, signal_rc, separator_color)
        if signal_color == SIGNAL_GREEN:
            continue
        if signal_color == SIGNAL_RED:
            for comp in components:
                coords = comp["coords"]
                out[coords[:, 0], coords[:, 1]] = BG
            continue
        target_color = _yellow_target_color(components)
        for comp in components:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = target_color

    return out


def _signal_list(n_channels: int) -> list[int]:
    signals = [SIGNAL_RED, SIGNAL_GREEN, SIGNAL_YELLOW]
    while len(signals) < n_channels:
        signals.append(random.choice((SIGNAL_RED, SIGNAL_GREEN, SIGNAL_YELLOW)))
    random.shuffle(signals)
    return signals


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*GEN_WIDTH_RANGE)
    n_channels = random.randint(*GEN_CHANNEL_COUNT_RANGE)
    channel_heights = [random.randint(*GEN_CHANNEL_HEIGHT_RANGE) for _ in range(n_channels)]
    signals = _signal_list(n_channels)

    channel_objects: list[list[dict]] = []
    for signal_color, channel_h in zip(signals, channel_heights):
        objects = _place_channel_objects(channel_h, width, signal_color)
        if objects is None:
            return None
        channel_objects.append(objects)

    inp = _render_scene(width, channel_heights, signals, channel_objects)
    expected = _expected_output(inp)

    if not np.array_equal(main(inp), expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_SCENE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid channel-signal ARC input")
