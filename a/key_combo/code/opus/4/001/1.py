from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
OBSTACLE_COLOR = 2
SOURCE_COLOR = 8


SPECS = [
    {
        "source_offsets": (-3, 0, 2),
        "base_spans": [(-2, 2), (-3, 3)],
        "row_modes": ("opp", "same"),
    },
    {
        "source_offsets": (-3, 0, 2, 5),
        "base_spans": [(-2, 2), (3, 5), (-3, -1)],
        "row_modes": ("opp", "same", "same"),
    },
    {
        "source_offsets": (-3, 0, 2, 7),
        "base_spans": [(-2, 2), (3, 7), (-3, -1)],
        "row_modes": ("opp", "same", "same"),
    },
    {
        "source_offsets": (-3, 0, 2, 5, 7),
        "base_spans": [(-2, 2), (3, 7), (-3, -1)],
        "row_modes": ("opp", "same", "same"),
    },
    {
        "source_offsets": (-3, 0, 2, 4, 7),
        "base_spans": [(-2, 2), (3, 7), (-3, -1)],
        "row_modes": ("opp", "same", "same"),
    },
]


def _source_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in np.flatnonzero(grid[0] == SOURCE_COLOR)]


def _obstacle_segments(grid: np.ndarray) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    h, w = grid.shape
    for r in range(1, h):
        c = 0
        while c < w:
            if int(grid[r, c]) != OBSTACLE_COLOR:
                c += 1
                continue
            c0 = c
            while c + 1 < w and int(grid[r, c + 1]) == OBSTACLE_COLOR:
                c += 1
            segments.append((int(r), int(c0), int(c)))
            c += 1
    segments.sort()
    return segments


def _detour_target(entry_col: int, c0: int, c1: int, width: int) -> tuple[int, bool]:
    choices: list[tuple[int, int, int]] = []
    if c0 > 0:
        left = c0 - 1
        choices.append((abs(entry_col - left), 1, left))
    if c1 < width - 1:
        right = c1 + 1
        choices.append((abs(entry_col - right), 0, right))
    if not choices:
        raise ValueError("obstacle must leave at least one outside detour cell")
    dist, side_rank, target = min(choices)
    tie = len(choices) == 2 and choices[0][0] == choices[1][0]
    if tie:
        right_target = max(choices, key=lambda item: item[2])[2]
        return int(right_target), True
    return int(target), False


def _trace_source(
    shape: tuple[int, int],
    source_col: int,
    segments: list[tuple[int, int, int]],
) -> tuple[list[tuple[int, int]], list[dict[str, int | bool | None]]]:
    h, w = map(int, shape)
    current_col = int(source_col)
    current_row = 1
    cells: list[tuple[int, int]] = []
    events: list[dict[str, int | bool | None]] = []

    for row, c0, c1 in segments:
        if row < current_row or not (c0 <= current_col <= c1):
            continue

        for rr in range(current_row, row):
            cells.append((int(rr), int(current_col)))

        source_match = (source_col % 2) == (row % 2)
        current_match = (current_col % 2) == (row % 2)
        event: dict[str, int | bool | None] = {
            "row": int(row),
            "c0": int(c0),
            "c1": int(c1),
            "source_col": int(source_col),
            "entry_col": int(current_col),
            "source_parity_match": bool(source_match),
            "current_parity_match": bool(current_match),
            "transparent": bool(source_match),
            "target_col": None,
            "tie": False,
        }

        if source_match:
            current_row = int(row) + 1
            events.append(event)
            continue

        target_col, tie = _detour_target(current_col, c0, c1, w)
        step = 1 if target_col > current_col else -1
        for cc in range(current_col + step, target_col + step, step):
            cells.append((int(row - 1), int(cc)))

        event["target_col"] = int(target_col)
        event["tie"] = bool(tie)
        events.append(event)
        current_col = int(target_col)
        current_row = int(row)

    for rr in range(current_row, h):
        cells.append((int(rr), int(current_col)))

    return cells, events


def _render_paths(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    segments = _obstacle_segments(out)
    for source_col in _source_columns(out):
        cells, _events = _trace_source(out.shape, source_col, segments)
        for r, c in cells:
            if int(out[r, c]) == BG_COLOR:
                out[r, c] = SOURCE_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend top-row source columns downward while parity-gating obstacle transparency and detours.

    Input structure:
    - A black background grid with 3-5 teal source pixels on the top row.
    - The lower area contains 2-3 red horizontal obstacle segments on distinct rows.
    - Each source column has at least one obstacle somewhere below its original column.

    Rule:
    - Copy the input, keep every red obstacle unchanged, then process sources from left to right.
    - From each source, draw teal downward toward the bottom; when its path meets an obstacle under the current path column,
      compare the obstacle row parity with the original source column parity.
    - Matching parity makes the obstacle transparent, so the path continues straight through while the red cell stays visible.
    - Mismatching parity makes the obstacle solid, so the path stops one row above, detours horizontally to the nearer outside end
      of the obstacle (breaking ties to the right), then continues vertically from that outside column.

    Key insight:
    - The gate depends on the source column's original parity, not on the path's later detoured column, so one path can detour and
      still pass through a later obstacle whose row parity matches the original source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _render_paths(g)


def _pick_focus(spec: dict, width: int) -> int | None:
    source_offsets = spec["source_offsets"]
    obs_starts = [span[0] for span in spec["base_spans"]]
    obs_ends = [span[1] for span in spec["base_spans"]]
    lo = max(2 - min(source_offsets), 1 - min(obs_starts))
    hi = min(width - 3 - max(source_offsets), width - 2 - max(obs_ends))
    if lo > hi:
        return None
    return random.randint(int(lo), int(hi))


def _pick_rows(height: int, focus_parity: int, row_modes: tuple[str, ...]) -> list[int] | None:
    rows: list[int] = []
    low = 4
    tail_margin = 2
    for idx, mode in enumerate(row_modes):
        high = height - (len(row_modes) - idx) * 3 + 1 - tail_margin
        parity = focus_parity if mode == "same" else 1 - focus_parity
        choices = [r for r in range(low, high + 1) if r % 2 == parity]
        if not choices:
            return None
        row = random.choice(choices)
        rows.append(int(row))
        low = int(row) + 3
    return rows


def _expand_span(base_start: int, base_end: int, width: int, max_len: int = 7) -> tuple[int, int]:
    start = int(base_start)
    end = int(base_end)
    candidates = [(start, end)]
    for left_extra in range(0, 2):
        for right_extra in range(0, 2):
            c0 = start - left_extra
            c1 = end + right_extra
            if c0 < 1 or c1 > width - 2:
                continue
            if c1 - c0 + 1 > max_len:
                continue
            candidates.append((int(c0), int(c1)))
    return random.choice(candidates)


def _build_grid(spec: dict, height: int, width: int, focus: int, rows: list[int]) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)

    for offset in spec["source_offsets"]:
        grid[0, focus + int(offset)] = SOURCE_COLOR

    for idx, (row, (span_start, span_end)) in enumerate(zip(rows, spec["base_spans"])):
        c0, c1 = focus + int(span_start), focus + int(span_end)
        if idx == 0:
            left = c0
            right = c1
        else:
            left, right = _expand_span(c0, c1, width)
        grid[int(row), int(left) : int(right) + 1] = OBSTACLE_COLOR

    return grid


def _validate_scene(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(np.unique(grid)) - {BG_COLOR, OBSTACLE_COLOR, SOURCE_COLOR}:
        return False

    h, w = map(int, grid.shape)
    sources = _source_columns(grid)
    if not (3 <= len(sources) <= 5):
        return False
    if np.count_nonzero(grid == SOURCE_COLOR) != len(sources):
        return False
    if any(b - a < 2 for a, b in zip(sources, sources[1:])):
        return False
    if len({c % 2 for c in sources}) < 2:
        return False

    segments = _obstacle_segments(grid)
    if not (2 <= len(segments) <= 3):
        return False
    if len({row for row, _c0, _c1 in segments}) != len(segments):
        return False
    if any(row < 3 or row > h - 2 for row, _c0, _c1 in segments):
        return False
    if any(segments[i + 1][0] - segments[i][0] < 3 for i in range(len(segments) - 1)):
        return False
    if any(c0 <= 0 or c1 >= w - 1 for _row, c0, c1 in segments):
        return False
    if any(not (3 <= c1 - c0 + 1 <= 7) for _row, c0, c1 in segments):
        return False
    if len({row % 2 for row, _c0, _c1 in segments}) < 2:
        return False
    if any(not any(c0 <= source <= c1 for _row, c0, c1 in segments) for source in sources):
        return False

    per_source_events: dict[int, list[dict[str, int | bool | None]]] = {}
    all_events: list[dict[str, int | bool | None]] = []
    for source in sources:
        cells, events = _trace_source(grid.shape, source, segments)
        if not cells:
            return False
        per_source_events[int(source)] = events
        all_events.extend(events)

    if not all_events:
        return False
    blocks = [event for event in all_events if not bool(event["transparent"])]
    transparents = [event for event in all_events if bool(event["transparent"])]
    if not blocks or not transparents:
        return False
    if not any(int(event["target_col"]) > int(event["entry_col"]) and bool(event["tie"]) for event in blocks):
        return False
    if not any(int(event["target_col"]) < int(event["entry_col"]) for event in blocks):
        return False
    if not any(len(events) >= 2 for events in per_source_events.values()):
        return False
    if not any(
        bool(event["transparent"]) and (int(event["entry_col"]) % 2 != int(event["source_col"]) % 2)
        for event in all_events
    ):
        return False
    if not any(
        bool(event["source_parity_match"]) != bool(event["current_parity_match"])
        for event in all_events
    ):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.all(out[grid == OBSTACLE_COLOR] == OBSTACLE_COLOR):
        return False
    if np.count_nonzero(out == SOURCE_COLOR) <= len(sources):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    spec = random.choice(SPECS)
    obstacle_count = len(spec["base_spans"])

    width_low = 16 if obstacle_count == 2 else 18
    width_high = 24
    height_low = 14 if obstacle_count == 2 else 17
    height_high = 22

    width = random.randint(width_low, width_high)
    height = random.randint(height_low, height_high)
    focus = _pick_focus(spec, width)
    if focus is None:
        return None

    rows = _pick_rows(height, focus % 2, spec["row_modes"])
    if rows is None:
        return None

    grid = _build_grid(spec, height, width, focus, rows)
    if not _validate_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-gated gravity scene")
