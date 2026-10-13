from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg


_RNG = np.random.default_rng()

BG = Color.BLACK
WATER = Color.TEAL
OBSTACLE = Color.GREY
MARKER = Color.RED

_GEN_RETRY_LIMIT = 500
_HEIGHT_RANGE = (13, 24)
_WIDTH_RANGE = (16, 29)
_SOURCE_GAP = 3


def _rows_from_component(coords: np.ndarray, width: int) -> np.ndarray:
    rows = np.full(width, -1, dtype=int)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        assert 0 <= cc < width
        assert rows[cc] == -1, "each path must occupy exactly one cell per column"
        rows[cc] = rr
    assert np.all(rows >= 0), "each path must span from the left edge to the right edge"
    diffs = np.diff(rows)
    assert np.all(np.abs(diffs) <= 1), "path steps must be horizontal or diagonal"
    return rows


def _extract_path_rows(grid: np.ndarray) -> np.ndarray:
    water_components = components_mask(grid == WATER, connectivity=8)
    assert len(water_components) >= 2, "expected multiple water paths"

    width = grid.shape[1]
    paths = []
    for comp in water_components:
        rows = _rows_from_component(comp, width)
        assert rows[0] >= 0 and rows[-1] >= 0
        paths.append(rows)

    paths.sort(key=lambda arr: int(arr[0]))
    row_matrix = np.vstack(paths)
    assert len(set(map(int, row_matrix[:, 0]))) == row_matrix.shape[0], "source rows must be distinct"
    return row_matrix


def _hidden_events_from_rows(row_matrix: np.ndarray) -> dict[int, int]:
    hidden: dict[int, int] = {}
    deltas = row_matrix[:, 1:] - row_matrix[:, :-1]
    for c in range(1, row_matrix.shape[1]):
        step = deltas[:, c - 1]
        if np.all(step == step[0]) and abs(int(step[0])) == 1:
            hidden[c] = int(step[0])
    return hidden


def _simulate_one_path(
    source_row: int,
    obstacle_mask: np.ndarray,
    *,
    hidden_events: dict[int, int] | None = None,
) -> np.ndarray:
    hidden_events = hidden_events or {}
    height, width = obstacle_mask.shape
    rows = np.full(width, -1, dtype=int)
    base_row = int(source_row)
    rows[0] = base_row

    c = 1
    while c < width:
        if c in hidden_events:
            base_row += int(hidden_events[c])
            assert 0 <= base_row < height
            assert not obstacle_mask[base_row, c], "hidden shift cannot land on an obstacle"
            rows[c] = base_row
            c += 1
            continue

        if obstacle_mask[base_row, c]:
            detour_row = base_row + 1
            assert detour_row < height, "obstacle detour must fit below the path"

            first_free = c
            while first_free < width and obstacle_mask[base_row, first_free]:
                first_free += 1

            assert first_free < width - 1, "detour must have room to rejoin before the edge"
            assert not obstacle_mask[detour_row, c : first_free + 1].any(), "detour lane must stay clear"

            rows[c : first_free + 1] = detour_row
            c = first_free + 1
            rows[c] = base_row
            c += 1
            continue

        rows[c] = base_row
        c += 1

    assert np.all(rows >= 0)
    return rows


def _render_world(
    shape: tuple[int, int],
    source_rows: np.ndarray,
    obstacle_mask: np.ndarray,
    *,
    hidden_events: dict[int, int] | None = None,
    add_markers: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    out = np.full((height, width), BG, dtype=int)
    out[obstacle_mask] = OBSTACLE

    hidden_events = hidden_events or {}
    if add_markers:
        for c, step in hidden_events.items():
            out[0 if step > 0 else height - 1, int(c)] = MARKER

    path_rows = []
    cols = np.arange(width, dtype=int)
    for source_row in source_rows:
        rows = _simulate_one_path(int(source_row), obstacle_mask, hidden_events=hidden_events)
        assert not obstacle_mask[rows, cols].any()
        out[rows, cols] = WATER
        path_rows.append(rows)

    row_matrix = np.vstack(path_rows)
    return out, row_matrix


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the shared hidden deflection columns, keep the grey obstacle bypasses, and mark the erased deflectors on the border.

    Input structure:
    - The background is black.
    - Teal pixels form several separate left-to-right 8-connected water paths, each with one teal source on the left edge.
    - Grey pixels are fixed obstacle cells, and red is absent from the input.

    Rule:
    - Read each teal path as a one-cell-wide eastward trace with exactly one teal cell in every column.
    - Any column where every path shifts by the same one-row diagonal step is a hidden deflector; mark that column in red on the top edge for a downward shift or on the bottom edge for an upward shift.
    - Rebuild every teal path from its source row with those shared deflectors removed: go straight right, and whenever the next cell on the current row is grey, bypass the whole contiguous grey run by stepping down-right, moving right underneath it, and stepping up-right back to the original row.
    - Keep every grey obstacle in place.

    Key insight:
    - Visible obstacle bends are local down-then-up bulges that affect only some paths, while hidden deflectors are the only bends that happen in the same column and direction for every path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG, "expected black background"
    assert not np.any(g == MARKER), "input must not already contain red markers"

    row_matrix = _extract_path_rows(g)
    hidden_events = _hidden_events_from_rows(row_matrix)
    source_rows = row_matrix[:, 0].copy()
    obstacle_mask = g == OBSTACLE

    out, _ = _render_world(
        g.shape,
        source_rows,
        obstacle_mask,
        hidden_events={},
        add_markers=False,
    )
    for c, step in hidden_events.items():
        out[0 if step > 0 else g.shape[0] - 1, int(c)] = MARKER
    return out


def _offset_profile(width: int, hidden_events: dict[int, int]) -> np.ndarray:
    offsets = np.zeros(width, dtype=int)
    running = 0
    for c in range(width):
        if c in hidden_events:
            running += int(hidden_events[c])
        offsets[c] = running
    return offsets


def _pick_spaced_values(values: list[int], count: int, gap: int) -> list[int] | None:
    if count == 0:
        return []
    shuffled = values[:]
    _RNG.shuffle(shuffled)
    picked: list[int] = []
    for value in shuffled:
        if all(abs(int(value) - int(prev)) >= gap for prev in picked):
            picked.append(int(value))
            if len(picked) == count:
                return sorted(picked)
    return None


def _sample_source_rows(height: int) -> np.ndarray | None:
    candidates = list(range(2, height - 3))
    max_paths = min(4, 1 + (len(candidates) - 1) // _SOURCE_GAP)
    if max_paths < 2:
        return None
    num_paths = int(_RNG.integers(2, max_paths + 1))
    picked = _pick_spaced_values(candidates, num_paths, _SOURCE_GAP)
    if picked is None:
        return None
    return np.array(sorted(picked), dtype=int)


def _sample_hidden_events(height: int, width: int, source_rows: np.ndarray) -> dict[int, int] | None:
    candidate_cols = list(range(2, width - 3))
    max_events = min(3, max(1, len(candidate_cols) // 4))
    num_events = int(_RNG.integers(1, max_events + 1))
    cols = _pick_spaced_values(candidate_cols, num_events, 4)
    if cols is None:
        return None

    top = int(source_rows.min())
    bottom = int(source_rows.max())
    hidden: dict[int, int] = {}
    running = 0
    for c in cols:
        valid_steps = []
        for step in (-1, 1):
            shifted_top = top + running + step
            shifted_bottom = bottom + running + step
            if 1 <= shifted_top and shifted_bottom <= height - 3:
                valid_steps.append(step)
        if not valid_steps:
            return None
        step = int(valid_steps[int(_RNG.integers(len(valid_steps)))])
        hidden[int(c)] = step
        running += step
    return hidden


def _sample_obstacles(
    shape: tuple[int, int],
    source_rows: np.ndarray,
    hidden_events: dict[int, int],
) -> np.ndarray | None:
    height, width = shape
    offsets = _offset_profile(width, hidden_events)
    obstacle_mask = np.zeros((height, width), dtype=bool)

    allowed_cols = [
        c
        for c in range(1, width - 2)
        if all(abs(c - hidden_col) >= 3 for hidden_col in hidden_events)
    ]
    if not allowed_cols:
        return None

    changed_cols = [c for c in allowed_cols if offsets[c] != 0]
    max_obstacles = min(4, len(allowed_cols))
    num_obstacles = int(_RNG.integers(1, max_obstacles + 1))

    selected_cols: list[int] = []
    if changed_cols:
        first_col = int(changed_cols[int(_RNG.integers(len(changed_cols)))])
        selected_cols.append(first_col)

    remaining = [
        c
        for c in allowed_cols
        if c not in selected_cols and all(abs(c - picked) >= 4 for picked in selected_cols)
    ]
    extra = _pick_spaced_values(remaining, num_obstacles - len(selected_cols), 4)
    if num_obstacles > len(selected_cols) and extra is None:
        return None
    if extra:
        selected_cols.extend(extra)
    selected_cols = sorted(selected_cols)

    if len(selected_cols) != num_obstacles:
        return None

    changed_used = False
    for index, c in enumerate(selected_cols):
        path_idx = int(_RNG.integers(len(source_rows)))
        source_row = int(source_rows[path_idx])
        input_row = source_row + int(offsets[c])

        if offsets[c] == 0:
            row = source_row
        else:
            if index == len(selected_cols) - 1 and not changed_used:
                kind = "output_only"
            else:
                kind = ("input_only", "output_only")[int(_RNG.integers(2))]
            row = input_row if kind == "input_only" else source_row
            changed_used = True

        obstacle_mask[row, c] = True

    return obstacle_mask


def _path_count_is_valid(grid: np.ndarray, path_count: int) -> bool:
    return len(components_mask(grid == WATER, connectivity=8)) == path_count


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(_HEIGHT_RANGE[0], _HEIGHT_RANGE[1] + 1))
    width = int(_RNG.integers(_WIDTH_RANGE[0], _WIDTH_RANGE[1] + 1))

    source_rows = _sample_source_rows(height)
    if source_rows is None:
        return None

    hidden_events = _sample_hidden_events(height, width, source_rows)
    if not hidden_events:
        return None

    obstacle_mask = _sample_obstacles((height, width), source_rows, hidden_events)
    if obstacle_mask is None or not obstacle_mask.any():
        return None

    inp, input_rows = _render_world(
        (height, width),
        source_rows,
        obstacle_mask,
        hidden_events=hidden_events,
        add_markers=False,
    )
    out, output_rows = _render_world(
        (height, width),
        source_rows,
        obstacle_mask,
        hidden_events={},
        add_markers=False,
    )
    for c, step in hidden_events.items():
        out[0 if step > 0 else height - 1, int(c)] = MARKER

    if not _path_count_is_valid(inp, len(source_rows)):
        return None
    if not _path_count_is_valid(out, len(source_rows)):
        return None
    if np.any(inp == MARKER):
        return None
    if len(_hidden_events_from_rows(input_rows)) != len(hidden_events):
        return None
    if _hidden_events_from_rows(input_rows) != hidden_events:
        return None
    if np.array_equal(inp, out):
        return None
    if np.array_equal(input_rows, output_rows):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid water-deflection sample")
