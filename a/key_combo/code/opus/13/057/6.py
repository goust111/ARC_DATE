import itertools
import random

import numpy as np


BG = 0
MARKER = 2
SEPARATOR = 5
STAMP = 7
PATH = 8

TILE_ROW_CHOICES = (3, 4)
TILE_COL_COUNT = 3
TILE_H_CHOICES = (3, 4)
TILE_W_CHOICES = (4, 5)
PATTERN_AREA_CHOICES = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 500


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for idx in sorted(separators):
        assert start <= idx - 1
        spans.append((start, idx - 1))
        start = idx + 1
    assert start <= length - 1
    spans.append((start, length - 1))
    return spans


def _tile_spans(tile_count: int, tile_size: int) -> list[tuple[int, int]]:
    return [(k * (tile_size + 1), k * (tile_size + 1) + tile_size - 1) for k in range(tile_count)]


def _parse_grid(grid: np.ndarray, *, require_controls: bool = True) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    height, width = map(int, g.shape)
    assert height >= 2
    assert np.all(np.isin(g, [BG, MARKER, SEPARATOR, STAMP, PATH]))

    control_row = height - 1
    assert set(map(int, np.unique(g[control_row]))) <= {BG, PATH}
    controls = sorted(int(c) for c in np.flatnonzero(g[control_row] == PATH))
    if require_controls:
        assert 1 <= len(controls) <= 3
    else:
        assert len(controls) == 0

    body = g[:-1, :]
    sep_rows = [r for r in range(body.shape[0]) if np.all(body[r] == SEPARATOR)]
    sep_cols = [c for c in range(body.shape[1]) if np.all(body[:, c] == SEPARATOR)]
    row_spans = _spans(body.shape[0], sep_rows)
    col_spans = _spans(body.shape[1], sep_cols)
    assert len(row_spans) in (3, 4)
    assert len(col_spans) == 3

    tile_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    tile_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(tile_heights) == 1 and len(tile_widths) == 1
    tile_h = tile_heights.pop()
    tile_w = tile_widths.pop()

    source = None
    markers = []
    for tr, (r0, r1) in enumerate(row_spans):
        for tc, (c0, c1) in enumerate(col_spans):
            tile = body[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(tile)))
            has_stamp = STAMP in colors
            has_marker = MARKER in colors
            if has_stamp:
                assert not has_marker
                assert colors <= {BG, STAMP}
                assert source is None
                source = {"tile_rc": (tr, tc), "origin": (r0, c0), "tile": tile.copy()}
            elif has_marker:
                assert colors <= {BG, MARKER}
                assert int(np.count_nonzero(tile == MARKER)) == 1
                markers.append({"tile_rc": (tr, tc), "origin": (r0, c0)})
            else:
                assert colors <= {BG}

    assert source is not None
    assert len(markers) >= 2
    assert source["tile_rc"][0] != len(row_spans) - 1
    assert any(m["tile_rc"][0] == len(row_spans) - 1 for m in markers)
    assert any(m["tile_rc"][0] != len(row_spans) - 1 for m in markers)
    assert all(c not in sep_cols for c in controls)

    return {
        "grid": g,
        "body": body,
        "controls": controls,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "source": source,
        "markers": markers,
    }


def _apply_stamp(meta: dict) -> tuple[np.ndarray, set[tuple[int, int]]]:
    stamped = meta["grid"].copy()
    source_tile = meta["source"]["tile"]
    tile_h, tile_w = source_tile.shape
    copied_orange = set()
    orange_offsets = [(r, c) for r, c in zip(*np.where(source_tile == STAMP))]
    for marker in meta["markers"]:
        r0, c0 = marker["origin"]
        stamped[r0 : r0 + tile_h, c0 : c0 + tile_w] = source_tile
        for dr, dc in orange_offsets:
            copied_orange.add((int(r0 + dr), int(c0 + dc)))
    return stamped, copied_orange


def _classify_obstacle(r: int, c: int, color: int, copied_orange: set[tuple[int, int]]) -> str:
    if color == SEPARATOR:
        return "separator"
    if color == STAMP:
        return "orange_copy" if (int(r), int(c)) in copied_orange else "orange_source"
    if color == PATH:
        return "prior_teal"
    raise AssertionError(f"unexpected obstacle color {color}")


def _simulate_paths(
    base_grid: np.ndarray,
    controls: list[int],
    copied_orange: set[tuple[int, int]],
    *,
    prior_teal_as_obstacle: bool,
) -> tuple[np.ndarray, list[dict]]:
    base = np.asarray(base_grid, dtype=int)
    out = base.copy()
    traces = []
    view = base.copy()
    for c in controls:
        out[-1, c] = PATH
        view[-1, c] = PATH

    for start_col in sorted(int(c) for c in controls):
        r = out.shape[0] - 1
        c = int(start_col)
        path_cells = []
        events = []
        while True:
            if r == 0:
                stop = "top"
                break
            probe = out if prior_teal_as_obstacle else view
            above = int(probe[r - 1, c])
            if above == BG:
                r -= 1
                out[r, c] = PATH
                path_cells.append((r, c))
                continue
            if above in (SEPARATOR, STAMP) or (prior_teal_as_obstacle and above == PATH):
                events.append(
                    {
                        "row": int(r - 1),
                        "col": int(c),
                        "kind": _classify_obstacle(r - 1, c, above, copied_orange),
                    }
                )
                if c + 1 >= out.shape[1] or int(probe[r, c + 1]) != BG:
                    stop = "blocked_right"
                    break
                c += 1
                out[r, c] = PATH
                path_cells.append((r, c))
                continue
            raise AssertionError(f"unexpected color above path: {above}")
        traces.append(
            {
                "start_col": int(start_col),
                "cells": path_cells,
                "events": events,
                "stop": stop,
            }
        )
    return out, traces


def _erase_markers(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == MARKER] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the source tile into marked cells, then grow bottom control paths that deflect under separators and stamped obstacles.

    Input structure:
    - A black grid is partitioned by full gray separator rows and columns into 3 or 4 tile rows and 3 tile columns, plus one extra bottom control row.
    - Exactly one non-bottom tile contains an orange template pattern on black.
    - Other tiles are either empty black or contain a single red marker pixel, and the bottom extra row contains 1 to 3 teal control pixels.

    Rule:
    - First copy the entire template tile into every marked tile, replacing each red marker with the template's black/orange contents.
    - Then process the teal controls from left to right: move upward through black cells, but if the cell above is gray, orange, or a previously grown teal cell, step one cell right on the current row instead; stop when that right step is unavailable or non-black.
    - Keep the original template, separators, and control pixels in place while adding the stamped copies and grown teal path cells.

    Key insight:
    - The true obstacle field is not visible directly in the input: it must be constructed by stamping the template into the red-marked tiles before simulating the locally deflected upward paths.
    """
    meta = _parse_grid(grid)
    stamped, copied_orange = _apply_stamp(meta)
    out, _ = _simulate_paths(stamped, meta["controls"], copied_orange, prior_teal_as_obstacle=True)
    return out


def _random_connected_pattern(tile_h: int, tile_w: int) -> np.ndarray:
    area = random.choice(PATTERN_AREA_CHOICES)
    for _ in range(100):
        cells = {(random.randrange(tile_h), random.randrange(tile_w))}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            neighbors = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < tile_h and 0 <= cc < tile_w and (rr, cc) not in cells:
                    neighbors.append((rr, cc))
            if not neighbors:
                break
            cells.add(random.choice(neighbors))
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(cells) != area or len(rows) < 2 or len(cols) < 2:
            continue
        pattern = np.zeros((tile_h, tile_w), dtype=int)
        for r, c in cells:
            pattern[r, c] = STAMP
        return pattern
    raise RuntimeError("failed to build a connected template pattern")


def _build_blank_grid(tile_rows: int, tile_cols: int, tile_h: int, tile_w: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans = _tile_spans(tile_rows, tile_h)
    col_spans = _tile_spans(tile_cols, tile_w)
    body_h = row_spans[-1][1] + 1
    body_w = col_spans[-1][1] + 1
    grid = np.zeros((body_h + 1, body_w), dtype=int)
    for k in range(1, tile_rows):
        grid[k * tile_h + (k - 1), :] = SEPARATOR
    for k in range(1, tile_cols):
        grid[:body_h, k * tile_w + (k - 1)] = SEPARATOR
    return grid, row_spans, col_spans


def _candidate_control_sets(interior_cols: list[int], need_teal_interaction: bool) -> list[tuple[int, ...]]:
    sizes = [2, 3] if need_teal_interaction else [1, 2, 3]
    combos = []
    for size in sizes:
        for combo in itertools.combinations(interior_cols, size):
            if any(b - a <= 0 for a, b in zip(combo, combo[1:])):
                continue
            combos.append(combo)
    random.shuffle(combos)
    return combos


def _choose_controls(base_grid: np.ndarray) -> tuple[list[int], np.ndarray] | None:
    meta = _parse_grid(base_grid, require_controls=False)
    stamped, copied_orange = _apply_stamp(meta)
    sep_cols = {c for c in range(base_grid.shape[1]) if np.all(base_grid[:-1, c] == SEPARATOR)}
    interior_cols = [c for c in range(base_grid.shape[1]) if c not in sep_cols]

    for need_teal_interaction in (random.random() < 0.65, False):
        valid = []
        for combo in _candidate_control_sets(interior_cols, need_teal_interaction):
            test_grid = base_grid.copy()
            test_grid[-1, list(combo)] = PATH
            true_out = main(test_grid)

            stamped_with_controls, copied_orange = _apply_stamp(_parse_grid(test_grid))
            traced_out, traces = _simulate_paths(
                stamped_with_controls,
                list(combo),
                copied_orange,
                prior_teal_as_obstacle=True,
            )
            assert np.array_equal(true_out, traced_out)

            no_stamp_base = _erase_markers(test_grid)
            no_stamp_out, _ = _simulate_paths(
                no_stamp_base,
                list(combo),
                set(),
                prior_teal_as_obstacle=True,
            )
            if np.array_equal(true_out, no_stamp_out):
                continue

            no_teal_out, _ = _simulate_paths(
                stamped_with_controls,
                list(combo),
                copied_orange,
                prior_teal_as_obstacle=False,
            )
            if need_teal_interaction and np.array_equal(true_out, no_teal_out):
                continue

            event_kinds = [event["kind"] for trace in traces for event in trace["events"]]
            if "orange_copy" not in event_kinds:
                continue
            if "separator" not in event_kinds:
                continue
            if len(combo) == 1 and need_teal_interaction:
                continue

            valid.append((combo, true_out))
        if valid:
            combo, out = random.choice(valid)
            return list(combo), out
    return None


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    tile_rows = random.choice(TILE_ROW_CHOICES)
    tile_cols = TILE_COL_COUNT
    tile_h = random.choice(TILE_H_CHOICES)
    tile_w = random.choice(TILE_W_CHOICES)

    grid, row_spans, col_spans = _build_blank_grid(tile_rows, tile_cols, tile_h, tile_w)
    template = _random_connected_pattern(tile_h, tile_w)

    all_tiles = [(tr, tc) for tr in range(tile_rows) for tc in range(tile_cols)]
    upper_tiles = [pos for pos in all_tiles if pos[0] < tile_rows - 1]
    bottom_tiles = [pos for pos in all_tiles if pos[0] == tile_rows - 1]
    template_pos = random.choice(upper_tiles)

    bottom_marker_count = random.choice((1, 1, 2))
    upper_marker_count = random.choice((1, 2))
    available_upper = [pos for pos in upper_tiles if pos != template_pos]
    if len(available_upper) < upper_marker_count:
        upper_marker_count = len(available_upper)
    marker_tiles = random.sample(bottom_tiles, bottom_marker_count) + random.sample(available_upper, upper_marker_count)
    if len(marker_tiles) < 2:
        return None

    src_r0, src_c0 = row_spans[template_pos[0]][0], col_spans[template_pos[1]][0]
    grid[src_r0 : src_r0 + tile_h, src_c0 : src_c0 + tile_w] = template

    for tr, tc in marker_tiles:
        r0, c0 = row_spans[tr][0], col_spans[tc][0]
        rr = random.randrange(tile_h)
        cc = random.randrange(tile_w)
        grid[r0 + rr, c0 + cc] = MARKER

    chosen = _choose_controls(grid)
    if chosen is None:
        return None
    controls, expected = chosen
    grid[-1, controls] = PATH

    # Extra semantic checks beyond solver self-consistency.
    parsed = _parse_grid(grid)
    stamped, copied_orange = _apply_stamp(parsed)
    solved, traces = _simulate_paths(stamped, parsed["controls"], copied_orange, prior_teal_as_obstacle=True)
    if not np.array_equal(solved, expected):
        return None
    no_stamp_out, _ = _simulate_paths(
        _erase_markers(grid),
        parsed["controls"],
        set(),
        prior_teal_as_obstacle=True,
    )
    if np.array_equal(expected, no_stamp_out):
        return None
    if not any(trace["stop"] == "blocked_right" for trace in traces):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            grid, _ = result
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
