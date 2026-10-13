from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
ARROW_COLOR = 2
GROUND_COLOR = 3
ANCHOR_COLOR = 5
BODY_COLORS = [1, 4, 6, 7, 8, 9]

LEFT_ARROW = np.array(
    [
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 0],
    ],
    dtype=bool,
)
RIGHT_ARROW = np.array(
    [
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
    ],
    dtype=bool,
)

MAX_ATTEMPTS = 400
MODE_CYCLE = ("anchor", "short", "end")
_generate_counter = 0


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_arrow(grid: np.ndarray) -> tuple[int, set[tuple[int, int]]]:
    coords = np.argwhere(grid == ARROW_COLOR)
    assert coords.shape[0] == 5, "expected exactly one 5-cell arrow sprite"
    r0, c0, r1, c1 = _bbox(coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "arrow bbox must be 3x3"
    mask = grid[r0 : r1 + 1, c0 : c1 + 1] == ARROW_COLOR
    if np.array_equal(mask, RIGHT_ARROW):
        direction = 1
    elif np.array_equal(mask, LEFT_ARROW):
        direction = -1
    else:
        raise AssertionError("arrow sprite must point left or right")
    return direction, {tuple(map(int, p)) for p in coords}


def _extract_pillars(grid: np.ndarray, arrow_cells: set[tuple[int, int]]) -> list[dict[str, int | bool]]:
    h, w = grid.shape
    base_row = h - 2
    ground = int(grid[h - 1, 0])
    pillars: list[dict[str, int | bool]] = []
    for c in range(w):
        if (base_row, c) in arrow_cells:
            continue
        bottom = int(grid[base_row, c])
        if bottom in (BACKGROUND, ground, ARROW_COLOR):
            continue
        r = base_row
        while r >= 0 and int(grid[r, c]) != BACKGROUND:
            r -= 1
        top = int(r + 1)
        segment = [int(v) for v in grid[top : base_row + 1, c]]
        anchor = segment[-1] == ANCHOR_COLOR
        if anchor:
            assert len(segment) >= 2, "anchor pillar must show a body above the anchor base"
            body_colors = set(segment[:-1])
            assert len(body_colors) == 1 and ANCHOR_COLOR not in body_colors
            fall_color = int(segment[0])
        else:
            assert len(set(segment)) == 1 and segment[0] not in (ANCHOR_COLOR, ARROW_COLOR, ground)
            fall_color = int(segment[0])
        assert all(int(grid[rr, c]) == BACKGROUND for rr in range(top)), "pillar must be vertically contiguous"
        pillars.append(
            {
                "c": int(c),
                "top": top,
                "height": int(len(segment)),
                "anchor": bool(anchor),
                "fall_color": fall_color,
            }
        )
    assert len(pillars) >= 2, "expected at least two pillars"
    return pillars


def _paint_fallen(out: np.ndarray, base_row: int, col: int, direction: int, length: int, color: int) -> None:
    for step in range(int(length)):
        cc = int(col + direction * step)
        if 0 <= cc < out.shape[1]:
            out[base_row, cc] = int(color)


def _chain_info(
    pillars: list[dict[str, int | bool]],
    direction: int,
) -> dict[str, int | str | list[int]]:
    ordered = sorted(pillars, key=lambda p: int(p["c"]), reverse=direction < 0)
    idx = 0
    triggered_cols: list[int] = []
    stop_reason = "end"
    while True:
        cur = ordered[idx]
        triggered_cols.append(int(cur["c"]))
        if idx == len(ordered) - 1:
            stop_reason = "end"
            break
        nxt = ordered[idx + 1]
        dist = abs(int(nxt["c"]) - int(cur["c"]))
        if int(cur["height"]) > dist:
            if bool(nxt["anchor"]):
                stop_reason = "anchor"
                break
            idx += 1
            continue
        stop_reason = "short"
        break
    return {
        "ordered_cols": [int(p["c"]) for p in ordered],
        "triggered_cols": triggered_cols,
        "triggered_count": int(len(triggered_cols)),
        "stop_reason": stop_reason,
    }


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int | str | list[int]]]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    assert h >= 5 and g.ndim == 2
    ground = int(g[h - 1, 0])
    assert np.all(g[h - 1, :] == ground), "bottom row must be the ground row"
    direction, arrow_cells = _find_arrow(g)
    pillars = _extract_pillars(g, arrow_cells)
    ordered = sorted(pillars, key=lambda p: int(p["c"]), reverse=direction < 0)
    out = g.copy()
    base_row = h - 2
    idx = 0
    stop_reason = "end"
    while True:
        cur = ordered[idx]
        cur_col = int(cur["c"])
        out[int(cur["top"]) : base_row + 1, cur_col] = BACKGROUND
        if idx == len(ordered) - 1:
            _paint_fallen(out, base_row, cur_col, direction, int(cur["height"]), int(cur["fall_color"]))
            stop_reason = "end"
            break
        nxt = ordered[idx + 1]
        dist = abs(int(nxt["c"]) - cur_col)
        if int(cur["height"]) > dist:
            _paint_fallen(out, base_row, cur_col, direction, dist, int(cur["fall_color"]))
            if bool(nxt["anchor"]):
                stop_reason = "anchor"
                break
            idx += 1
            continue
        _paint_fallen(out, base_row, cur_col, direction, int(cur["height"]), int(cur["fall_color"]))
        stop_reason = "short"
        break
    info = _chain_info(pillars, direction)
    info["stop_reason"] = stop_reason
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push the first standing pillar and propagate a left/right domino chain until height or an anchor stops it.

    Input structure:
    - The bottom row is a full ground strip, with empty background above it.
    - Several 1-cell-wide vertical pillars stand on the row just above the ground; each pillar has its own body color.
    - A single 3x3 arrow sprite at the left or right edge gives the push direction, and any anchor pillar has the special anchor color only on its bottom cell.

    Rule:
    - The first pillar in the push direction falls horizontally from its base cell for exactly its height many cells, and its original vertical column is erased.
    - If that fall would reach the next standing pillar column, the fallen bar is truncated one cell before that column and the next pillar is triggered to fall in the same direction.
    - A reached anchor pillar never falls; it stays upright and the chain stops there, while a pillar whose height is too short to reach the next one simply finishes its own full-length fall and ends the chain.
    - Any pillar never triggered by the chain stays unchanged.

    Key insight: Chain propagation depends on comparing each fallen pillar's height to the column distance of the next pillar, with the anchor color absorbing the first reachable hit.
    """
    out, _info = _simulate(np.asarray(grid, dtype=int))
    return out


def _choose_push_specs(mode: str, max_height: int) -> tuple[list[dict[str, int | bool]], list[int]]:
    n = random.randint(4, 6)
    colors = random.sample(BODY_COLORS, n)
    push_pillars: list[dict[str, int | bool]] = [
        {"height": random.randint(2, max_height), "anchor": False, "body_color": int(colors[i])}
        for i in range(n)
    ]
    push_gaps = [0] * (n - 1)

    if mode == "anchor":
        anchor_idx = random.randint(2, n - 1)
        push_pillars[anchor_idx]["anchor"] = True
        for i in range(anchor_idx):
            hi = random.randint(3, max_height)
            gap = random.randint(2, min(5, hi - 1))
            push_pillars[i]["height"] = hi
            push_gaps[i] = gap
        for i in range(anchor_idx + 1, n):
            push_pillars[i]["height"] = random.randint(2, max_height)
        for i in range(anchor_idx, n - 1):
            if push_gaps[i] == 0:
                push_gaps[i] = random.randint(2, 5)
    elif mode == "short":
        stop_idx = random.randint(1, n - 2)
        for i in range(stop_idx):
            hi = random.randint(3, max_height)
            gap = random.randint(2, min(5, hi - 1))
            push_pillars[i]["height"] = hi
            push_gaps[i] = gap
        hi = random.randint(2, min(6, max_height))
        push_pillars[stop_idx]["height"] = hi
        push_gaps[stop_idx] = random.randint(hi, 6)
        for i in range(stop_idx + 1, n):
            push_pillars[i]["height"] = random.randint(2, max_height)
        for i in range(stop_idx + 1, n - 1):
            if push_gaps[i] == 0:
                push_gaps[i] = random.randint(2, 5)
    else:
        for i in range(n - 1):
            hi = random.randint(3, max_height)
            gap = random.randint(2, min(5, hi - 1))
            push_pillars[i]["height"] = hi
            push_gaps[i] = gap
        push_pillars[-1]["height"] = random.randint(2, max_height)

    assert not bool(push_pillars[0]["anchor"]), "the directly pushed pillar must not be an anchor"
    return push_pillars, push_gaps


def _layout_columns(
    push_pillars: list[dict[str, int | bool]],
    push_gaps: list[int],
    direction: int,
    width: int,
    mode: str,
) -> list[int] | None:
    if direction == 1:
        physical_gaps = list(push_gaps)
        left_extra = 5
        right_limit = width - 2
        if mode == "end":
            right_limit = min(right_limit, width - int(push_pillars[-1]["height"]) - 1)
    else:
        physical_gaps = list(reversed(push_gaps))
        left_extra = 1
        if mode == "end":
            left_extra = max(left_extra, int(push_pillars[-1]["height"]))
        right_limit = width - 6

    span = sum(physical_gaps)
    left_min = left_extra
    left_max = right_limit - span
    if left_min > left_max:
        return None

    left = random.randint(left_min, left_max)
    cols = [left]
    for gap in physical_gaps:
        cols.append(cols[-1] + int(gap))
    return cols


def _physical_pillars(
    push_pillars: list[dict[str, int | bool]],
    push_gaps: list[int],
    direction: int,
    width: int,
    mode: str,
) -> list[dict[str, int | bool]] | None:
    cols = _layout_columns(push_pillars, push_gaps, direction, width, mode)
    if cols is None:
        return None
    if direction == 1:
        physical_specs = push_pillars
    else:
        physical_specs = list(reversed(push_pillars))
    physical_pillars: list[dict[str, int | bool]] = []
    for c, spec in zip(cols, physical_specs):
        physical_pillars.append(
            {
                "c": int(c),
                "height": int(spec["height"]),
                "anchor": bool(spec["anchor"]),
                "body_color": int(spec["body_color"]),
            }
        )
    return physical_pillars


def _render_input(
    height: int,
    width: int,
    direction: int,
    physical_pillars: list[dict[str, int | bool]],
) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[-1, :] = GROUND_COLOR
    arrow_top = height - 4
    arrow_left = 0 if direction == 1 else width - 3
    arrow_shape = RIGHT_ARROW if direction == 1 else LEFT_ARROW
    for rr in range(3):
        for cc in range(3):
            if bool(arrow_shape[rr, cc]):
                grid[arrow_top + rr, arrow_left + cc] = ARROW_COLOR

    base_row = height - 2
    for pillar in physical_pillars:
        c = int(pillar["c"])
        h = int(pillar["height"])
        top = base_row - h + 1
        body_color = int(pillar["body_color"])
        grid[top : base_row + 1, c] = body_color
        if bool(pillar["anchor"]):
            grid[base_row, c] = ANCHOR_COLOR
    return grid


def _expected_from_specs(
    height: int,
    width: int,
    direction: int,
    push_pillars: list[dict[str, int | bool]],
    push_gaps: list[int],
    physical_pillars: list[dict[str, int | bool]],
) -> tuple[np.ndarray, str, list[int]]:
    out = _render_input(height, width, direction, physical_pillars)
    base_row = height - 2
    stop_reason = "end"
    triggered_cols: list[int] = []
    cols_in_push_order = [int(p["c"]) for p in physical_pillars]
    if direction == -1:
        cols_in_push_order = list(reversed(cols_in_push_order))

    idx = 0
    while True:
        col = cols_in_push_order[idx]
        spec = push_pillars[idx]
        top = base_row - int(spec["height"]) + 1
        out[top : base_row + 1, col] = BACKGROUND
        triggered_cols.append(int(col))
        if idx == len(push_pillars) - 1:
            _paint_fallen(out, base_row, col, direction, int(spec["height"]), int(spec["body_color"]))
            stop_reason = "end"
            break
        dist = int(push_gaps[idx])
        if int(spec["height"]) > dist:
            _paint_fallen(out, base_row, col, direction, dist, int(spec["body_color"]))
            if bool(push_pillars[idx + 1]["anchor"]):
                stop_reason = "anchor"
                break
            idx += 1
            continue
        _paint_fallen(out, base_row, col, direction, int(spec["height"]), int(spec["body_color"]))
        stop_reason = "short"
        break
    return out, stop_reason, triggered_cols


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    mode: str,
    physical_pillars: list[dict[str, int | bool]],
    stop_reason: str,
    triggered_cols: list[int],
) -> bool:
    if np.array_equal(inp, expected):
        return False
    parsed_direction, arrow_cells = _find_arrow(inp)
    parsed_pillars = _extract_pillars(inp, arrow_cells)
    parsed_sorted = sorted(parsed_pillars, key=lambda p: int(p["c"]))
    expected_sorted = sorted(physical_pillars, key=lambda p: int(p["c"]))
    if len(parsed_sorted) != len(expected_sorted):
        return False
    for got, want in zip(parsed_sorted, expected_sorted):
        if (
            int(got["c"]) != int(want["c"])
            or int(got["height"]) != int(want["height"])
            or bool(got["anchor"]) != bool(want["anchor"])
            or int(got["fall_color"]) != int(want["body_color"])
        ):
            return False

    _out, info = _simulate(inp)
    if not np.array_equal(_out, expected):
        return False
    if str(info["stop_reason"]) != stop_reason or str(info["stop_reason"]) != mode:
        return False
    if list(info["triggered_cols"]) != triggered_cols:
        return False
    if int(info["triggered_count"]) < 2:
        return False

    base_row = inp.shape[0] - 2
    if mode == "anchor":
        anchor_cols = [int(p["c"]) for p in physical_pillars if bool(p["anchor"])]
        if len(anchor_cols) != 1:
            return False
        anchor_col = anchor_cols[0]
        if int(expected[base_row, anchor_col]) != ANCHOR_COLOR:
            return False
        if not np.array_equal(inp[:, anchor_col], expected[:, anchor_col]):
            return False
    if mode == "short":
        next_cols = [int(p["c"]) for p in physical_pillars if int(p["c"]) not in triggered_cols]
        if not next_cols:
            return False
    if mode == "end":
        if len(triggered_cols) != len(physical_pillars):
            return False
    return True


def _gen_attempt(mode: str) -> np.ndarray | None:
    height = random.randint(10, 16)
    width = random.randint(18, 30)
    max_height = min(7, height - 2)
    if max_height < 4:
        return None
    direction = random.choice((-1, 1))
    push_pillars, push_gaps = _choose_push_specs(mode, max_height)
    physical_pillars = _physical_pillars(push_pillars, push_gaps, direction, width, mode)
    if physical_pillars is None:
        return None
    inp = _render_input(height, width, direction, physical_pillars)
    expected, stop_reason, triggered_cols = _expected_from_specs(
        height,
        width,
        direction,
        push_pillars,
        push_gaps,
        physical_pillars,
    )
    if stop_reason != mode:
        return None
    if not _validate_semantics(inp, expected, mode, physical_pillars, stop_reason, triggered_cols):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _generate_counter
    preferred_mode = MODE_CYCLE[_generate_counter % len(MODE_CYCLE)]
    _generate_counter += 1
    modes = (preferred_mode,) + tuple(m for m in MODE_CYCLE if m != preferred_mode)
    for mode in modes:
        for _ in range(MAX_ATTEMPTS):
            grid = _gen_attempt(mode)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid domino-chain input")
