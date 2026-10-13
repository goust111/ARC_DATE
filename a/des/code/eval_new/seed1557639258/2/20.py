import random

import numpy as np


WIRE = 5
SIGNAL_COLORS = (1, 2, 3)
BACKGROUND_CHOICES = (0, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    counts[WIRE] = -1
    return int(np.argmax(counts))


def _lane_row(idx: int) -> int:
    return 2 + 2 * idx


def _selector_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 2, c - 1),
        (r - 2, c),
        (r - 1, c),
        (r, c - 1),
        (r, c),
        (r, c + 1),
        (r + 1, c),
        (r + 2, c - 1),
        (r + 2, c),
    ]


def _mix_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 2, c),
        (r - 1, c),
        (r, c - 1),
        (r, c),
        (r, c + 1),
        (r + 1, c),
        (r + 2, c),
    ]


def _is_gray(g: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(int(g[r, c]) == WIRE for r, c in cells)


def _is_not_gray(g: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(int(g[r, c]) != WIRE for r, c in cells)


def _detect_gates(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    gates = []
    occupied: set[tuple[int, int]] = set()

    for c in range(2, w - 2):
        for r in range(4, h - 4, 2):
            selector = _selector_cells(r, c)
            if _is_gray(g, selector) and _is_not_gray(
                g,
                [
                    (r - 2, c + 1),
                    (r - 1, c - 1),
                    (r - 1, c + 1),
                    (r + 1, c - 1),
                    (r + 1, c + 1),
                    (r + 2, c + 1),
                ],
            ):
                cells = set(selector)
                assert not (cells & occupied), "gate sprites must not overlap"
                occupied |= cells
                gates.append({"kind": "selector", "row": r, "col": c})
                continue

            mix = _mix_cells(r, c)
            if not _is_gray(g, mix):
                continue
            if not _is_not_gray(g, [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]):
                continue

            up_in = int(g[r - 2, c - 1]) == WIRE and int(g[r + 2, c - 1]) != WIRE
            down_in = int(g[r + 2, c - 1]) == WIRE and int(g[r - 2, c - 1]) != WIRE
            if not (up_in ^ down_in):
                continue

            cells = set(mix)
            assert not (cells & occupied), "gate sprites must not overlap"
            occupied |= cells
            gates.append({"kind": "mix_up" if up_in else "mix_down", "row": r, "col": c})

    gates.sort(key=lambda item: item["col"])
    return gates


def _paint_segment(out: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    if c0 > c1:
        return
    for c in range(c0, c1 + 1):
        assert int(out[row, c]) == WIRE, "wire segment must stay on gray cells"
        out[row, c] = int(color)


def _paint_selector_gate(out: np.ndarray, r: int, c: int, top: int, mid: int, bot: int, chosen: int) -> None:
    out[r - 2, c - 1] = int(top)
    out[r, c - 1] = int(mid)
    out[r + 2, c - 1] = int(bot)
    for rr, cc in [(r - 2, c), (r - 1, c), (r, c), (r, c + 1), (r + 1, c), (r + 2, c)]:
        out[rr, cc] = int(chosen)


def _paint_mix_gate(out: np.ndarray, r: int, c: int, horiz: int, vert: int) -> None:
    out[r, c - 1] = int(horiz)
    out[r, c + 1] = int(vert)
    for rr in range(r - 2, r + 3):
        out[rr, c] = int(vert)


def _apply_selector(active: dict[int, int], starts: dict[int, int], out: np.ndarray, r: int, c: int) -> None:
    top_r, mid_r, bot_r = r - 2, r, r + 2
    top = int(active[top_r])
    mid = int(active[mid_r])
    bot = int(active[bot_r])

    _paint_segment(out, top_r, starts[top_r], c - 2, top)
    _paint_segment(out, mid_r, starts[mid_r], c - 2, mid)
    _paint_segment(out, bot_r, starts[bot_r], c - 2, bot)

    assert top != bot, "selector candidates must differ"
    if mid == top:
        chosen = top
    else:
        assert mid == bot, "selector control must match exactly one candidate"
        chosen = bot

    _paint_selector_gate(out, r, c, top, mid, bot, chosen)
    del active[top_r], active[bot_r], starts[top_r], starts[bot_r]
    active[mid_r] = int(chosen)
    starts[mid_r] = c + 2


def _apply_mix(active: dict[int, int], starts: dict[int, int], out: np.ndarray, r: int, c: int, source_r: int) -> None:
    horiz = int(active[r])
    vert = int(active[source_r])

    _paint_segment(out, r, starts[r], c - 2, horiz)
    _paint_segment(out, source_r, starts[source_r], c - 1, vert)
    _paint_mix_gate(out, r, c, horiz, vert)

    del active[source_r], starts[source_r]
    active[r] = int(vert)
    starts[r] = c + 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate left-edge color signals through a grey rail circuit and repaint the full wiring with the resulting outputs.

    Input structure:
    - The background is the most common non-grey color. Grey (5) pixels form a left-to-right wire network drawn on evenly spaced horizontal rail rows.
    - Every rail starts with a colored input pixel on the left edge, and some rails survive to grey output endpoints on the right edge.
    - Two gate sprites can appear inside the grey network: a tall T-like selector gate spanning three rails, and a cross-like mix gate spanning two neighboring rails.

    Rule:
    - Signals move from left to right along each rail, recoloring every grey wire pixel they traverse.
    - In a selector gate, the middle control rail chooses between the top and bottom candidate rails by color match: if the control color equals the top candidate, the gate forwards the top color; otherwise it must equal the bottom candidate and forwards the bottom color.
    - In a mix gate, the vertical feeder rail overwrites the horizontal rail at the crossing and continues to the right, while the horizontal signal stops at the gate's left arm.
    - After all gates are resolved in left-to-right order, every remaining right-edge output rail and every gate body is recolored by the signal that reaches it.

    Key insight:
    - The circuit is solved by tracing which rails remain active after each gate: selector gates compare visible colors across three aligned inputs, while mix gates are directional because only the rail physically connected through the vertical spine survives to the right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _bg_color(g)

    lane_rows = [int(r) for r in np.flatnonzero((g[:, 0] != bg) & (g[:, 0] != WIRE))]
    assert lane_rows, "expected left-edge colored inputs"
    assert all(b - a == 2 for a, b in zip(lane_rows, lane_rows[1:])), "input rails must be evenly spaced"

    gates = _detect_gates(g)
    out = g.copy()
    active = {row: int(g[row, 0]) for row in lane_rows}
    starts = {row: 1 for row in lane_rows}

    for gate in gates:
        r = int(gate["row"])
        c = int(gate["col"])
        if gate["kind"] == "selector":
            assert r - 2 in active and r in active and r + 2 in active, "selector inputs must all be active"
            _apply_selector(active, starts, out, r, c)
        elif gate["kind"] == "mix_up":
            assert r - 2 in active and r in active and r + 2 not in active, "upward mix must use the upper feeder rail"
            _apply_mix(active, starts, out, r, c, r - 2)
        else:
            assert gate["kind"] == "mix_down"
            assert r + 2 in active and r in active and r - 2 not in active, "downward mix must use the lower feeder rail"
            _apply_mix(active, starts, out, r, c, r + 2)

    for row, color in list(active.items()):
        _paint_segment(out, row, starts[row], w - 1, color)

    assert not np.any(out == WIRE), "all grey wire pixels should be replaced by propagated colors"
    return out


def _enumerate_ops(active: tuple[bool, ...], colors: tuple[int, ...]) -> list[dict]:
    ops = []
    n = len(active)
    for i in range(1, n - 1):
        if active[i - 1] and active[i] and active[i + 1] and colors[i - 1] != colors[i + 1]:
            if colors[i] == colors[i - 1]:
                ops.append({"kind": "select_top", "family": "selector", "center": i})
            if colors[i] == colors[i + 1]:
                ops.append({"kind": "select_bottom", "family": "selector", "center": i})

        if active[i - 1] and active[i] and not active[i + 1]:
            ops.append({"kind": "mix_up", "family": "mix", "center": i})
        if not active[i - 1] and active[i] and active[i + 1]:
            ops.append({"kind": "mix_down", "family": "mix", "center": i})
    return ops


def _apply_op(active: tuple[bool, ...], colors: tuple[int, ...], op: dict) -> tuple[tuple[bool, ...], tuple[int, ...]]:
    i = int(op["center"])
    active_list = list(active)
    color_list = list(colors)
    if op["kind"] == "select_top":
        active_list[i - 1] = False
        active_list[i + 1] = False
        color_list[i] = color_list[i - 1]
    elif op["kind"] == "select_bottom":
        active_list[i - 1] = False
        active_list[i + 1] = False
        color_list[i] = color_list[i + 1]
    elif op["kind"] == "mix_up":
        active_list[i - 1] = False
        color_list[i] = color_list[i - 1]
    else:
        assert op["kind"] == "mix_down"
        active_list[i + 1] = False
        color_list[i] = color_list[i + 1]
    return tuple(active_list), tuple(int(c) for c in color_list)


def _search_plan(colors: tuple[int, ...], target_len: int) -> list[dict] | None:
    active0 = tuple(True for _ in colors)
    need = {"selector", "mix"}

    def rec(active: tuple[bool, ...], cur_colors: tuple[int, ...], depth: int, used: frozenset[str]) -> list[dict] | None:
        if depth == target_len:
            return [] if need.issubset(used) and 1 <= sum(active) <= 3 else None

        remaining = target_len - depth
        missing = need - set(used)
        if len(missing) > remaining:
            return None

        ops = _enumerate_ops(active, cur_colors)
        if not ops:
            return None

        random.shuffle(ops)
        ops.sort(key=lambda op: (op["family"] not in missing, random.random()))
        for op in ops:
            next_active, next_colors = _apply_op(active, cur_colors, op)
            tail = rec(next_active, next_colors, depth + 1, frozenset(set(used) | {op["family"]}))
            if tail is not None:
                return [op] + tail
        return None

    return rec(active0, colors, 0, frozenset())


def _draw_segment(grid: np.ndarray, row: int, c0: int, c1: int) -> None:
    if c0 <= c1:
        grid[row, c0 : c1 + 1] = WIRE


def _draw_gate(grid: np.ndarray, op: dict, col: int) -> None:
    r = _lane_row(int(op["center"]))
    cells = _selector_cells(r, col) if op["family"] == "selector" else _mix_cells(r, col)
    for rr, cc in cells:
        grid[rr, cc] = WIRE


def _render_input(initial_colors: tuple[int, ...], plan: list[dict], bg: int, start_col: int, gap: int, tail: int) -> np.ndarray:
    lane_count = len(initial_colors)
    h = 2 * lane_count + 3
    cols = [start_col + gap * i for i in range(len(plan))]
    w = (cols[-1] + tail + 2) if cols else 8
    grid = np.full((h, w), bg, dtype=int)

    for i, color in enumerate(initial_colors):
        grid[_lane_row(i), 0] = int(color)

    active = {i: int(color) for i, color in enumerate(initial_colors)}
    starts = {i: 1 for i in range(lane_count)}

    for op, col in zip(plan, cols):
        center = int(op["center"])
        if op["family"] == "selector":
            for lane in (center - 1, center, center + 1):
                _draw_segment(grid, _lane_row(lane), starts[lane], col - 2)
            _draw_gate(grid, op, col)
            del active[center - 1], active[center + 1], starts[center - 1], starts[center + 1]
            starts[center] = col + 2
        elif op["kind"] == "mix_up":
            _draw_segment(grid, _lane_row(center), starts[center], col - 2)
            _draw_segment(grid, _lane_row(center - 1), starts[center - 1], col - 1)
            _draw_gate(grid, op, col)
            del active[center - 1], starts[center - 1]
            starts[center] = col + 2
        else:
            assert op["kind"] == "mix_down"
            _draw_segment(grid, _lane_row(center), starts[center], col - 2)
            _draw_segment(grid, _lane_row(center + 1), starts[center + 1], col - 1)
            _draw_gate(grid, op, col)
            del active[center + 1], starts[center + 1]
            starts[center] = col + 2

    for lane in active:
        _draw_segment(grid, _lane_row(lane), starts[lane], w - 1)

    return grid


def _sample_colors(lane_count: int) -> tuple[int, ...]:
    while True:
        colors = tuple(random.choice(SIGNAL_COLORS) for _ in range(lane_count))
        if len(set(colors)) >= 2:
            return colors


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        lane_count = random.randint(5, 8)
        colors = _sample_colors(lane_count)

        possible_lengths = [length for length in range(2, 5) if length < lane_count]
        random.shuffle(possible_lengths)

        plan = None
        for target_len in possible_lengths:
            plan = _search_plan(colors, target_len)
            if plan is not None:
                break
        if plan is None:
            continue

        bg = random.choice(BACKGROUND_CHOICES)
        start_col = random.randint(4, 5)
        gap = random.randint(5, 6)
        tail = random.randint(3, 5)
        cols = [start_col + gap * i for i in range(len(plan))]
        w = cols[-1] + tail + 2
        h = 2 * lane_count + 3
        if max(h, w) > 30:
            continue

        inp = _render_input(colors, plan, bg, start_col, gap, tail)
        try:
            out = main(inp)
        except AssertionError:
            continue

        if inp.shape == out.shape and not np.any(out == WIRE):
            return inp

    raise RuntimeError("failed to generate a valid circuit input")
