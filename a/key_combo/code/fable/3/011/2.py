from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _infer_structure(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3 and w >= 5

    wall = int(g[0, 0])
    assert np.all(g[:, 0] == wall) and np.all(g[:, -1] == wall)

    bg = _get_bg(g[:, 1:-1])
    tokens = [
        (int(c), int(g[0, c]))
        for c in range(1, w - 1)
        if int(g[0, c]) not in {bg, wall}
    ]
    assert tokens

    gates = np.full((h, w), -1, dtype=int)
    for r in range(1, h):
        for c in range(1, w - 1):
            value = int(g[r, c])
            if value not in {bg, wall}:
                gates[r, c] = value
    return bg, wall, tokens, gates


def _segment_bounds(gates: np.ndarray, row: int, col: int) -> tuple[int, int]:
    color = int(gates[row, col])
    left = int(col)
    while left - 1 >= 1 and int(gates[row, left - 1]) == color:
        left -= 1
    right = int(col)
    while right + 1 < gates.shape[1] - 1 and int(gates[row, right + 1]) == color:
        right += 1
    return left, right


def _compose_output(
    shape: tuple[int, int],
    bg: int,
    wall: int,
    gates: np.ndarray,
    settled: np.ndarray,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    out[:, 0] = int(wall)
    out[:, -1] = int(wall)
    gate_mask = gates >= 0
    out[gate_mask] = gates[gate_mask]
    token_mask = settled >= 0
    out[token_mask] = settled[token_mask]
    return out


def _simulate_tokens(
    shape: tuple[int, int],
    bg: int,
    wall: int,
    tokens: list[tuple[int, int]],
    gates: np.ndarray,
    *,
    descending: bool = False,
) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray]:
    h, w = shape
    current_gates = np.array(gates, copy=True)
    settled = np.full(shape, -1, dtype=int)
    trace: list[dict] = []

    ordered = sorted(tokens, key=lambda item: item[0], reverse=descending)
    for start_col, color in ordered:
        row = 0
        opened: list[tuple[int, int, int, int]] = []

        while True:
            next_row = row + 1
            if next_row >= h:
                stop_reason = "bottom"
                break
            if int(settled[next_row, start_col]) >= 0:
                stop_reason = "token"
                break

            gate_color = int(current_gates[next_row, start_col])
            if gate_color >= 0:
                if gate_color == int(color):
                    left, right = _segment_bounds(current_gates, next_row, start_col)
                    current_gates[next_row, left : right + 1] = -1
                    opened.append((int(next_row), int(left), int(right), int(color)))
                    continue
                stop_reason = "gate"
                break

            row = next_row

        settled[row, start_col] = int(color)
        trace.append(
            {
                "start_col": int(start_col),
                "color": int(color),
                "final_row": int(row),
                "dissolved": int(len(opened)),
                "opened": opened,
                "stopped_by": stop_reason,
            }
        )

    out = _compose_output(shape, bg, wall, current_gates, settled)
    return out, trace, current_gates, settled


def _simulate(
    grid: np.ndarray,
    *,
    descending: bool = False,
) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray]:
    bg, wall, tokens, gates = _infer_structure(grid)
    return _simulate_tokens(grid.shape, bg, wall, tokens, gates, descending=descending)


def _independent_final_rows(grid: np.ndarray) -> dict[int, int]:
    bg, wall, tokens, gates = _infer_structure(grid)
    rows: dict[int, int] = {}
    for start_col, color in sorted(tokens, key=lambda item: item[0]):
        _, trace, _, _ = _simulate_tokens(
            grid.shape,
            bg,
            wall,
            [(int(start_col), int(color))],
            gates,
        )
        rows[int(start_col)] = int(trace[0]["final_row"])
    return rows


def _count_gate_segments(gates: np.ndarray) -> int:
    count = 0
    h, w = gates.shape
    for r in range(1, h):
        c = 1
        while c < w - 1:
            if int(gates[r, c]) < 0:
                c += 1
                continue
            count += 1
            color = int(gates[r, c])
            while c + 1 < w - 1 and int(gates[r, c + 1]) == color:
                c += 1
            c += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the top-row tokens left-to-right, erasing matching horizontal gates on contact.

    Input structure:
    - The grid is a vertical chamber whose left and right border columns are a solid wall color.
    - Interior cells are mostly background, with several single-pixel colored tokens on the top row.
    - Lower rows contain disjoint one-row horizontal gate bars, each monochrome and length at least three.

    Rule:
    - Process tokens strictly by increasing start column.
    - Each token falls straight down in its own column.
    - When the cell below is a same-color gate, erase that whole gate bar to background and keep falling.
    - When the cell below is a different-color gate, a settled token, or the bottom boundary, the token stops one cell above it.
    - Keep the walls and every undissolved gate; output the final chamber after all tokens have settled.

    Key insight:
    - Gates are shared terrain across columns, so an earlier token can delete a wide blocker and open a deeper route for later tokens.
    """
    out, _trace, _gates, _settled = _simulate(grid)
    return out


def _paint_gate(grid: np.ndarray, row: int, left: int, right: int, color: int) -> None:
    grid[int(row), int(left) : int(right) + 1] = int(color)


def _random_palette() -> tuple[int, int, int, int, int, int]:
    bg, wall, a, b, c, d = random.sample(list(range(10)), 6)
    return int(bg), int(wall), int(a), int(b), int(c), int(d)


def _build_candidate() -> tuple[np.ndarray, dict] | None:
    bg, wall, color_a, color_b, color_c, color_d = _random_palette()

    x1 = random.randint(6, 8)
    x2 = x1 + random.randint(2, 3)
    x3 = x2 + random.randint(2, 3)
    x4 = x3 + random.randint(2, 3)
    width = x4 + random.randint(7, 10) + 2

    r1 = random.randint(2, 4)
    r2 = r1 + random.randint(2, 4)
    r3 = r2 + random.randint(2, 4)
    r4 = r3 + random.randint(2, 4)
    height = r4 + random.randint(5, 8) + 1

    if width > 30 or height > 30:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    grid[:, 0] = int(wall)
    grid[:, -1] = int(wall)

    core_tokens = [
        (int(x1), int(color_a)),
        (int(x2), int(color_b)),
        (int(x3), int(color_a)),
        (int(x4), int(color_c)),
    ]
    for col, color in core_tokens:
        grid[0, col] = int(color)

    g1_left = x1 - random.randint(1, 2)
    g1_right = x4 + random.randint(1, 2)
    g2_left = x2 - random.randint(1, 2)
    g2_right = x4 + random.randint(1, 2)
    g3_left = x3 - random.randint(1, 2)
    g3_right = x4 + random.randint(1, 2)
    g4_left = x3
    g4_right = x4 + 1

    _paint_gate(grid, r1, g1_left, g1_right, color_b)
    _paint_gate(grid, r2, g2_left, g2_right, color_a)
    _paint_gate(grid, r3, g3_left, g3_right, color_a)
    _paint_gate(grid, r4, g4_left, g4_right, color_c)

    right_lo = x4 + 2
    right_hi = width - 3
    if right_hi - right_lo + 1 < 3:
        return None
    survivor_len = random.randint(3, min(5, right_hi - right_lo + 1))
    survivor_left = random.randint(right_lo, right_hi - survivor_len + 1)
    survivor_right = survivor_left + survivor_len - 1
    survivor_row = random.randint(r4 + 2, height - 2)
    _paint_gate(grid, survivor_row, survivor_left, survivor_right, color_d)

    used_rows = {r1, r2, r3, r4, survivor_row}
    extra_kind = random.choice(["none", "free", "match", "block"])
    extra_meta: dict[str, int | str] = {"kind": extra_kind}

    left_lo = 2
    left_hi = x1 - 2
    left_width = left_hi - left_lo + 1
    extra_rows = [r for r in range(2, height - 1) if all(abs(r - u) > 1 for u in used_rows)]
    if left_width >= 3 and extra_kind != "none":
        extra_col = random.randint(left_lo, left_hi)
        extra_color = random.choice([color_a, color_b, color_c, color_d])
        grid[0, extra_col] = int(extra_color)
        extra_meta["col"] = int(extra_col)
        extra_meta["color"] = int(extra_color)

        if extra_kind in {"match", "block"} and extra_rows:
            extra_row = random.choice(extra_rows)
            gate_len = random.randint(3, left_width)
            start_min = max(left_lo, extra_col - gate_len + 1)
            start_max = min(extra_col, left_hi - gate_len + 1)
            if start_min > start_max:
                return None
            extra_left = random.randint(start_min, start_max)
            extra_right = extra_left + gate_len - 1
            if extra_kind == "match":
                extra_gate_color = int(extra_color)
            else:
                candidates = [color_a, color_b, color_c, color_d]
                extra_gate_color = int(random.choice([c for c in candidates if c != extra_color]))
            _paint_gate(grid, extra_row, extra_left, extra_right, extra_gate_color)
            extra_meta["row"] = int(extra_row)
            extra_meta["gate_color"] = int(extra_gate_color)
        elif extra_kind in {"match", "block"}:
            return None

    meta = {
        "core_cols": (int(x1), int(x2), int(x3), int(x4)),
        "core_rows": (int(r1), int(r2), int(r3), int(r4)),
        "survivor_row": int(survivor_row),
        "extra": extra_meta,
    }
    return grid, meta


def _validate_candidate(grid: np.ndarray, meta: dict) -> bool:
    out, trace, gates_final, _settled = _simulate(grid)
    reverse_out, _reverse_trace, _reverse_gates, _reverse_settled = _simulate(grid, descending=True)
    independent_rows = _independent_final_rows(grid)
    trace_by_col = {int(item["start_col"]): item for item in trace}

    x1, x2, x3, x4 = meta["core_cols"]
    r1, r2, _r3, r4 = meta["core_rows"]
    h = int(grid.shape[0])

    if np.array_equal(out, reverse_out):
        return False
    if _count_gate_segments(gates_final) < 1:
        return False
    if int(trace_by_col[x1]["final_row"]) != r1 - 1:
        return False
    if int(trace_by_col[x1]["dissolved"]) != 0 or trace_by_col[x1]["stopped_by"] != "gate":
        return False
    if int(trace_by_col[x2]["final_row"]) != r2 - 1:
        return False
    if int(trace_by_col[x2]["dissolved"]) != 1 or trace_by_col[x2]["stopped_by"] != "gate":
        return False
    if int(trace_by_col[x3]["final_row"]) != r4 - 1:
        return False
    if int(trace_by_col[x3]["dissolved"]) < 2 or trace_by_col[x3]["stopped_by"] != "gate":
        return False
    if int(trace_by_col[x4]["final_row"]) != h - 1:
        return False
    if int(trace_by_col[x4]["dissolved"]) < 1 or trace_by_col[x4]["stopped_by"] != "bottom":
        return False
    if int(trace_by_col[x3]["final_row"]) <= independent_rows[x3]:
        return False
    if int(trace_by_col[x4]["final_row"]) <= independent_rows[x4]:
        return False
    if not np.array_equal(main(grid), out):
        return False

    extra = meta["extra"]
    extra_col = extra.get("col")
    if extra_col is not None:
        extra_trace = trace_by_col[int(extra_col)]
        kind = str(extra["kind"])
        if kind == "free":
            if int(extra_trace["dissolved"]) != 0 or int(extra_trace["final_row"]) != h - 1:
                return False
        elif kind == "match":
            if int(extra_trace["dissolved"]) < 1 or int(extra_trace["final_row"]) != h - 1:
                return False
        elif kind == "block":
            if int(extra_trace["final_row"]) != int(extra["row"]) - 1:
                return False
            if extra_trace["stopped_by"] != "gate":
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_candidate()
        if built is None:
            continue
        grid, meta = built
        if _validate_candidate(grid, meta):
            return grid
    raise RuntimeError("failed to generate a non-degenerate gate-ablation chamber")
