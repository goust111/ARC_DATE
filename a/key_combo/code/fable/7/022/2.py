from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    loader = SourceFileLoader("_arc_utils_fable_7_022_2", str(path))
    return loader.load_module()


_arc_utils = _load_arc_utils()
Color = _arc_utils.Color
find_components = _arc_utils.find_components
get_bg = _arc_utils.get_bg

BG = Color.BLACK
GREY = Color.GREY
COLOR_POOL = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
GEN_ATTEMPTS = 400


def _scan_platforms(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    platforms: list[dict] = []
    for r in range(h):
        c = 0
        while c < w:
            color = int(grid[r, c])
            if color == bg:
                c += 1
                continue
            c0 = c
            while c + 1 < w and int(grid[r, c + 1]) == color:
                c += 1
            c1 = c
            if c1 - c0 + 1 >= 2:
                mask[r, c0 : c1 + 1] = True
                platforms.append({"row": int(r), "c0": int(c0), "c1": int(c1), "color": color})
            c += 1
    return platforms, mask


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))
    platforms, platform_mask = _scan_platforms(g, bg)

    leftovers = np.full_like(g, bg)
    token_mask = (g != bg) & ~platform_mask
    leftovers[token_mask] = g[token_mask]
    token_components = find_components(leftovers, bg=bg, connectivity=4, monochromatic=True)

    tokens: list[dict] = []
    for comp in token_components:
        assert int(comp["area"]) == 1, "all non-platform cells must be singleton tokens"
        r, c = map(int, comp["coords"][0])
        tokens.append({"row": r, "col": c, "color": int(comp["color"])})

    tokens.sort(key=lambda token: (token["col"], token["row"]))
    assert len({token["col"] for token in tokens}) == len(tokens), "expected at most one token per column"
    return bg, tokens, platforms, platform_mask


def _erase_platform_run(grid: np.ndarray, row: int, col: int, bg: int) -> dict:
    color = int(grid[row, col])
    c0 = int(col)
    c1 = int(col)
    while c0 - 1 >= 0 and int(grid[row, c0 - 1]) == color:
        c0 -= 1
    while c1 + 1 < grid.shape[1] and int(grid[row, c1 + 1]) == color:
        c1 += 1
    assert c1 - c0 + 1 >= 2, "a platform must be a horizontal run of length >= 2"
    grid[row, c0 : c1 + 1] = bg
    return {"row": int(row), "c0": int(c0), "c1": int(c1), "color": color}


def _next_support_row(grid: np.ndarray, row: int, col: int, bg: int) -> int | None:
    for rr in range(int(row) + 1, grid.shape[0]):
        if int(grid[rr, col]) != bg:
            return int(rr)
    return None


def _simulate_grid(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    bg, tokens, _platforms, platform_mask = _parse_scene(grid)
    working = np.asarray(grid, dtype=int).copy()
    working[(working != bg) & ~platform_mask] = bg

    finals: list[dict] = []
    traces: list[dict] = []
    consumed_runs: list[dict] = []
    for token in tokens:
        cur_row = int(token["row"])
        cur_col = int(token["col"])
        cur_color = int(token["color"])
        hit_colors: list[int] = []

        while True:
            support_row = _next_support_row(working, cur_row, cur_col, bg)
            if support_row is None:
                cur_row = working.shape[0] - 1
                stop_kind = "bottom"
                break

            cur_row = int(support_row - 1)
            support_color = int(working[support_row, cur_col])
            if support_color == GREY:
                stop_kind = "grey"
                break

            hit_colors.append(support_color)
            cur_color = support_color
            consumed_runs.append(_erase_platform_run(working, support_row, cur_col, bg))

        finals.append({"row": cur_row, "col": cur_col, "color": cur_color})
        traces.append(
            {
                "start_color": int(token["color"]),
                "final_color": cur_color,
                "final_row": cur_row,
                "col": cur_col,
                "hit_colors": tuple(hit_colors),
                "stop_kind": stop_kind,
            }
        )

    out = working.copy()
    for token in finals:
        out[int(token["row"]), int(token["col"])] = int(token["color"])

    return out, {"tokens": tokens, "finals": finals, "traces": traces, "consumed_runs": consumed_runs}


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Simulate left-to-right falling tokens that consume horizontal color gates and keep the last absorbed color.

    Input structure:
    - A black background contains several singleton tokens in the upper area, with at most one token per column.
    - The middle and lower area contains horizontal one-row platforms of length at least two.
    - Grey platforms are permanent stoppers; non-grey platforms are disposable recoloring gates.

    Rule:
    - Process tokens strictly from left to right by column.
    - Each token falls vertically; on a non-grey platform it lands, adopts that platform color, erases the whole run, and continues falling.
    - A token stops on top of a grey platform or on the grid bottom; consumed colored platforms disappear, untouched platforms remain, and every token is drawn at its final row with its final color.

    Key insight:
    - Colored platforms are one-use gates, so earlier tokens delete supports that later tokens would otherwise touch, making both depth and final color order-dependent.
    """
    out, _trace = _simulate_grid(np.asarray(input_grid, dtype=int))
    return out


def _simulate_spec(height: int, width: int, tokens: list[dict], platforms: list[dict], *, reverse: bool = False) -> tuple[np.ndarray, dict]:
    active = [True] * len(platforms)
    order = sorted(range(len(tokens)), key=lambda i: (tokens[i]["col"], tokens[i]["row"]), reverse=reverse)
    finals: list[dict | None] = [None] * len(tokens)
    traces: list[dict | None] = [None] * len(tokens)
    consumed: list[int] = []

    for idx in order:
        token = tokens[idx]
        cur_row = int(token["row"])
        cur_col = int(token["col"])
        cur_color = int(token["color"])
        hit_colors: list[int] = []

        while True:
            candidates = [
                (int(platform["row"]), platform_index)
                for platform_index, platform in enumerate(platforms)
                if active[platform_index]
                and int(platform["row"]) > cur_row
                and int(platform["c0"]) <= cur_col <= int(platform["c1"])
            ]
            if not candidates:
                cur_row = height - 1
                stop_kind = "bottom"
                break

            support_row, platform_index = min(candidates)
            platform = platforms[platform_index]
            cur_row = support_row - 1
            if int(platform["color"]) == GREY:
                stop_kind = "grey"
                break

            cur_color = int(platform["color"])
            hit_colors.append(cur_color)
            active[platform_index] = False
            consumed.append(platform_index)

        finals[idx] = {"row": cur_row, "col": cur_col, "color": cur_color}
        traces[idx] = {
            "start_color": int(token["color"]),
            "final_color": cur_color,
            "final_row": cur_row,
            "col": cur_col,
            "hit_colors": tuple(hit_colors),
            "stop_kind": stop_kind,
        }

    out = np.zeros((height, width), dtype=int)
    for platform_index, platform in enumerate(platforms):
        if active[platform_index]:
            out[int(platform["row"]), int(platform["c0"]) : int(platform["c1"]) + 1] = int(platform["color"])
    for token in finals:
        assert token is not None
        out[int(token["row"]), int(token["col"])] = int(token["color"])

    return out, {"finals": finals, "traces": traces, "consumed": consumed, "active": active}


def _sample_token_cols(width: int, count: int, min_gap: int = 4) -> list[int] | None:
    choices = list(range(1, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(choices, count))
        if all(cols[i + 1] - cols[i] >= min_gap for i in range(len(cols) - 1)):
            return cols
    return None


def _pick_row(used_rows: set[int], lo: int, hi: int) -> int | None:
    candidates = [row for row in range(max(4, lo), hi + 1) if row not in used_rows]
    if not candidates:
        return None
    return int(random.choice(candidates))


def _pick_color(*, exclude: set[int] | None = None) -> int:
    blocked = set() if exclude is None else {int(color) for color in exclude}
    candidates = [int(color) for color in COLOR_POOL if int(color) not in blocked]
    return int(random.choice(candidates))


def _single_column_span(col: int, token_cols: list[int], width: int) -> tuple[int, int] | None:
    others = set(token_cols) - {int(col)}
    candidates: list[tuple[int, int]] = []
    for length in range(2, 5):
        for c0 in range(col - length + 1, col + 1):
            c1 = c0 + length - 1
            if c0 < 0 or c1 >= width:
                continue
            if any(c0 <= other <= c1 for other in others):
                continue
            candidates.append((int(c0), int(c1)))
    if not candidates:
        return None
    return random.choice(candidates)


def _empty_span(width: int, token_cols: list[int]) -> tuple[int, int] | None:
    token_cols_set = set(token_cols)
    candidates: list[tuple[int, int]] = []
    for length in range(2, 6):
        for c0 in range(0, width - length + 1):
            c1 = c0 + length - 1
            if any(c0 <= col <= c1 for col in token_cols_set):
                continue
            candidates.append((int(c0), int(c1)))
    if not candidates:
        return None
    return random.choice(candidates)


def _render_input(height: int, width: int, tokens: list[dict], platforms: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for platform in platforms:
        row = int(platform["row"])
        c0 = int(platform["c0"])
        c1 = int(platform["c1"])
        assert 0 <= row < height and 0 <= c0 <= c1 < width
        assert np.all(grid[row, c0 : c1 + 1] == BG), "platform overlap"
        grid[row, c0 : c1 + 1] = int(platform["color"])
    for token in tokens:
        row = int(token["row"])
        col = int(token["col"])
        assert 0 <= row < height and 0 <= col < width
        assert int(grid[row, col]) == BG, "token must start in empty space"
        grid[row, col] = int(token["color"])
    return grid


def _choose_initial_color(path_colors: list[int], *, forbid: set[int] | None = None) -> int:
    blocked = set() if forbid is None else {int(color) for color in forbid}
    if path_colors and random.random() < 0.35:
        candidate = int(random.choice(path_colors))
        if candidate not in blocked:
            return candidate
    return _pick_color(exclude=blocked)


def _gen_attempt_with_spec() -> tuple[np.ndarray, dict] | None:
    height = random.randint(16, 24)
    width = random.randint(12, 22)
    token_count = random.randint(2, 4)
    token_cols = _sample_token_cols(width, token_count)
    if token_cols is None:
        return None

    shared_left = random.randint(0, token_count - 2)
    shared_right = shared_left + 1
    token_rows = [random.randint(0, 2) for _ in range(token_count)]

    platforms: list[dict] = []
    used_rows: set[int] = set()
    path_colors: list[list[int]] = [[] for _ in range(token_count)]

    shared_row_hi = min(8, height - 8)
    if shared_row_hi < 5:
        return None
    shared_row = random.randint(5, shared_row_hi)
    shared_color = _pick_color()
    shared_c0 = max(0, token_cols[shared_left] - random.randint(0, 1))
    shared_c1 = min(width - 1, token_cols[shared_right] + random.randint(0, 1))
    platforms.append({"row": shared_row, "c0": shared_c0, "c1": shared_c1, "color": shared_color})
    used_rows.add(shared_row)
    path_colors[shared_left].append(shared_color)
    path_colors[shared_right].append(shared_color)

    left_col = token_cols[shared_left]
    left_prev_row = shared_row
    left_prev_color = shared_color
    left_extra_count = random.randint(1, 2)
    for remaining_index in range(left_extra_count):
        row = _pick_row(used_rows, left_prev_row + 2, height - 3)
        span = _single_column_span(left_col, token_cols, width)
        if row is None or span is None:
            return None
        color = _pick_color(exclude={left_prev_color})
        platforms.append({"row": row, "c0": span[0], "c1": span[1], "color": color})
        used_rows.add(row)
        left_prev_row = row
        left_prev_color = color
        path_colors[shared_left].append(color)
        if remaining_index + 1 < left_extra_count and left_prev_row + 2 >= height - 2:
            return None

    if random.random() < 0.75:
        row = _pick_row(used_rows, left_prev_row + 2, height - 2)
        span = _single_column_span(left_col, token_cols, width)
        if row is None or span is None:
            return None
        platforms.append({"row": row, "c0": span[0], "c1": span[1], "color": GREY})
        used_rows.add(row)

    right_col = token_cols[shared_right]
    if random.random() < 0.65:
        row = _pick_row(used_rows, shared_row + 2, height - 2)
        span = _single_column_span(right_col, token_cols, width)
        if row is None or span is None:
            return None
        platforms.append({"row": row, "c0": span[0], "c1": span[1], "color": GREY})
        used_rows.add(row)

    for idx, col in enumerate(token_cols):
        if idx in {shared_left, shared_right}:
            continue
        gate_count = random.choices([0, 1, 2], weights=[3, 4, 2], k=1)[0]
        prev_row = 4
        prev_color = None
        for _ in range(gate_count):
            row = _pick_row(used_rows, prev_row + 2, height - 3)
            span = _single_column_span(col, token_cols, width)
            if row is None or span is None:
                return None
            color = _pick_color(exclude=set() if prev_color is None else {prev_color})
            platforms.append({"row": row, "c0": span[0], "c1": span[1], "color": color})
            used_rows.add(row)
            prev_row = row
            prev_color = color
            path_colors[idx].append(color)
        if gate_count == 0 or random.random() < 0.65:
            row = _pick_row(used_rows, prev_row + 2, height - 2)
            span = _single_column_span(col, token_cols, width)
            if row is None or span is None:
                return None
            platforms.append({"row": row, "c0": span[0], "c1": span[1], "color": GREY})
            used_rows.add(row)

    untouched_count = random.randint(1, 2)
    for _ in range(untouched_count):
        row = _pick_row(used_rows, 4, height - 2)
        span = _empty_span(width, token_cols)
        if row is None or span is None:
            return None
        color = _pick_color()
        platforms.append({"row": row, "c0": span[0], "c1": span[1], "color": color})
        used_rows.add(row)

    tokens: list[dict] = []
    for idx, col in enumerate(token_cols):
        forbid = {shared_color} if idx == shared_right else set()
        color = _choose_initial_color(path_colors[idx], forbid=forbid)
        tokens.append({"row": token_rows[idx], "col": col, "color": color})

    grid = _render_input(height, width, tokens, platforms)
    expected, spec_trace = _simulate_spec(height, width, tokens, platforms)
    actual, grid_trace = _simulate_grid(grid)

    if not np.array_equal(expected, actual):
        return None
    if not spec_trace["consumed"]:
        return None
    if sum(1 for platform_index in range(len(platforms)) if not spec_trace["active"][platform_index] and int(platforms[platform_index]["color"]) != GREY) == 0:
        return None
    if sum(1 for platform_index in range(len(platforms)) if spec_trace["active"][platform_index] and int(platforms[platform_index]["color"]) != GREY) == 0:
        return None
    if len(spec_trace["traces"][shared_left]["hit_colors"]) < 2:
        return None
    if spec_trace["traces"][shared_right]["hit_colors"]:
        return None

    reverse_output, reverse_trace = _simulate_spec(height, width, tokens, platforms, reverse=True)
    if np.array_equal(expected, reverse_output):
        return None
    if reverse_trace["traces"][shared_right]["hit_colors"] != (shared_color,):
        return None

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None

    return grid, {
        "height": height,
        "width": width,
        "tokens": tokens,
        "platforms": platforms,
        "shared_left": shared_left,
        "shared_right": shared_right,
        "expected": expected,
        "grid_trace": grid_trace,
        "reverse_output": reverse_output,
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        generated = _gen_attempt_with_spec()
        if generated is not None:
            return generated[0]
    raise RuntimeError("failed to generate a valid serial-gravity gate puzzle")
