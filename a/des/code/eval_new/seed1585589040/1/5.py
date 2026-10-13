import random

import numpy as np


BG = 0
GREY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
COUNT_PAIRS = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (1, 4),
    (4, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
)
GEN_ATTEMPTS = 400


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _legend_candidates(grid: np.ndarray) -> list[tuple[str, int, int, list[int]]]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int, list[int]]] = []

    if h >= 2:
        mask = (grid[0] != BG) & (grid[0] != GREY)
        for start, end in _runs(mask):
            colors = [int(v) for v in grid[0, start:end]]
            if len(colors) >= 2 and len(colors) == len(set(colors)) and np.all(grid[1, start:end] == BG):
                candidates.append(("top", start, end, colors))

        mask = (grid[h - 1] != BG) & (grid[h - 1] != GREY)
        for start, end in _runs(mask):
            colors = [int(v) for v in grid[h - 1, start:end]]
            if len(colors) >= 2 and len(colors) == len(set(colors)) and np.all(grid[h - 2, start:end] == BG):
                candidates.append(("bottom", start, end, colors))

    if w >= 2:
        mask = (grid[:, 0] != BG) & (grid[:, 0] != GREY)
        for start, end in _runs(mask):
            colors = [int(v) for v in grid[start:end, 0]]
            if len(colors) >= 2 and len(colors) == len(set(colors)) and np.all(grid[start:end, 1] == BG):
                candidates.append(("left", start, end, colors))

        mask = (grid[:, w - 1] != BG) & (grid[:, w - 1] != GREY)
        for start, end in _runs(mask):
            colors = [int(v) for v in grid[start:end, w - 1]]
            if len(colors) >= 2 and len(colors) == len(set(colors)) and np.all(grid[start:end, w - 2] == BG):
                candidates.append(("right", start, end, colors))

    return candidates


def _find_legend(grid: np.ndarray) -> tuple[list[int], dict[int, int], np.ndarray]:
    candidates = _legend_candidates(grid)
    assert candidates, "expected a border legend with distinct path colors"
    side, start, end, colors = max(candidates, key=lambda item: len(item[3]))

    legend_mask = np.zeros(grid.shape, dtype=bool)
    if side == "top":
        legend_mask[0, start:end] = True
    elif side == "bottom":
        legend_mask[grid.shape[0] - 1, start:end] = True
    elif side == "left":
        legend_mask[start:end, 0] = True
    else:
        legend_mask[start:end, grid.shape[1] - 1] = True

    priority = {color: idx for idx, color in enumerate(colors)}
    return colors, priority, legend_mask


def _trace_color(grid: np.ndarray, legend_mask: np.ndarray, row: int, col: int, dr: int, dc: int) -> int | None:
    h, w = grid.shape
    rr, cc = row + dr, col + dc
    while 0 <= rr < h and 0 <= cc < w and not legend_mask[rr, cc]:
        value = int(grid[rr, cc])
        if value == GREY:
            rr += dr
            cc += dc
            continue
        if value == BG:
            return None
        return value
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Resolve grey path crossings by using the edge legend to choose the visible path color.

    Input structure:
    - The background is black.
    - Several single-pixel-wide colored paths run orthogonally across the grid; a path may bend by 90 degrees but keeps one color.
    - True crossings between paths are hidden as grey pixels.
    - A straight legend on one grid edge lists each path color exactly once, in visible line order from highest to lowest priority.

    Rule:
    - Detect the border legend and read its color order.
    - For every non-legend grey pixel, trace outward in the four cardinal directions to find which path colors reach that crossing.
    - Recolor the crossing with the highest-priority participating path color from the legend.
    - Leave all other pixels unchanged.

    Key insight:
    - The grey pixel does not identify its owner directly; the solver must recover the crossing participants from surrounding path geometry, then apply the separate edge legend as an over-under priority table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _legend, priority, legend_mask = _find_legend(g)
    out = g.copy()

    for row, col in np.argwhere((g == GREY) & (~legend_mask)):
        seen: set[int] = set()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            color = _trace_color(g, legend_mask, int(row), int(col), dr, dc)
            if color is not None:
                seen.add(int(color))
        if seen:
            winner = min(seen, key=lambda color: priority.get(color, len(priority)))
            out[int(row), int(col)] = int(winner)

    return out


def _paint_segment(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = color
    else:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = color


def _local_layout(n_h: int, n_v: int) -> dict[str, list[int] | int]:
    left_cols = list(range(n_h))
    v_start = n_h + 2
    v_cols = [v_start + 3 * idx for idx in range(n_v)]
    right_start = v_cols[-1] + 2
    right_cols = [right_start + idx for idx in range(n_h)]

    top_rows = list(range(n_v))
    h_start = n_v + 2
    h_rows = [h_start + 3 * idx for idx in range(n_h)]
    bottom_start = h_rows[-1] + 2
    bottom_rows = [bottom_start + idx for idx in range(n_v)]

    return {
        "left_cols": left_cols,
        "right_cols": right_cols,
        "top_rows": top_rows,
        "bottom_rows": bottom_rows,
        "h_rows": h_rows,
        "v_cols": v_cols,
        "local_h": bottom_rows[-1] + 1,
        "local_w": right_cols[-1] + 1,
    }


def _add_horizontal_stub(grid: np.ndarray, path: dict[str, int], local_h: int) -> bool:
    endpoint = random.choice(("left", "right"))
    col = path["c0"] if endpoint == "left" else path["c1"]
    row = path["row"]

    options: list[tuple[int, int]] = []
    if row > 0:
        options.append((-1, min(2, row)))
    if row + 1 < local_h:
        options.append((1, min(2, local_h - 1 - row)))
    if not options:
        return False

    dr, max_len = random.choice(options)
    length = random.randint(1, max_len)
    _paint_segment(grid, row, col, row + dr * length, col, path["color"])
    return True


def _add_vertical_stub(grid: np.ndarray, path: dict[str, int], local_w: int) -> bool:
    endpoint = random.choice(("top", "bottom"))
    row = path["r0"] if endpoint == "top" else path["r1"]
    col = path["col"]

    options: list[tuple[int, int]] = []
    if col > 0:
        options.append((-1, min(2, col)))
    if col + 1 < local_w:
        options.append((1, min(2, local_w - 1 - col)))
    if not options:
        return False

    dc, max_len = random.choice(options)
    length = random.randint(1, max_len)
    _paint_segment(grid, row, col, row, col + dc * length, path["color"])
    return True


def _maybe_add_stubs(
    grid: np.ndarray,
    h_paths: list[dict[str, int]],
    v_paths: list[dict[str, int]],
    local_h: int,
    local_w: int,
) -> None:
    if not random.choice((False, True)):
        return

    bent = 0
    choices = [("h", idx) for idx in range(len(h_paths))] + [("v", idx) for idx in range(len(v_paths))]
    random.shuffle(choices)

    for kind, idx in choices:
        if random.random() < 0.5:
            if kind == "h":
                bent += int(_add_horizontal_stub(grid, h_paths[idx], local_h))
            else:
                bent += int(_add_vertical_stub(grid, v_paths[idx], local_w))

    if bent > 0:
        return

    kind, idx = random.choice(choices)
    if kind == "h":
        _add_horizontal_stub(grid, h_paths[idx], local_h)
    else:
        _add_vertical_stub(grid, v_paths[idx], local_w)


def _place_legend(grid: np.ndarray, side: str, colors: list[int], start: int) -> None:
    h, w = grid.shape
    n = len(colors)
    if side == "top":
        grid[0, start : start + n] = colors
    elif side == "bottom":
        grid[h - 1, start : start + n] = colors
    elif side == "left":
        grid[start : start + n, 0] = colors
    else:
        grid[start : start + n, w - 1] = colors


def _canvas_spec(local_h: int, local_w: int, side: str) -> tuple[tuple[int, int], int, int]:
    min_top = 2 if side == "top" else 1
    min_bottom = 2 if side == "bottom" else 1
    min_left = 2 if side == "left" else 1
    min_right = 2 if side == "right" else 1

    spare_h = 30 - local_h - min_top - min_bottom
    spare_w = 30 - local_w - min_left - min_right
    assert spare_h >= 0 and spare_w >= 0

    extra_top = random.randint(0, spare_h)
    extra_bottom = random.randint(0, spare_h - extra_top)
    extra_left = random.randint(0, spare_w)
    extra_right = random.randint(0, spare_w - extra_left)

    pad_top = min_top + extra_top
    pad_bottom = min_bottom + extra_bottom
    pad_left = min_left + extra_left
    pad_right = min_right + extra_right

    shape = (local_h + pad_top + pad_bottom, local_w + pad_left + pad_right)
    return shape, pad_top, pad_left


def _embed(local: np.ndarray, shape: tuple[int, int], row_off: int, col_off: int) -> np.ndarray:
    local_h, local_w = local.shape
    grid = np.full(shape, BG, dtype=int)
    grid[row_off : row_off + local_h, col_off : col_off + local_w] = local
    return grid


def _legend_start(shape: tuple[int, int], side: str, n: int) -> int:
    h, w = shape
    if side in ("top", "bottom"):
        return random.randint(0, w - n)
    return random.randint(0, h - n)


def _expected_local(base: np.ndarray, h_paths: list[dict[str, int]], v_paths: list[dict[str, int]], priority: dict[int, int]) -> np.ndarray:
    out = base.copy()
    for h_path in h_paths:
        for v_path in v_paths:
            winner = min(
                (h_path["color"], v_path["color"]),
                key=lambda color: priority[color],
            )
            out[h_path["row"], v_path["col"]] = winner
    return out


def _input_local(base: np.ndarray, h_paths: list[dict[str, int]], v_paths: list[dict[str, int]]) -> np.ndarray:
    inp = base.copy()
    for h_path in h_paths:
        for v_path in v_paths:
            inp[h_path["row"], v_path["col"]] = GREY
    return inp


def _gen_attempt() -> np.ndarray | None:
    n_h, n_v = random.choice(COUNT_PAIRS)
    layout = _local_layout(n_h, n_v)
    local_h = int(layout["local_h"])
    local_w = int(layout["local_w"])

    path_colors = random.sample(list(PATH_COLORS), n_h + n_v)
    h_colors = path_colors[:n_h]
    v_colors = path_colors[n_h:]
    legend_colors = random.sample(path_colors, len(path_colors))
    priority = {color: idx for idx, color in enumerate(legend_colors)}

    h_paths = [
        {"row": row, "c0": c0, "c1": c1, "color": color}
        for row, c0, c1, color in zip(layout["h_rows"], layout["left_cols"], layout["right_cols"], h_colors)
    ]
    v_paths = [
        {"col": col, "r0": r0, "r1": r1, "color": color}
        for col, r0, r1, color in zip(layout["v_cols"], layout["top_rows"], layout["bottom_rows"], v_colors)
    ]

    base = np.full((local_h, local_w), BG, dtype=int)
    for path in h_paths:
        _paint_segment(base, path["row"], path["c0"], path["row"], path["c1"], path["color"])
    for path in v_paths:
        _paint_segment(base, path["r0"], path["col"], path["r1"], path["col"], path["color"])

    _maybe_add_stubs(base, h_paths, v_paths, local_h, local_w)

    local_inp = _input_local(base, h_paths, v_paths)
    local_out = _expected_local(base, h_paths, v_paths, priority)

    legend_side = random.choice(SIDES)
    shape, row_off, col_off = _canvas_spec(local_h, local_w, legend_side)
    inp = _embed(local_inp, shape, row_off, col_off)
    out = _embed(local_out, shape, row_off, col_off)

    legend_start = _legend_start(shape, legend_side, len(legend_colors))
    _place_legend(inp, legend_side, legend_colors, legend_start)
    _place_legend(out, legend_side, legend_colors, legend_start)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
