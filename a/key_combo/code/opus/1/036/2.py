from __future__ import annotations

import itertools

import numpy as np


BACKGROUND = 0
LINE_COLOR_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GENERATION_RETRIES = 500
MIN_OUTPUT_BG_FRACTION = 0.15
EDGE_CLIP_PROB = 0.35

_RNG = np.random.default_rng()


def _majority_nonzero_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values[values != BACKGROUND], return_counts=True)
    assert colors.size > 0, "each recovered line must have a visible color"
    max_count = int(counts.max())
    winners = [int(color) for color, count in zip(colors, counts) if int(count) == max_count]
    assert len(winners) == 1, "each line must have a unique dominant color"
    return winners[0]


def _extract_lines(grid: np.ndarray) -> dict[str, dict]:
    h, w = grid.shape
    horizontal_rows = [r for r in range(h) if np.all(grid[r, :] != BACKGROUND)]
    vertical_cols = [c for c in range(w) if np.all(grid[:, c] != BACKGROUND)]
    assert horizontal_rows and vertical_cols, "expected at least one line of each orientation"

    lines: dict[str, dict] = {}
    covered = np.zeros_like(grid, dtype=bool)
    for r in horizontal_rows:
        key = f"h{r}"
        lines[key] = {"orientation": "h", "index": int(r), "color": _majority_nonzero_color(grid[r, :])}
        covered[r, :] = True
    for c in vertical_cols:
        key = f"v{c}"
        lines[key] = {"orientation": "v", "index": int(c), "color": _majority_nonzero_color(grid[:, c])}
        covered[:, c] = True

    assert 3 <= len(lines) <= 4, "expected 3 or 4 full-span lines"
    colors = [info["color"] for info in lines.values()]
    assert len(set(colors)) == len(colors), "each line must have its own color"
    assert np.all((grid != BACKGROUND) <= covered), "foreground must belong to the recovered lines"
    return lines


def _intersection_edges(grid: np.ndarray, lines: dict[str, dict]) -> set[tuple[str, str]]:
    horizontal = [key for key, info in lines.items() if info["orientation"] == "h"]
    vertical = [key for key, info in lines.items() if info["orientation"] == "v"]

    edges: set[tuple[str, str]] = set()
    for h_key in horizontal:
        row = lines[h_key]["index"]
        h_color = lines[h_key]["color"]
        for v_key in vertical:
            col = lines[v_key]["index"]
            v_color = lines[v_key]["color"]
            visible = int(grid[row, col])
            if visible == h_color:
                edges.add((v_key, h_key))
            elif visible == v_color:
                edges.add((h_key, v_key))
            else:
                raise AssertionError("each crossing must show one of the two line colors")
    return edges


def _consistent_orders(line_ids: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    valid = []
    for perm in itertools.permutations(line_ids):
        position = {key: idx for idx, key in enumerate(perm)}
        if all(position[lower] < position[upper] for lower, upper in edges):
            valid.append(list(perm))
    return valid


def _infer_unique_order(grid: np.ndarray) -> tuple[dict[str, dict], list[str]]:
    lines = _extract_lines(grid)
    orders = _consistent_orders(list(lines), _intersection_edges(grid, lines))
    assert len(orders) == 1, "crossings must imply a unique full layer order"
    return lines, orders[0]


def _render_expanded(shape: tuple[int, int], lines: dict[str, dict], order: list[str]) -> np.ndarray:
    h, w = shape
    out = np.full(shape, BACKGROUND, dtype=int)
    for rank, key in enumerate(order, start=1):
        info = lines[key]
        color = int(info["color"])
        radius = rank - 1
        if info["orientation"] == "h":
            row = int(info["index"])
            r0 = max(0, row - radius)
            r1 = min(h, row + radius + 1)
            out[r0:r1, :] = color
        else:
            col = int(info["index"])
            c0 = max(0, col - radius)
            c1 = min(w, col + radius + 1)
            out[:, c0:c1] = color
    return out


def _render_occluded_input(shape: tuple[int, int], lines: dict[str, dict], order: list[str]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for key in order:
        info = lines[key]
        color = int(info["color"])
        if info["orientation"] == "h":
            grid[int(info["index"]), :] = color
        else:
            grid[:, int(info["index"])] = color
    return grid


def _edges_from_true_order(order: list[str], orientations: dict[str, str]) -> set[tuple[str, str]]:
    position = {key: idx for idx, key in enumerate(order)}
    horizontal = [key for key, ori in orientations.items() if ori == "h"]
    vertical = [key for key, ori in orientations.items() if ori == "v"]
    edges: set[tuple[str, str]] = set()
    for h_key in horizontal:
        for v_key in vertical:
            if position[h_key] < position[v_key]:
                edges.add((h_key, v_key))
            else:
                edges.add((v_key, h_key))
    return edges


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the unique occlusion order of full-span crossing lines and redraw ranked-width bands.

    Input structure:
    - A black background contains 3 or 4 distinct-colored 1-pixel straight lines.
    - Each line spans its entire row or its entire column.
    - Horizontal and vertical lines cross, and each crossing cell shows only the higher line's color.

    Rule:
    - Recover every full row or full column foreground line, including the line color hidden at crossings.
    - Use each crossing cell's visible color to collect above/below constraints between the two lines that meet there.
    - Find the unique bottom-to-top total order consistent with all crossings.
    - Repaint the lines from bottom to top, expanding rank K to thickness 2K-1 perpendicular to the line and clipping at the grid boundary.

    Key insight:
    - The crossing winners encode a layer order, and that same inferred rank controls both how thick each band becomes and when it is painted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == BACKGROUND), "expected black background"
    lines, order = _infer_unique_order(g)
    return _render_expanded(g.shape, lines, order)


def _sample_line_counts() -> tuple[int, int]:
    if float(_RNG.random()) < 0.5:
        return (2, 1) if float(_RNG.random()) < 0.5 else (1, 2)
    return 2, 2


def _sample_shape(total_lines: int) -> tuple[int, int]:
    if total_lines == 4:
        height = int(_RNG.integers(16, 21))
        width = int(_RNG.integers(17, 22))
    else:
        height = int(_RNG.integers(14, 20))
        width = int(_RNG.integers(15, 21))
    return height, width


def _sample_unique_true_order(orientations: dict[str, str]) -> list[str]:
    line_ids = list(orientations)
    for _ in range(200):
        perm = [line_ids[idx] for idx in _RNG.permutation(len(line_ids))]
        edges = _edges_from_true_order(perm, orientations)
        if len(_consistent_orders(line_ids, edges)) == 1:
            return perm
    raise RuntimeError("failed to sample a uniquely inferable layer order")


def _interior_positions(length: int) -> list[int]:
    positions = [p for p in range(length) if 2 <= p <= length - 3]
    if positions:
        return positions
    positions = [p for p in range(length) if 1 <= p <= length - 2]
    return positions or list(range(length))


def _axis_positions(
    line_ids: list[str],
    length: int,
    total_lines: int,
    forced_edge_id: str | None,
) -> dict[str, int] | None:
    shuffled_ids = list(line_ids)
    _RNG.shuffle(shuffled_ids)
    min_gap = 0 if len(shuffled_ids) == 1 else (6 if total_lines == 4 else 4)
    hard_edges = sorted(set([0, length - 1]))
    interior = set(_interior_positions(length))

    def valid(values: tuple[int, ...], relaxed: bool) -> bool:
        if len(values) != len(set(values)):
            return False
        if len(values) == 2 and abs(values[0] - values[1]) < min_gap:
            return False
        for key, pos in zip(shuffled_ids, values):
            if key == forced_edge_id:
                if pos not in hard_edges:
                    return False
            else:
                if relaxed:
                    continue
                if pos not in interior:
                    return False
                if forced_edge_id is not None and pos in hard_edges:
                    return False
        return True

    all_assignments = list(itertools.permutations(range(length), len(shuffled_ids)))
    strict = [values for values in all_assignments if valid(values, relaxed=False)]
    candidates = strict or [values for values in all_assignments if valid(values, relaxed=True)]
    if not candidates:
        return None
    choice = candidates[int(_RNG.integers(0, len(candidates)))]
    return {key: int(pos) for key, pos in zip(shuffled_ids, choice)}


def _build_lines(
    horizontal_count: int,
    vertical_count: int,
    order: list[str],
    shape: tuple[int, int],
) -> dict[str, dict] | None:
    total_lines = horizontal_count + vertical_count
    orientations = {f"h{i}": "h" for i in range(horizontal_count)}
    orientations.update({f"v{i}": "v" for i in range(vertical_count)})

    forced_edge_id = None
    if float(_RNG.random()) < EDGE_CLIP_PROB:
        forced_edge_id = order[int(_RNG.integers(1, len(order)))]

    horizontal_ids = [key for key, ori in orientations.items() if ori == "h"]
    vertical_ids = [key for key, ori in orientations.items() if ori == "v"]
    row_map = _axis_positions(horizontal_ids, shape[0], total_lines, forced_edge_id)
    col_map = _axis_positions(vertical_ids, shape[1], total_lines, forced_edge_id)
    if row_map is None or col_map is None:
        return None

    colors = [int(c) for c in _RNG.choice(LINE_COLOR_POOL, size=total_lines, replace=False)]
    _RNG.shuffle(colors)

    lines: dict[str, dict] = {}
    for color, key in zip(colors, sorted(orientations)):
        lines[key] = {"orientation": orientations[key], "color": int(color)}
        if orientations[key] == "h":
            lines[key]["index"] = row_map[key]
        else:
            lines[key]["index"] = col_map[key]
    return lines


def _validate_generated(grid: np.ndarray, lines: dict[str, dict], order: list[str]) -> bool:
    try:
        parsed_lines, parsed_order = _infer_unique_order(grid)
    except Exception:
        return False

    expected_colors = [int(lines[key]["color"]) for key in order]
    parsed_colors = [int(parsed_lines[key]["color"]) for key in parsed_order]
    if parsed_colors != expected_colors:
        return False

    for h_key, h_info in lines.items():
        if h_info["orientation"] != "h":
            continue
        for v_key, v_info in lines.items():
            if v_info["orientation"] != "v":
                continue
            position = {key: idx for idx, key in enumerate(order)}
            winner = h_key if position[h_key] > position[v_key] else v_key
            if int(grid[int(h_info["index"]), int(v_info["index"])]) != int(lines[winner]["color"]):
                return False

    expected = _render_expanded(grid.shape, lines, order)
    if np.mean(expected == BACKGROUND) < MIN_OUTPUT_BG_FRACTION:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    horizontal_count, vertical_count = _sample_line_counts()
    orientations = {f"h{i}": "h" for i in range(horizontal_count)}
    orientations.update({f"v{i}": "v" for i in range(vertical_count)})
    order = _sample_unique_true_order(orientations)
    shape = _sample_shape(horizontal_count + vertical_count)
    lines = _build_lines(horizontal_count, vertical_count, order, shape)
    if lines is None:
        return None

    grid = _render_occluded_input(shape, lines, order)
    return grid if _validate_generated(grid, lines, order) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
