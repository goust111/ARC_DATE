import numpy as np


BG = 0
MIX_COLOR = 3
PATH_PALETTE = (1, 2, 4, 5, 6, 7, 8, 9)
GRID_MIN = 10
GRID_MAX = 22
MIN_PATHS_PER_AXIS = 2
MAX_PATHS_PER_AXIS = 4
MIN_GAP = 2
MAX_GENERATION_ATTEMPTS = 400


def _dominant_non_bg(values: np.ndarray, bg: int) -> int | None:
    foreground = values[values != bg]
    if foreground.size == 0:
        return None
    colors, counts = np.unique(foreground, return_counts=True)
    return int(colors[np.argmax(counts)])


def _detect_signal(edge_a: int, edge_b: int, path_color: int, bg: int) -> int | None:
    candidates = [edge for edge in (int(edge_a), int(edge_b)) if edge != bg and edge != path_color]
    if not candidates:
        return None
    return candidates[0]


def _detect_paths(grid: np.ndarray, bg: int) -> tuple[list[dict[str, int | None]], list[dict[str, int | None]]]:
    height, width = grid.shape
    horizontal = []
    vertical = []

    for r in range(height):
        if int(grid[r, 0]) == bg or int(grid[r, width - 1]) == bg:
            continue
        path_color = _dominant_non_bg(grid[r, :], bg)
        if path_color is None:
            continue
        horizontal.append(
            {
                "index": r,
                "path_color": path_color,
                "signal_color": _detect_signal(grid[r, 0], grid[r, width - 1], path_color, bg),
            }
        )

    for c in range(width):
        if int(grid[0, c]) == bg or int(grid[height - 1, c]) == bg:
            continue
        path_color = _dominant_non_bg(grid[:, c], bg)
        if path_color is None:
            continue
        vertical.append(
            {
                "index": c,
                "path_color": path_color,
                "signal_color": _detect_signal(grid[0, c], grid[height - 1, c], path_color, bg),
            }
        )

    return horizontal, vertical


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate endpoint signal colors along straight paths and mix them at crossings.

    Input structure:
    - The background is black.
    - Some single-pixel-wide horizontal and vertical paths span from one grid edge to the opposite edge.
    - Each path may have one border endpoint recolored to a signal color different from the path's original color.
    - Crossings between horizontal and vertical paths appear as black cells in the input.

    Rule:
    - Detect horizontal paths by non-black pixels on both left and right edges, and vertical paths by non-black pixels on both top and bottom edges.
    - On each detected row or column, treat the dominant non-black color as the original path color and any different edge color as that path's signal source.
    - Repaint every signaled path entirely with its signal color; paths with no detected source stay unchanged.
    - At each row-column crossing, equal signal colors stay as that color, while different signal colors turn green.

    Key insight:
    - The long path color only locates the carrier line; the true control information is the lone edge pixel whose color breaks that line and therefore determines how the whole path should be repainted.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())

    horizontal, vertical = _detect_paths(g, bg)
    out = g.copy()

    for path in horizontal:
        signal = path["signal_color"]
        if signal is not None:
            out[int(path["index"]), :] = int(signal)

    for path in vertical:
        signal = path["signal_color"]
        if signal is not None:
            out[:, int(path["index"])] = int(signal)

    for h_path in horizontal:
        r = int(h_path["index"])
        h_signal = h_path["signal_color"]
        for v_path in vertical:
            c = int(v_path["index"])
            v_signal = v_path["signal_color"]

            if h_signal is None and v_signal is None:
                continue
            if h_signal is None:
                out[r, c] = int(v_signal)
            elif v_signal is None:
                out[r, c] = int(h_signal)
            else:
                out[r, c] = int(h_signal) if int(h_signal) == int(v_signal) else MIX_COLOR

    return out


def _sample_positions(
    rng: np.random.Generator, start: int, stop: int, count: int, min_gap: int
) -> list[int] | None:
    candidates = np.arange(start, stop + 1, dtype=int)
    for _ in range(200):
        picked = sorted(map(int, rng.choice(candidates, size=count, replace=False)))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _choose_signals(
    rng: np.random.Generator, horizontal: list[dict[str, int | str]], vertical: list[dict[str, int | str]]
) -> tuple[int, int, list[int], list[int]] | None:
    sig_a, sig_b = map(int, rng.choice(np.array(PATH_PALETTE, dtype=int), size=2, replace=False))
    h_same = [i for i, path in enumerate(horizontal) if int(path["path_color"]) != sig_a]
    v_same = [i for i, path in enumerate(vertical) if int(path["path_color"]) != sig_a]
    if not h_same or not v_same:
        return None

    rng.shuffle(h_same)
    rng.shuffle(v_same)
    for h_idx in h_same:
        for v_idx in v_same:
            h_diff = [i for i, path in enumerate(horizontal) if i != h_idx and int(path["path_color"]) != sig_b]
            v_diff = [i for i, path in enumerate(vertical) if i != v_idx and int(path["path_color"]) != sig_b]
            if not h_diff and not v_diff:
                continue

            h_signals = [0] * len(horizontal)
            v_signals = [0] * len(vertical)

            for i, path in enumerate(horizontal):
                allowed = [sig for sig in (sig_a, sig_b) if sig != int(path["path_color"])]
                if not allowed:
                    return None
                h_signals[i] = int(rng.choice(np.array(allowed, dtype=int)))

            for i, path in enumerate(vertical):
                allowed = [sig for sig in (sig_a, sig_b) if sig != int(path["path_color"])]
                if not allowed:
                    return None
                v_signals[i] = int(rng.choice(np.array(allowed, dtype=int)))

            h_signals[h_idx] = sig_a
            v_signals[v_idx] = sig_a
            if h_diff and v_diff:
                if bool(rng.integers(0, 2)):
                    h_signals[int(rng.choice(np.array(h_diff, dtype=int)))] = sig_b
                else:
                    v_signals[int(rng.choice(np.array(v_diff, dtype=int)))] = sig_b
            elif h_diff:
                h_signals[int(rng.choice(np.array(h_diff, dtype=int)))] = sig_b
            else:
                v_signals[int(rng.choice(np.array(v_diff, dtype=int)))] = sig_b

            return sig_a, sig_b, h_signals, v_signals

    return None


def _build_input(
    height: int, width: int, horizontal: list[dict[str, int | str]], vertical: list[dict[str, int | str]]
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)

    for path in horizontal:
        grid[int(path["index"]), :] = int(path["path_color"])
    for path in vertical:
        grid[:, int(path["index"])] = int(path["path_color"])

    for h_path in horizontal:
        r = int(h_path["index"])
        for v_path in vertical:
            grid[r, int(v_path["index"])] = BG

    for path in horizontal:
        r = int(path["index"])
        c = 0 if path["source_side"] == "left" else width - 1
        grid[r, c] = int(path["signal_color"])

    for path in vertical:
        c = int(path["index"])
        r = 0 if path["source_side"] == "top" else height - 1
        grid[r, c] = int(path["signal_color"])

    return grid


def _expected_output(
    height: int, width: int, horizontal: list[dict[str, int | str]], vertical: list[dict[str, int | str]]
) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)

    for path in horizontal:
        out[int(path["index"]), :] = int(path["signal_color"])
    for path in vertical:
        out[:, int(path["index"])] = int(path["signal_color"])

    for h_path in horizontal:
        r = int(h_path["index"])
        h_signal = int(h_path["signal_color"])
        for v_path in vertical:
            c = int(v_path["index"])
            v_signal = int(v_path["signal_color"])
            out[r, c] = h_signal if h_signal == v_signal else MIX_COLOR

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))

    max_h = min(MAX_PATHS_PER_AXIS, (height - 1) // MIN_GAP)
    max_v = min(MAX_PATHS_PER_AXIS, (width - 1) // MIN_GAP)
    if max_h < MIN_PATHS_PER_AXIS or max_v < MIN_PATHS_PER_AXIS:
        return None

    h_count = int(rng.integers(MIN_PATHS_PER_AXIS, max_h + 1))
    v_count = int(rng.integers(MIN_PATHS_PER_AXIS, max_v + 1))
    if h_count + v_count > len(PATH_PALETTE):
        return None

    rows = _sample_positions(rng, 1, height - 2, h_count, MIN_GAP)
    cols = _sample_positions(rng, 1, width - 2, v_count, MIN_GAP)
    if rows is None or cols is None:
        return None

    path_colors = list(map(int, rng.choice(np.array(PATH_PALETTE, dtype=int), size=h_count + v_count, replace=False)))
    horizontal = [
        {
            "index": row,
            "path_color": path_colors[i],
            "source_side": "left" if bool(rng.integers(0, 2)) else "right",
        }
        for i, row in enumerate(rows)
    ]
    vertical = [
        {
            "index": col,
            "path_color": path_colors[h_count + i],
            "source_side": "top" if bool(rng.integers(0, 2)) else "bottom",
        }
        for i, col in enumerate(cols)
    ]

    chosen = _choose_signals(rng, horizontal, vertical)
    if chosen is None:
        return None
    _sig_a, _sig_b, h_signals, v_signals = chosen

    for path, signal in zip(horizontal, h_signals):
        path["signal_color"] = int(signal)
    for path, signal in zip(vertical, v_signals):
        path["signal_color"] = int(signal)

    grid = _build_input(height, width, horizontal, vertical)
    expected = _expected_output(height, width, horizontal, vertical)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
