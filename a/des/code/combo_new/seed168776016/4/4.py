from __future__ import annotations

from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()

# --------------- generator ---------------

DIVIDER_COLOR = 5
MAX_GENERATION_ATTEMPTS = 400
PIPE_COUNT_OPTIONS = (2, 3, 4)
LEFT_WIDTH_RANGE = (8, 15)  # inclusive, exclusive
RIGHT_WIDTH_MIN = 5
RIGHT_WIDTH_MAX = 13
TOP_PAD_OPTIONS = (0, 1, 2)
BOTTOM_PAD_OPTIONS = (0, 1, 2)
INTER_PIPE_GAP_OPTIONS = (1, 2)
MAX_TURNS = 4


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if bool(seen[r, c]):
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            coords.append((int(rr), int(cc)))
            for dr, dc in DIR4:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(np.array(coords, dtype=int))
    return out


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, int]] = []
    for c in range(w - 1):
        color = int(grid[0, c])
        if color == int(bg):
            continue
        if not bool(np.all(grid[:, c] == color)):
            continue
        if bool(np.all(grid[:, c + 1 :] == int(bg))):
            candidates.append((int(c), int(color)))
    if not candidates:
        raise ValueError("divider not found")
    return candidates[0]


def _trace_pipe(coords: np.ndarray) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in coords.tolist()}
    if len(cells) < 2:
        raise ValueError("pipe must have at least two cells")

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    endpoints: list[tuple[int, int]] = []
    for r, c in cells:
        ns = [(r + dr, c + dc) for dr, dc in DIR4 if (r + dr, c + dc) in cells]
        neighbors[(r, c)] = ns
        deg = len(ns)
        if deg == 1:
            endpoints.append((r, c))
        elif deg > 2:
            raise ValueError("pipe branches")

    if len(endpoints) != 2:
        raise ValueError("pipe must have exactly two endpoints")

    start = min(endpoints)
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [pt for pt in neighbors[cur] if pt != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            raise ValueError("pipe is not a simple path")
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt

    if len(path) != len(cells):
        raise ValueError("pipe revisits cells or forms a loop")
    return path


def _turn_count(coords: np.ndarray) -> int:
    path = _trace_pipe(coords)
    turns = 0
    for i in range(1, len(path) - 1):
        d1 = (int(path[i][0] - path[i - 1][0]), int(path[i][1] - path[i - 1][1]))
        d2 = (int(path[i + 1][0] - path[i][0]), int(path[i + 1][1] - path[i][1]))
        if d1 != d2:
            turns += 1
    return int(turns)


def _scan_key(coords: np.ndarray) -> tuple[int, int]:
    return min((int(r), int(c)) for r, c in coords.tolist())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each orthogonal pipe's bends and record that count as dots on the right panel.

    Input structure:
    - The dominant color is background.
    - A full-height constant divider column splits the grid into a left pipe field and a right empty panel.
    - The left field contains several different-colored width-1 orthogonal pipes, each a 4-connected simple path with two endpoints.
    - The right panel starts as background only.

    Rule:
    - Keep the left field and the divider unchanged.
    - For each pipe, trace its unique endpoint-to-endpoint path and count the cells where the step direction changes by 90 degrees.
    - Order pipes by the first cell encountered in top-to-bottom, then left-to-right scan order on the left field.
    - Use the top rows of the right panel as a count board: row i contains that pipe's color repeated for exactly its turn count, with one background cell between consecutive dots.

    Key insight:
    - The right side is a separate tally panel: each color reports only how many bends its matching left-side pipe makes, not its length or endpoint positions.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    divider_col, _divider_color = _find_divider(g, bg)
    left = g[:, :divider_col]

    pipes: list[dict[str, object]] = []
    for color in np.unique(left):
        color = int(color)
        if color == int(bg):
            continue
        for coords in _components(left == color):
            pipes.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "turns": int(_turn_count(coords)),
                }
            )

    pipes.sort(key=lambda item: _scan_key(item["coords"]))  # type: ignore[arg-type]

    out = g.copy()
    out[:, divider_col + 1 :] = int(bg)
    panel_start = int(divider_col + 1)
    for row_idx, pipe in enumerate(pipes):
        if row_idx >= out.shape[0]:
            break
        turns = int(pipe["turns"])
        color = int(pipe["color"])
        for k in range(turns):
            cc = int(panel_start + 2 * k)
            if cc < out.shape[1]:
                out[row_idx, cc] = color
    return out


def _build_pipe(turns: int, width: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        start_axis = str(rng.choice(["H", "V"]))
        segment_count = int(turns + 1)
        axes = [start_axis if i % 2 == 0 else ("V" if start_axis == "H" else "H") for i in range(segment_count)]

        if "H" in axes and width < 2:
            continue

        v_lengths: list[int] = []
        for idx, axis in enumerate(axes):
            if axis != "V":
                continue
            has_prev_h = idx > 0 and axes[idx - 1] == "H"
            has_next_h = idx + 1 < segment_count and axes[idx + 1] == "H"
            min_len = 2 if has_prev_h and has_next_h else 1
            max_len = 3
            v_lengths.append(int(rng.integers(min_len, max_len + 1)))

        coords: list[tuple[int, int]] = []
        x = int(rng.integers(width))
        r = 0
        coords.append((int(r), int(x)))
        v_pos = 0

        for axis in axes:
            if axis == "H":
                choices = [xx for xx in range(width) if xx != x]
                if not choices:
                    break
                x2 = int(rng.choice(choices))
                step = 1 if x2 > x else -1
                for cc in range(x + step, x2 + step, step):
                    coords.append((int(r), int(cc)))
                x = int(x2)
            else:
                length = int(v_lengths[v_pos])
                v_pos += 1
                for _ in range(length):
                    r += 1
                    coords.append((int(r), int(x)))
        else:
            arr = np.array(coords, dtype=int)
            if len({tuple(map(int, rc)) for rc in arr.tolist()}) != len(arr):
                continue
            try:
                if _turn_count(arr) != int(turns):
                    continue
            except ValueError:
                continue
            return arr

    raise RuntimeError("failed to build pipe")


def _sample_turn_counts(pipe_count: int, rng: np.random.Generator) -> list[int]:
    counts = [int(rng.integers(1, MAX_TURNS + 1)) for _ in range(pipe_count)]
    if pipe_count >= 3 and bool(rng.random() < 0.35):
        zero_idx = int(rng.integers(1, pipe_count - 1))
        counts[zero_idx] = 0

    if MAX_TURNS >= 2 and max(counts) < 2:
        counts[int(rng.integers(pipe_count))] = 2

    if len(set(counts)) == 1:
        idx = int(rng.integers(pipe_count))
        replacement = int((counts[idx] + 1) % (MAX_TURNS + 1))
        if replacement == 0 and (pipe_count < 3 or idx in (0, pipe_count - 1)):
            replacement = 1
        counts[idx] = replacement

    return counts


def _render_expected(
    grid: np.ndarray,
    divider_col: int,
    bg: int,
    pipe_specs: list[dict[str, int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[:, divider_col + 1 :] = int(bg)
    for row_idx, spec in enumerate(pipe_specs):
        for k in range(int(spec["turns"])):
            out[row_idx, divider_col + 1 + 2 * k] = int(spec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    bg_choices = [c for c in range(10) if c != DIVIDER_COLOR]
    bg = int(RNG.choice(bg_choices))

    pipe_count = int(RNG.choice(PIPE_COUNT_OPTIONS))
    turn_counts = _sample_turn_counts(pipe_count, RNG)
    max_turns = max(turn_counts)

    left_width = int(RNG.integers(*LEFT_WIDTH_RANGE))
    min_right_width = max(RIGHT_WIDTH_MIN, 2 * max_turns - 1 if max_turns > 0 else 1)
    if min_right_width > RIGHT_WIDTH_MAX:
        return None
    right_width = int(RNG.integers(min_right_width, RIGHT_WIDTH_MAX + 1))
    total_width = int(left_width + 1 + right_width)
    if total_width > 30:
        return None

    left_pad = int(RNG.integers(0, 3))
    right_pad = int(RNG.integers(1, 4))
    usable_width = int(left_width - left_pad - right_pad)
    if usable_width < 2:
        return None

    pipe_colors = [c for c in range(10) if c not in (bg, DIVIDER_COLOR)]
    RNG.shuffle(pipe_colors)
    pipe_colors = pipe_colors[:pipe_count]

    bands: list[np.ndarray] = []
    band_heights: list[int] = []
    for turns in turn_counts:
        coords = _build_pipe(int(turns), usable_width, RNG)
        bands.append(coords)
        band_heights.append(int(coords[:, 0].max()) + 1)

    top_pad = int(RNG.choice(TOP_PAD_OPTIONS))
    bottom_pad = int(RNG.choice(BOTTOM_PAD_OPTIONS))
    gaps = [int(RNG.choice(INTER_PIPE_GAP_OPTIONS)) for _ in range(pipe_count - 1)]
    height = int(top_pad + sum(band_heights) + sum(gaps) + bottom_pad)
    if height > 30:
        return None

    grid = np.full((height, total_width), bg, dtype=int)
    divider_col = int(left_width)
    grid[:, divider_col] = DIVIDER_COLOR

    pipe_specs: list[dict[str, int]] = []
    row_cursor = int(top_pad)
    for idx, coords in enumerate(bands):
        shifted = np.array(coords, copy=True)
        shifted[:, 0] += int(row_cursor)
        shifted[:, 1] += int(left_pad)

        if int(shifted[:, 0].max()) >= height or int(shifted[:, 1].max()) >= divider_col:
            return None
        if not bool(np.all(grid[shifted[:, 0], shifted[:, 1]] == bg)):
            return None

        color = int(pipe_colors[idx])
        grid[shifted[:, 0], shifted[:, 1]] = color
        turns = int(turn_counts[idx])
        if _turn_count(shifted) != turns:
            return None

        pipe_specs.append({"color": color, "turns": turns})
        row_cursor += int(band_heights[idx])
        if idx < len(gaps):
            row_cursor += int(gaps[idx])

    if not bool(np.all(grid[:, divider_col + 1 :] == bg)):
        return None

    expected = _render_expected(grid, divider_col, bg, pipe_specs)
    actual = main(grid)
    if not bool(np.array_equal(actual, expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-counting ARC input")
