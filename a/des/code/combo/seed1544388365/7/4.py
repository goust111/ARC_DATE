from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SIGNAL_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _nonblack_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == BLACK:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if visited[nr, nc] or int(grid[nr, nc]) == BLACK:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            components.append(comp)

    return components


def _trace_path(grid: np.ndarray, component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    comp_set = set(component)
    endpoints = []
    for cell in component:
        degree = sum((nr, nc) in comp_set for nr, nc in _neighbors4(cell[0], cell[1], h, w))
        assert degree <= 2, "pipe must not branch"
        if degree == 1:
            endpoints.append(cell)

    assert len(endpoints) == 2, "each pipe must have exactly two endpoints"
    source_candidates = [pos for pos in endpoints if int(grid[pos]) not in (BLACK, GREY, MAROON)]
    sink_candidates = [pos for pos in endpoints if int(grid[pos]) == MAROON]
    assert len(source_candidates) == 1, "each pipe must have one colored source endpoint"
    assert len(sink_candidates) == 1, "each pipe must have one maroon sink endpoint"

    source = source_candidates[0]
    sink = sink_candidates[0]
    path = [source]
    prev: tuple[int, int] | None = None
    cur = source

    while cur != sink:
        next_cells = [nb for nb in _neighbors4(cur[0], cur[1], h, w) if nb in comp_set and nb != prev]
        assert len(next_cells) == 1, "simple pipe should have a unique forward step"
        prev, cur = cur, next_cells[0]
        path.append(cur)

    assert len(path) == len(component), "component must be a single simple path"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color along its pipe, replacing the signal color at every filter gate.

    Input structure
    - The background is black.
    - Each pipe is a 4-connected, one-cell-wide, non-branching path.
    - One endpoint is a colored source, the other is a maroon sink.
    - Internal grey cells may be replaced by colored filter gates.

    Rule
    - Trace each pipe from its source endpoint to its maroon sink endpoint.
    - The signal starts with the source color and is overwritten by every gate color it visits.
    - Grey pipe cells are recolored with the signal color carried when the signal reaches them.
    - Gate cells and source cells stay unchanged, while the sink becomes the final signal color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for component in _nonblack_components(g):
        path = _trace_path(g, component)
        signal = int(g[path[0]])
        assert signal not in (BLACK, GREY, MAROON)

        for idx, cell in enumerate(path[1:], start=1):
            color = int(g[cell])
            is_sink = idx == len(path) - 1
            if is_sink:
                assert color == MAROON
                out[cell] = signal
            elif color == GREY:
                out[cell] = signal
            else:
                assert color not in (BLACK, MAROON)
                signal = color

    return out


# --------------- generator (gpt-5) ---------------

_RNG = np.random.default_rng()

MIN_H, MAX_H = 10, 18
MIN_W, MAX_W = 10, 18
MAX_PATHS = 4
PATH_BUILD_ATTEMPTS = 60
SCENE_ATTEMPTS = 400


def _positive_composition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    if parts == total:
        return [1] * parts
    cuts = np.sort(_RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    out: list[int] = []
    prev = 0
    for cut in cuts:
        value = int(cut)
        out.append(value - prev)
        prev = value
    out.append(total - prev)
    return out


def _materialize_runs(runs: list[tuple[str, int]], v_sign: int, h_sign: int) -> list[tuple[int, int]]:
    r = 0
    c = 0
    coords = [(0, 0)]
    for axis, steps in runs:
        dr = v_sign if axis == "v" else 0
        dc = h_sign if axis == "h" else 0
        for _ in range(steps):
            r += dr
            c += dc
            coords.append((r, c))
    return coords


def _relative_path() -> list[tuple[int, int]]:
    kind = ("straight", "elbow", "stair")[int(_RNG.integers(0, 3))]
    v_sign = -1 if int(_RNG.integers(0, 2)) == 0 else 1
    h_sign = -1 if int(_RNG.integers(0, 2)) == 0 else 1

    if kind == "straight":
        axis = "v" if int(_RNG.integers(0, 2)) == 0 else "h"
        steps = int(_RNG.integers(2, 9))
        return _materialize_runs([(axis, steps)], v_sign, h_sign)

    if kind == "elbow":
        v_steps = int(_RNG.integers(1, 7))
        h_steps = int(_RNG.integers(1, 7))
        first = "v" if int(_RNG.integers(0, 2)) == 0 else "h"
        second = "h" if first == "v" else "v"
        runs = [(first, v_steps if first == "v" else h_steps), (second, h_steps if second == "h" else v_steps)]
        return _materialize_runs(runs, v_sign, h_sign)

    while True:
        v_parts = int(_RNG.integers(2, 4))
        h_parts = int(_RNG.integers(2, 4))
        if abs(v_parts - h_parts) <= 1:
            break
    v_steps = int(_RNG.integers(v_parts, 8))
    h_steps = int(_RNG.integers(h_parts, 8))
    v_runs = _positive_composition(v_steps, v_parts)
    h_runs = _positive_composition(h_steps, h_parts)
    if v_parts > h_parts:
        first = "v"
    elif h_parts > v_parts:
        first = "h"
    else:
        first = "v" if int(_RNG.integers(0, 2)) == 0 else "h"

    runs: list[tuple[str, int]] = []
    vi = 0
    hi = 0
    axis = first
    while vi < v_parts or hi < h_parts:
        if axis == "v":
            if vi < v_parts:
                runs.append(("v", v_runs[vi]))
                vi += 1
            axis = "h"
        else:
            if hi < h_parts:
                runs.append(("h", h_runs[hi]))
                hi += 1
            axis = "v"
    return _materialize_runs(runs, v_sign, h_sign)


def _translate_path(
    rel_path: list[tuple[int, int]],
    blocked: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    rel = np.asarray(rel_path, dtype=int)
    min_r = int(rel[:, 0].min())
    max_r = int(rel[:, 0].max())
    min_c = int(rel[:, 1].min())
    max_c = int(rel[:, 1].max())

    row_lo = 1 - min_r
    row_hi = (h - 2) - max_r
    col_lo = 1 - min_c
    col_hi = (w - 2) - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(PATH_BUILD_ATTEMPTS):
        base_r = int(_RNG.integers(row_lo, row_hi + 1))
        base_c = int(_RNG.integers(col_lo, col_hi + 1))
        path = [(base_r + r, base_c + c) for r, c in rel_path]
        if all(not blocked[r, c] for r, c in path):
            return path
    return None


def _mark_blocked(blocked: np.ndarray, path: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in path:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _gate_positions(path_len: int) -> list[int]:
    internal = list(range(1, path_len - 1))
    internal_count = len(internal)
    assert internal_count >= 1

    modes = ["none"]
    if internal_count >= 2:
        modes.append("single")
    if internal_count >= 3:
        modes.append("multi")
    mode = modes[int(_RNG.integers(0, len(modes)))]

    if mode == "none":
        return []
    if mode == "single":
        return [int(_RNG.choice(internal))]

    max_gates = min(4, internal_count - 1)
    gate_count = int(_RNG.integers(2, max_gates + 1))
    picks = np.sort(_RNG.choice(np.array(internal), size=gate_count, replace=False))
    return [int(x) for x in picks]


def _make_path_spec(path: list[tuple[int, int]]) -> dict[str, object]:
    source_color = int(_RNG.choice(SIGNAL_COLORS))
    gate_indices = _gate_positions(len(path))
    gate_colors: dict[int, int] = {}
    current = source_color
    for idx in gate_indices:
        choices = [color for color in SIGNAL_COLORS if color != current]
        gate_color = int(_RNG.choice(np.array(choices, dtype=int)))
        gate_colors[idx] = gate_color
        current = gate_color
    return {
        "path": path,
        "source_color": source_color,
        "gate_colors": gate_colors,
    }


def _paint_input(grid: np.ndarray, spec: dict[str, object]) -> None:
    path = spec["path"]
    source_color = int(spec["source_color"])
    gate_colors = spec["gate_colors"]
    assert isinstance(path, list)
    assert isinstance(gate_colors, dict)

    grid[path[0]] = source_color
    grid[path[-1]] = MAROON
    for idx, cell in enumerate(path[1:-1], start=1):
        grid[cell] = int(gate_colors.get(idx, GREY))


def _expected_output(inp: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        path = spec["path"]
        source_color = int(spec["source_color"])
        gate_colors = spec["gate_colors"]
        assert isinstance(path, list)
        assert isinstance(gate_colors, dict)
        signal = source_color
        for idx, cell in enumerate(path[1:], start=1):
            if idx == len(path) - 1:
                out[cell] = signal
            elif idx in gate_colors:
                signal = int(gate_colors[idx])
            else:
                out[cell] = signal
    return out


def _scene_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(MIN_H, MAX_H + 1))
    w = int(_RNG.integers(MIN_W, MAX_W + 1))
    grid = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    max_paths = min(MAX_PATHS, 1 + (min(h, w) >= 12) + (min(h, w) >= 15) + (h * w >= 220))
    path_count = int(_RNG.integers(1, max_paths + 1))
    specs: list[dict[str, object]] = []

    for _ in range(path_count):
        placed = None
        for _ in range(PATH_BUILD_ATTEMPTS):
            rel_path = _relative_path()
            placed = _translate_path(rel_path, blocked)
            if placed is not None:
                break
        if placed is None:
            return None
        spec = _make_path_spec(placed)
        _paint_input(grid, spec)
        _mark_blocked(blocked, placed)
        specs.append(spec)

    expected = _expected_output(grid, specs)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        grid = _scene_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-filter scene")
