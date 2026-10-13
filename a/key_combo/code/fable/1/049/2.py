from __future__ import annotations

from collections import Counter

import numpy as np


def _infer_walls(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    """Return the wall color, the full-height wall columns, and the pipe columns."""
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, int, list[int]]] = []
    for color in sorted(set(map(int, grid.flatten()))):
        full_cols = [c for c in range(width) if bool(np.all(grid[:, c] == int(color)))]
        if len(full_cols) >= 2:
            candidates.append((len(full_cols), int(color), full_cols))

    assert candidates, "expected a wall color that fills multiple columns"
    _count, wall_color, wall_cols = max(candidates, key=lambda item: (item[0], item[1]))
    wall_cols = sorted(int(c) for c in wall_cols)
    assert wall_cols[0] == 0 and wall_cols[-1] == width - 1, "pipes must be enclosed by boundary walls"
    assert height >= 3 and width >= 5, "grid is too small to contain walled pipes"
    assert all(b - a == 2 for a, b in zip(wall_cols, wall_cols[1:])), "expected single-cell-wide pipes"

    pipe_cols = [a + 1 for a, _b in zip(wall_cols, wall_cols[1:])]
    assert pipe_cols, "expected at least one pipe column"
    assert sorted(set(wall_cols) | set(pipe_cols)) == list(range(width)), "unexpected extra columns"
    return int(wall_color), wall_cols, pipe_cols


def _infer_background(grid: np.ndarray, pipe_cols: list[int], wall_color: int) -> int:
    values = [int(v) for v in grid[:, pipe_cols].ravel() if int(v) != int(wall_color)]
    assert values, "expected non-wall pipe cells"
    return int(Counter(values).most_common(1)[0][0])


def _pipe_foreground(grid: np.ndarray, col: int, bg: int, wall_color: int) -> tuple[list[int], list[int]]:
    rows: list[int] = []
    colors: list[int] = []
    for r in range(int(grid.shape[0])):
        value = int(grid[r, col])
        if value in (int(bg), int(wall_color)):
            continue
        rows.append(int(r))
        colors.append(int(value))

    assert rows, "every pipe must contain a token"
    assert len(rows) >= 2, "every pipe must contain at least one gate below the token"
    assert all(b > a for a, b in zip(rows, rows[1:])), "pipe foreground must be vertically ordered"
    assert rows[-1] < int(grid.shape[0]) - 1, "bottom cell must stay free for the falling token"
    return rows, colors


def _simulate_chain(token_color: int, gate_colors: list[int]) -> tuple[list[int], int, int | None]:
    recolored = [int(color) for color in gate_colors]
    current = int(token_color)
    for idx, gate_color in enumerate(gate_colors):
        gate_color = int(gate_color)
        if gate_color == current:
            return recolored, int(current), int(idx)
        recolored[idx] = int(current)
        current = int(gate_color)
    return recolored, int(current), None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let each token fall through its one-cell-wide pipe, swapping colors with mismatched gates until it stops.

    Input structure:
    - Full-height vertical wall columns split the grid into enclosed single-cell-wide pipe columns.
    - Each pipe column contains one topmost token pixel and one or more lower gate pixels; all other pipe cells are background.
    - Tokens and gates use non-background, non-wall colors, while wall columns remain solid.

    Rule:
    - In each pipe, treat the topmost colored pixel as the token and the lower colored pixels as gates in top-to-bottom order.
    - When the falling token reaches a gate of a different color, swap their colors and keep falling; when it reaches a gate of the same color, stop one cell above that gate.
    - If no gate matches, stop at the bottom cell of the pipe; keep all gates in their post-swap colors and redraw the token at its final resting cell.

    Key insight:
    - The swap cascade is equivalent to finding the first adjacent equal pair in the sequence [token, gate1, gate2, ...]: all gates above that stop point inherit the color from one step above, and the token ends with the last color it carried.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    height, _width = map(int, g.shape)

    wall_color, wall_cols, pipe_cols = _infer_walls(g)
    bg = _infer_background(g, pipe_cols, wall_color)

    out = np.full_like(g, int(bg))
    out[:, wall_cols] = int(wall_color)

    for col in pipe_cols:
        rows, colors = _pipe_foreground(g, int(col), int(bg), int(wall_color))
        token_row = int(rows[0])
        token_color = int(colors[0])
        gate_rows = [int(r) for r in rows[1:]]
        gate_colors = [int(c) for c in colors[1:]]

        recolored_gates, final_color, stop_idx = _simulate_chain(int(token_color), gate_colors)
        for row, color in zip(gate_rows, recolored_gates):
            out[int(row), int(col)] = int(color)

        if stop_idx is None:
            stop_row = int(height - 1)
        else:
            stop_row = int(gate_rows[int(stop_idx)] - 1)
        assert stop_row >= token_row, "token cannot rise while falling"
        assert stop_row not in gate_rows, "token stop cell must remain distinct from gates"
        out[int(stop_row), int(col)] = int(final_color)

    return out


GEN_MAX_ATTEMPTS = 400
GEN_HEIGHT_RANGE = (12, 27)
GEN_PIPE_COUNT_RANGE = (4, 9)
GEN_BRANCHES = ("immediate", "later", "bottom")
GEN_COLORS = tuple(range(10))
_RNG = np.random.default_rng()


def _pick_distinct(colors: list[int], forbidden: set[int], count: int) -> list[int]:
    choices = [color for color in colors if int(color) not in forbidden]
    picked = _RNG.choice(np.array(choices, dtype=int), size=int(count), replace=False)
    return [int(v) for v in picked.tolist()]


def _pick_other(palette: list[int], previous: int) -> int:
    choices = [color for color in palette if int(color) != int(previous)]
    return int(choices[int(_RNG.integers(0, len(choices)))])


def _sample_gate_rows(height: int, min_count: int) -> list[int]:
    start_choices = [start for start in (2, 3) if len(list(range(start, height - 1, 2))) >= int(min_count)]
    assert start_choices, "height is too small for the requested gate count"
    start = int(start_choices[int(_RNG.integers(0, len(start_choices)))])
    candidates = list(range(start, height - 1, 2))
    max_count = min(len(candidates), 6)
    gate_count = int(_RNG.integers(int(min_count), int(max_count) + 1))
    chosen = _RNG.choice(np.array(candidates, dtype=int), size=int(gate_count), replace=False)
    return sorted(int(v) for v in chosen.tolist())


def _build_gate_sequence(branch: str, gate_count: int, palette: list[int]) -> tuple[int, list[int]]:
    token_color = int(palette[int(_RNG.integers(0, len(palette)))])
    gates: list[int] = []

    if branch == "immediate":
        gates.append(int(token_color))
        previous = int(token_color)
        for _ in range(1, gate_count):
            previous = _pick_other(palette, int(previous))
            gates.append(int(previous))
        return int(token_color), gates

    if branch == "later":
        stop_idx = int(_RNG.integers(1, gate_count))
        previous = int(token_color)
        for _ in range(stop_idx):
            previous = _pick_other(palette, int(previous))
            gates.append(int(previous))
        gates.append(int(previous))
        for _ in range(stop_idx + 1, gate_count):
            previous = _pick_other(palette, int(previous))
            gates.append(int(previous))
        return int(token_color), gates

    assert branch == "bottom", f"unknown branch {branch}"
    previous = int(token_color)
    for _ in range(gate_count):
        previous = _pick_other(palette, int(previous))
        gates.append(int(previous))
    return int(token_color), gates


def _branch_from_sequence(token_color: int, gate_colors: list[int]) -> str:
    _recolored, _final_color, stop_idx = _simulate_chain(int(token_color), gate_colors)
    if stop_idx is None:
        return "bottom"
    if int(stop_idx) == 0:
        return "immediate"
    return "later"


def _render_input(height: int, bg: int, wall_color: int, pipe_specs: list[dict]) -> np.ndarray:
    width = 2 * len(pipe_specs) + 1
    grid = np.full((height, width), int(bg), dtype=int)
    grid[:, ::2] = int(wall_color)
    for pipe_idx, spec in enumerate(pipe_specs):
        col = 2 * int(pipe_idx) + 1
        grid[0, col] = int(spec["token_color"])
        for row, color in zip(spec["gate_rows"], spec["gate_colors"]):
            grid[int(row), int(col)] = int(color)
    return grid


def _render_output(height: int, bg: int, wall_color: int, pipe_specs: list[dict]) -> np.ndarray:
    width = 2 * len(pipe_specs) + 1
    out = np.full((height, width), int(bg), dtype=int)
    out[:, ::2] = int(wall_color)
    for pipe_idx, spec in enumerate(pipe_specs):
        col = 2 * int(pipe_idx) + 1
        recolored, final_color, stop_idx = _simulate_chain(
            int(spec["token_color"]),
            [int(color) for color in spec["gate_colors"]],
        )
        for row, color in zip(spec["gate_rows"], recolored):
            out[int(row), int(col)] = int(color)
        if stop_idx is None:
            stop_row = int(height - 1)
        else:
            stop_row = int(spec["gate_rows"][int(stop_idx)] - 1)
        out[int(stop_row), int(col)] = int(final_color)
    return out


def _validate_plan(height: int, bg: int, wall_color: int, pipe_specs: list[dict]) -> None:
    width = 2 * len(pipe_specs) + 1
    assert height <= 30 and width <= 30, "ARC grids must stay within 30x30"
    assert {spec["branch"] for spec in pipe_specs} == set(GEN_BRANCHES), "must cover all key branches"
    for spec in pipe_specs:
        rows = [int(r) for r in spec["gate_rows"]]
        assert rows == sorted(rows), "gate rows must be sorted"
        assert rows[0] >= 2 and rows[-1] < height - 1, "gates must leave room above and below"
        assert all(b - a >= 2 for a, b in zip(rows, rows[1:])), "gates must stay visually separated"
        actual_branch = _branch_from_sequence(int(spec["token_color"]), [int(v) for v in spec["gate_colors"]])
        assert actual_branch == spec["branch"], "planned branch must match the actual swap cascade"
        assert int(spec["token_color"]) not in (int(bg), int(wall_color))
        assert all(int(color) not in (int(bg), int(wall_color)) for color in spec["gate_colors"])


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(*GEN_HEIGHT_RANGE))
    n_pipes = int(_RNG.integers(*GEN_PIPE_COUNT_RANGE))

    bg, wall_color = _pick_distinct(list(GEN_COLORS), set(), 2)
    palette_size = int(_RNG.integers(3, 7))
    palette = _pick_distinct(list(GEN_COLORS), {int(bg), int(wall_color)}, palette_size)

    branches = list(GEN_BRANCHES)
    while len(branches) < n_pipes:
        branches.append(str(GEN_BRANCHES[int(_RNG.integers(0, len(GEN_BRANCHES)))]))
    _RNG.shuffle(branches)

    pipe_specs: list[dict] = []
    for branch in branches:
        min_gates = 1 if branch == "immediate" else 3 if branch == "later" else 2
        gate_rows = _sample_gate_rows(int(height), int(min_gates))
        token_color, gate_colors = _build_gate_sequence(str(branch), len(gate_rows), palette)
        pipe_specs.append(
            {
                "branch": str(branch),
                "token_color": int(token_color),
                "gate_rows": [int(v) for v in gate_rows],
                "gate_colors": [int(v) for v in gate_colors],
            }
        )

    try:
        _validate_plan(int(height), int(bg), int(wall_color), pipe_specs)
    except AssertionError:
        return None

    inp = _render_input(int(height), int(bg), int(wall_color), pipe_specs)
    expected = _render_output(int(height), int(bg), int(wall_color), pipe_specs)
    try:
        pred = main(inp)
    except Exception:
        return None
    if not np.array_equal(pred, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-swap cascade puzzle")
