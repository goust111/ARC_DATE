from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
SEED_COLORS = (1, 2, 3, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _nonblack_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    idx = 0
    while idx < len(values):
        if int(values[idx]) == BLACK:
            idx += 1
            continue
        start = idx
        while idx < len(values) and int(values[idx]) != BLACK:
            idx += 1
        runs.append((start, idx - 1))
    return runs


def _extract_horizontal_pipes(grid: np.ndarray) -> list[dict]:
    pipes: list[dict] = []
    height, width = grid.shape
    for r in range(height):
        row = grid[r]
        for start, end in _nonblack_runs(row):
            if end - start + 1 < 2:
                continue
            vals = row[start : end + 1]
            if len(vals) > 2 and not np.all(vals[1:-1] == GREY):
                continue
            seed_cols: list[int] = []
            if int(vals[0]) != GREY:
                seed_cols.append(start)
            if int(vals[-1]) != GREY:
                seed_cols.append(end)
            if len(seed_cols) != 1:
                continue
            seed_c = seed_cols[0]
            seed_color = int(grid[r, seed_c])
            other_c = end if seed_c == start else start
            if seed_color in (BLACK, GREY, YELLOW):
                continue
            if int(grid[r, other_c]) != GREY:
                continue
            pipes.append(
                {
                    "orientation": "h",
                    "row": r,
                    "left": start,
                    "right": end,
                    "seed_col": seed_c,
                    "seed_color": seed_color,
                }
            )
    return pipes


def _extract_vertical_pipes(grid: np.ndarray) -> list[dict]:
    pipes: list[dict] = []
    height, width = grid.shape
    for c in range(width):
        col = grid[:, c]
        for start, end in _nonblack_runs(col):
            if end - start + 1 < 2:
                continue
            vals = col[start : end + 1]
            if len(vals) > 2 and not np.all(vals[1:-1] == GREY):
                continue
            seed_rows: list[int] = []
            if int(vals[0]) != GREY:
                seed_rows.append(start)
            if int(vals[-1]) != GREY:
                seed_rows.append(end)
            if len(seed_rows) != 1:
                continue
            seed_r = seed_rows[0]
            seed_color = int(grid[seed_r, c])
            other_r = end if seed_r == start else start
            if seed_color in (BLACK, GREY, YELLOW):
                continue
            if int(grid[other_r, c]) != GREY:
                continue
            pipes.append(
                {
                    "orientation": "v",
                    "col": c,
                    "top": start,
                    "bottom": end,
                    "seed_row": seed_r,
                    "seed_color": seed_color,
                }
            )
    return pipes


def _coverage_mask(shape: tuple[int, int], horizontal: list[dict], vertical: list[dict]) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    for pipe in horizontal:
        cover[pipe["row"], pipe["left"] : pipe["right"] + 1] += 1
    for pipe in vertical:
        cover[pipe["top"] : pipe["bottom"] + 1, pipe["col"]] += 1
    return cover


def _solve_from_pipes(shape: tuple[int, int], horizontal: list[dict], vertical: list[dict]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    h_contrib: dict[tuple[int, int], int] = {}
    v_contrib: dict[tuple[int, int], int] = {}

    for pipe in horizontal:
        r = pipe["row"]
        seed_c = pipe["seed_col"]
        color = pipe["seed_color"]
        for c in range(pipe["left"], pipe["right"] + 1):
            dist = abs(c - seed_c)
            h_contrib[(r, c)] = color if dist == 0 or dist % 2 == 1 else GREY

    for pipe in vertical:
        c = pipe["col"]
        seed_r = pipe["seed_row"]
        color = pipe["seed_color"]
        for r in range(pipe["top"], pipe["bottom"] + 1):
            dist = abs(r - seed_r)
            v_contrib[(r, c)] = color if dist == 0 or dist % 2 == 1 else GREY

    all_cells = set(h_contrib) | set(v_contrib)
    for cell in all_cells:
        h_val = h_contrib.get(cell)
        v_val = v_contrib.get(cell)
        if h_val is not None and v_val is not None:
            if h_val != GREY and v_val != GREY:
                out[cell] = YELLOW
            elif h_val != GREY:
                out[cell] = h_val
            elif v_val != GREY:
                out[cell] = v_val
            else:
                out[cell] = GREY
        else:
            out[cell] = h_val if h_val is not None else v_val

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow alternating color bands along grey pipes and mark colored crossings in yellow.

    Input structure
    - Background is black.
    - Grey one-pixel-wide horizontal and vertical pipes may cross on shared grey cells.
    - Each pipe has exactly one colored seed on one endpoint.

    Rule
    - Starting from each seed, color every odd step along that pipe with the seed color.
    - Even steps stay grey; the seed itself stays colored.
    - At a crossing, keep the single colored result if only one pipe is colored there.
    - If both horizontal and vertical results are colored, paint the crossing yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((g == BLACK) | (g == GREY) | (g == YELLOW) | np.isin(g, SEED_COLORS))
    assert not np.any(g == YELLOW), "yellow is reserved for the output crossing marker"

    horizontal = _extract_horizontal_pipes(g)
    vertical = _extract_vertical_pipes(g)
    assert horizontal or vertical, "expected at least one pipe"

    cover = _coverage_mask(g.shape, horizontal, vertical)
    assert int(cover.max()) <= 2
    assert np.all(cover[g != BLACK] >= 1), "every non-black cell must belong to a parsed pipe"
    assert np.all(g[cover >= 2] == GREY), "crossings must be grey in the input"

    return _solve_from_pipes(g.shape, horizontal, vertical)


def _sample_spaced_positions(lo: int, hi: int, count: int, min_gap: int, rng: random.Random) -> list[int] | None:
    population = list(range(lo, hi + 1))
    if len(population) < count:
        return None
    for _ in range(120):
        picks = sorted(rng.sample(population, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _crossing_label(horizontal_color: bool, vertical_color: bool) -> int:
    if horizontal_color and vertical_color:
        return 3
    if horizontal_color:
        return 1
    if vertical_color:
        return 2
    return 0


def _horizontal_has_color(pipe: dict, col: int, seed_side: str) -> bool:
    dist = col - pipe["left"] if seed_side == "left" else pipe["right"] - col
    return dist % 2 == 1


def _vertical_has_color(pipe: dict, row: int, seed_side: str) -> bool:
    dist = row - pipe["top"] if seed_side == "top" else pipe["bottom"] - row
    return dist % 2 == 1


def _choose_seed_sides(horizontal: list[dict], vertical: list[dict], rng: random.Random) -> tuple[list[str], list[str]] | None:
    best_score: tuple[int, int, int, int, int, int] | None = None
    best_choices: list[tuple[list[str], list[str]]] = []

    for bits in itertools.product((0, 1), repeat=len(horizontal) + len(vertical)):
        h_sides = ["left" if bit == 0 else "right" for bit in bits[: len(horizontal)]]
        v_sides = ["top" if bit == 0 else "bottom" for bit in bits[len(horizontal) :]]
        labels: list[int] = []
        for i, h_pipe in enumerate(horizontal):
            for j, v_pipe in enumerate(vertical):
                h_on = _horizontal_has_color(h_pipe, v_pipe["col"], h_sides[i])
                v_on = _vertical_has_color(v_pipe, h_pipe["row"], v_sides[j])
                labels.append(_crossing_label(h_on, v_on))

        label_set = set(labels)
        counts = [labels.count(label) for label in range(4)]
        score = (
            len(label_set),
            int(3 in label_set),
            int(1 in label_set),
            int(2 in label_set),
            int(0 in label_set),
            -max(counts),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_choices = [(h_sides, v_sides)]
        elif score == best_score:
            best_choices.append((h_sides, v_sides))

    if best_score is None:
        return None
    if best_score[0] < 3 or best_score[1] == 0:
        return None
    return rng.choice(best_choices)


def _render_input(shape: tuple[int, int], horizontal: list[dict], vertical: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for pipe in horizontal:
        grid[pipe["row"], pipe["left"] : pipe["right"] + 1] = GREY
    for pipe in vertical:
        grid[pipe["top"] : pipe["bottom"] + 1, pipe["col"]] = GREY
    for pipe in horizontal:
        seed_c = pipe["left"] if pipe["seed_side"] == "left" else pipe["right"]
        grid[pipe["row"], seed_c] = pipe["seed_color"]
    for pipe in vertical:
        seed_r = pipe["top"] if pipe["seed_side"] == "top" else pipe["bottom"]
        grid[seed_r, pipe["col"]] = pipe["seed_color"]
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(10, 18)
    width = rng.randint(10, 18)
    n_horizontal = rng.randint(2, 3)
    n_vertical = rng.randint(2, 3)

    rows = _sample_spaced_positions(2, height - 3, n_horizontal, 2, rng)
    cols = _sample_spaced_positions(2, width - 3, n_vertical, 2, rng)
    if rows is None or cols is None:
        return None
    if len({row % 2 for row in rows}) < 2 or len({col % 2 for col in cols}) < 2:
        return None

    horizontal: list[dict] = []
    for row in rows:
        left = rng.randint(0, cols[0] - 1)
        right = rng.randint(cols[-1] + 1, width - 1)
        horizontal.append(
            {
                "orientation": "h",
                "row": row,
                "left": left,
                "right": right,
            }
        )

    vertical: list[dict] = []
    for col in cols:
        top = rng.randint(0, rows[0] - 1)
        bottom = rng.randint(rows[-1] + 1, height - 1)
        vertical.append(
            {
                "orientation": "v",
                "col": col,
                "top": top,
                "bottom": bottom,
            }
        )

    seed_sides = _choose_seed_sides(horizontal, vertical, rng)
    if seed_sides is None:
        return None
    h_sides, v_sides = seed_sides

    colors = list(SEED_COLORS)
    rng.shuffle(colors)
    for pipe, color, seed_side in zip(horizontal, colors[:n_horizontal], h_sides):
        pipe["seed_color"] = color
        pipe["seed_side"] = seed_side
        pipe["seed_col"] = pipe["left"] if seed_side == "left" else pipe["right"]
    for pipe, color, seed_side in zip(vertical, colors[n_horizontal : n_horizontal + n_vertical], v_sides):
        pipe["seed_color"] = color
        pipe["seed_side"] = seed_side
        pipe["seed_row"] = pipe["top"] if seed_side == "top" else pipe["bottom"]

    inp = _render_input((height, width), horizontal, vertical)
    expected = _solve_from_pipes(inp.shape, horizontal, vertical)

    parsed_h = _extract_horizontal_pipes(inp)
    parsed_v = _extract_vertical_pipes(inp)
    if len(parsed_h) != n_horizontal or len(parsed_v) != n_vertical:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-growth input")
