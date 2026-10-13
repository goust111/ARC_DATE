from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


# Local helpers follow arc_utils.py conventions:
# - grid.shape == (H, W)
# - coords are always (r, c)
# - colors are plain ARC ints
BLACK = 0
BLUE = 1
GREY = 5
TEAL = 8

MAX_SIDE = 30
RETRY_LIMIT = 600


@dataclass(frozen=True)
class Bar:
    row: int
    c0: int
    c1: int


@dataclass(frozen=True)
class PathResult:
    start_col: int
    init_dc: int
    path: tuple[tuple[int, int], ...]
    wall_bounces: int
    hit_bar: bool
    first_wall_entry_index: int | None
    bar_turn_entry_index: int | None


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _find_bar(grid: np.ndarray) -> Bar:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "expected one grey obstacle bar"
    rows = {int(r) for r, _c in coords}
    assert len(rows) == 1, "grey obstacle must lie on one row"
    row = next(iter(rows))
    cols = sorted(int(c) for _r, c in coords)
    c0, c1 = cols[0], cols[-1]
    assert cols == list(range(c0, c1 + 1)), "grey obstacle must be contiguous"
    h, w = grid.shape
    assert 0 < row < h - 1, "grey obstacle must be internal"
    assert 0 < c0 and c1 < w - 1, "grey obstacle must be suspended away from side walls"
    return Bar(row=row, c0=c0, c1=c1)


def _simulate_path(h: int, w: int, bar: Bar, start_col: int, init_dc: int) -> PathResult:
    r = h - 1
    c = int(start_col)
    dr = -1
    dc = int(init_dc)
    path = [(r, c)]
    wall_bounces = 0
    first_wall_entry_index: int | None = None
    bar_turn_entry_index: int | None = None

    while True:
        next_dc = dc
        if not (0 <= c + next_dc < w):
            next_dc = -next_dc
            wall_bounces += 1
            if first_wall_entry_index is None:
                first_wall_entry_index = len(path)

        next_dr = dr
        next_r = r + next_dr
        next_c = c + next_dc
        hit_bar = False
        if next_dr == -1 and next_r == bar.row and bar.c0 <= next_c <= bar.c1:
            next_dr = 1
            next_r = r + next_dr
            hit_bar = True
            if bar_turn_entry_index is None:
                bar_turn_entry_index = len(path)

        if not (0 <= next_r < h):
            break

        r, c, dr, dc = next_r, next_c, next_dr, next_dc
        path.append((r, c))

        if hit_bar:
            # Vertical direction only flips once because the beam can then only travel downward.
            pass

    return PathResult(
        start_col=int(start_col),
        init_dc=int(init_dc),
        path=tuple(path),
        wall_bounces=int(wall_bounces),
        hit_bar=bar_turn_entry_index is not None,
        first_wall_entry_index=first_wall_entry_index,
        bar_turn_entry_index=bar_turn_entry_index,
    )


def _enumerate_candidates(h: int, w: int, bar: Bar) -> list[PathResult]:
    out = []
    for start_col in range(w):
        for init_dc in (-1, 1):
            out.append(_simulate_path(h, w, bar, start_col, init_dc))
    return out


def _render_output(shape: tuple[int, int], bar: Bar, path: tuple[tuple[int, int], ...]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in path:
        out[r, c] = BLUE
    out[bar.row, bar.c0 : bar.c1 + 1] = GREY
    return out


def _upward_trace_without_bar(h: int, w: int, start_col: int, init_dc: int) -> list[dict[str, int | bool]]:
    r = h - 1
    c = int(start_col)
    dc = int(init_dc)
    wall_bounces = 0
    moves: list[dict[str, int | bool]] = []

    while True:
        bounced_now = False
        next_dc = dc
        if not (0 <= c + next_dc < w):
            next_dc = -next_dc
            wall_bounces += 1
            bounced_now = True

        next_r = r - 1
        next_c = c + next_dc
        if next_r < 0:
            break

        moves.append(
            {
                "from_r": r,
                "from_c": c,
                "to_r": next_r,
                "to_c": next_c,
                "wall_bounces": wall_bounces,
                "bounced_now": bounced_now,
            }
        )
        r, c, dc = next_r, next_c, next_dc

    return moves


def _segment_indices(result: PathResult) -> tuple[list[int], list[int], list[int]]:
    assert result.first_wall_entry_index is not None
    assert result.bar_turn_entry_index is not None
    n = len(result.path)

    pre_wall = list(range(1, result.first_wall_entry_index))
    between = list(range(result.first_wall_entry_index, result.bar_turn_entry_index))
    after_bar = list(range(result.bar_turn_entry_index, n))
    return pre_wall, between, after_bar


def _choose_bar_interval(
    w: int,
    collision_col: int,
    blocked_cols: set[int],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    options: list[tuple[int, int]] = []
    max_len = min(w - 2, 9)
    for length in range(3, max_len + 1):
        c0_lo = max(1, collision_col - length + 1)
        c0_hi = min(collision_col, w - 1 - length)
        for c0 in range(c0_lo, c0_hi + 1):
            c1 = c0 + length - 1
            if any(c0 <= c <= c1 for c in blocked_cols):
                continue
            options.append((c0, c1))
    if not options:
        return None
    return options[int(rng.integers(len(options)))]


def _pick_remnants(
    true_result: PathResult,
    candidates: list[PathResult],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    true_cells = list(true_result.path)
    true_sets = [set(candidate.path) for candidate in candidates]
    true_idx = next(
        i
        for i, candidate in enumerate(candidates)
        if candidate.start_col == true_result.start_col and candidate.init_dc == true_result.init_dc
    )
    others = {i for i in range(len(candidates)) if i != true_idx}
    pre_wall, between, after_bar = _segment_indices(true_result)
    pools = [pre_wall, between, after_bar]

    selected: list[int] = []

    def survivors(indices: list[int]) -> set[int]:
        keep = set(others)
        for idx in indices:
            cell = true_cells[idx]
            keep = {j for j in keep if cell in true_sets[j]}
        return keep

    for pool in pools:
        options = [idx for idx in pool if idx not in selected]
        rng.shuffle(options)
        best_idx = min(options, key=lambda idx: (len(survivors(selected + [idx])), idx))
        selected.append(best_idx)

    while survivors(selected):
        options = [idx for idx in range(1, len(true_cells)) if idx not in selected]
        rng.shuffle(options)
        best_idx = min(options, key=lambda idx: (len(survivors(selected + [idx])), idx))
        selected.append(best_idx)
        if len(selected) > 7:
            return []

    changed = True
    while changed:
        changed = False
        for idx in list(selected):
            trial = [x for x in selected if x != idx]
            if not trial:
                continue
            if survivors(trial):
                continue
            chosen_set = set(trial)
            if all(any(seg_idx in chosen_set for seg_idx in pool) for pool in pools):
                selected = trial
                changed = True
                break

    selected.sort()
    return [true_cells[idx] for idx in selected]


def _build_input(shape: tuple[int, int], bar: Bar, remnants: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[bar.row, bar.c0 : bar.c1 + 1] = GREY
    for r, c in remnants:
        assert grid[r, c] == BLACK
        grid[r, c] = TEAL
    return grid


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, PathResult]:
    h = int(rng.integers(12, 25))
    w = int(rng.integers(10, 25))
    start_col = int(rng.integers(w))
    init_dc = -1 if int(rng.integers(2)) == 0 else 1

    upward_moves = _upward_trace_without_bar(h, w, start_col, init_dc)
    first_wall_move = next((i for i, move in enumerate(upward_moves) if bool(move["bounced_now"])), None)
    if first_wall_move is None:
        raise ValueError("sampled path never hits a side wall before leaving")

    collision_choices: list[tuple[int, int, tuple[int, int]]] = []
    for i, move in enumerate(upward_moves):
        if i <= first_wall_move:
            continue
        bar_row = int(move["to_r"])
        collision_col = int(move["to_c"])
        earlier_cols = {
            int(prev["to_c"])
            for prev in upward_moves[:i]
            if int(prev["to_r"]) == bar_row
        }
        interval = _choose_bar_interval(w, collision_col, earlier_cols, rng)
        if interval is None:
            continue
        collision_choices.append((i, bar_row, interval))

    if not collision_choices:
        raise ValueError("could not place a bar that is first hit after a wall bounce")

    _move_index, bar_row, (bar_c0, bar_c1) = collision_choices[int(rng.integers(len(collision_choices)))]
    bar = Bar(row=bar_row, c0=bar_c0, c1=bar_c1)
    true_result = _simulate_path(h, w, bar, start_col, init_dc)

    if not true_result.hit_bar or true_result.wall_bounces < 1:
        raise ValueError("generated path missed a required reflection")
    if true_result.first_wall_entry_index is None or true_result.bar_turn_entry_index is None:
        raise ValueError("generated path lacks required event markers")

    pre_wall, between, after_bar = _segment_indices(true_result)
    if not pre_wall or not between or len(after_bar) < 2:
        raise ValueError("generated path does not leave enough cells across semantic segments")

    candidates = _enumerate_candidates(h, w, bar)
    remnants = _pick_remnants(true_result, candidates, rng)
    if not remnants:
        raise ValueError("failed to find a sparse unique remnant set")

    rows = {r for r, _c in remnants}
    cols = {c for _r, c in remnants}
    if len(rows) < 3 or len(cols) < 2:
        raise ValueError("remnants are visually too degenerate")

    grid = _build_input((h, w), bar, remnants)
    return grid, true_result


def _validate_generated_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= MAX_SIDE
    assert set(map(int, np.unique(grid))) <= {BLACK, GREY, TEAL}

    bar = _find_bar(grid)
    remnants = [tuple(map(int, coord)) for coord in np.argwhere(grid == TEAL)]
    assert len(remnants) >= 3

    candidates = _enumerate_candidates(*grid.shape, bar)
    remnant_set = set(remnants)
    matches = [candidate for candidate in candidates if remnant_set <= set(candidate.path)]
    assert len(matches) == 1, "input must determine a unique candidate path"
    match = matches[0]

    assert match.hit_bar, "generator must force a bar bounce"
    assert match.wall_bounces >= 1, "generator must force at least one wall bounce"
    assert match.first_wall_entry_index is not None and match.bar_turn_entry_index is not None

    pre_wall, between, after_bar = _segment_indices(match)
    path = list(match.path)
    assert any(path[i] in remnant_set for i in pre_wall), "need remnants before the first wall bounce"
    assert any(path[i] in remnant_set for i in between), "need remnants between wall bounce and bar bounce"
    assert any(path[i] in remnant_set for i in after_bar), "need remnants after the bar bounce"

    out = main(grid)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2 and out.shape == grid.shape
    assert max(out.shape) <= MAX_SIDE
    assert np.array_equal(out, _render_output(grid.shape, bar, match.path))


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the unique bouncing diagonal path that explains the cyan remnants.

    Input structure:
    - The grid has a black background, a single internal grey horizontal bar, and several cyan remnant pixels.
    - The cyan pixels are a subset of one legal diagonal path launched from some bottom-edge cell.
    - The missing full path is not drawn; only the bar and sparse remnants are visible.

    Rule:
    - Enumerate every candidate formed by choosing a bottom-edge start cell and an initial left/right diagonal direction.
    - Simulate each path on cell centers: when the next column would leave the grid, flip the horizontal direction first; when an upward next cell would land on the grey bar, flip the vertical direction and move downward instead.
    - Keep the unique candidate whose visited cells cover every cyan remnant, then paint its whole path blue while preserving the grey bar and leaving all other cells black.

    Key insight:
    - The start point is hidden, so the task is inverse inference: only one legal trajectory is consistent with all remnants once both side-wall reflection and the internal bar bounce are applied.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    assert max(grid.shape) <= MAX_SIDE, f"grid exceeds {MAX_SIDE}x{MAX_SIDE}"

    bar = _find_bar(grid)
    remnants = {tuple(map(int, coord)) for coord in np.argwhere(grid == TEAL)}
    assert remnants, "expected at least one cyan remnant"
    assert not any(r == bar.row and bar.c0 <= c <= bar.c1 for r, c in remnants), "remnants cannot overlap the bar"

    matches = []
    for candidate in _enumerate_candidates(*grid.shape, bar):
        if remnants <= set(candidate.path):
            matches.append(candidate)

    assert len(matches) == 1, "cyan remnants must determine a unique legal path"
    return _render_output(grid.shape, bar, matches[0].path)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(RETRY_LIMIT):
        try:
            grid, _true_result = _generate_case(rng)
            _validate_generated_input(grid)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid residual-bounce inference input")
