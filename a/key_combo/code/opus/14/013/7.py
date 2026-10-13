from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

MAX_GENERATION_ATTEMPTS = 400
HEIGHT_RANGE = (11, 16)
WIDTH_OPTIONS = (9, 10)

# Each layout is defined relative to one guaranteed finishing row.
# The chosen green pixels generate:
# - one row that reaches the last column,
# - one row that stops because the next channel cell is a green marker,
# - one row that stops at a gap even though later channel cells still exist.
LAYOUT_DELTAS = {
    9: [(-3, 5), (-2, 5), (-1, 5), (1, 6)],
    10: [(-3, 5), (3, 6), (-1, 5), (1, 6)],
}


def _channel_mask(shape: tuple[int, int], greens: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    rows, cols = np.indices((h, w))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in greens:
        mask |= (rows - cols == r - c) | (rows + cols == r + c)
    return mask


def _red_rows(grid: np.ndarray) -> list[int]:
    rows = [int(r) for r, c in np.argwhere(grid == RED)]
    assert len(rows) == len(set(rows)), "each row can contain at most one red marker"
    assert all(int(np.count_nonzero(grid[r] == RED)) == 1 for r in rows), "red markers must be single pixels"
    assert all(int(np.argmax(grid[r] == RED)) == 1 for r in rows), "red markers must be in column 1"
    return sorted(rows)


def _analyze_row(
    grid: np.ndarray,
    channel: np.ndarray,
    row: int,
) -> dict[str, int | bool | str]:
    h, w = grid.shape
    assert 0 <= row < h

    pos = 1
    while pos + 1 < w:
        nxt = pos + 1
        if not channel[row, nxt]:
            reason = "gap"
            break
        if int(grid[row, nxt]) != BG:
            reason = "green" if int(grid[row, nxt]) == GREEN else "blocked"
            break
        pos = nxt
    else:
        reason = "edge"

    if pos == w - 1:
        reason = "finish"

    later_channel = any(bool(channel[row, c]) for c in range(pos + 2, w))
    return {
        "run": pos - 1,
        "final_col": pos,
        "reason": reason,
        "later_channel": later_channel,
    }


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, int | bool | str]]]:
    g = np.asarray(grid, dtype=int)
    greens = [(int(r), int(c)) for r, c in np.argwhere(g == GREEN)]
    channel = _channel_mask(g.shape, greens)

    out = g.copy()
    analyses: dict[int, dict[str, int | bool | str]] = {}
    for row in _red_rows(g):
        analyses[row] = _analyze_row(g, channel, row)
        pos = 1
        while pos + 1 < g.shape[1] and channel[row, pos + 1] and int(out[row, pos + 1]) == BG:
            out[row, pos] = BG
            pos += 1
            out[row, pos] = RED
        if pos == g.shape[1] - 1:
            out[row, pos] = BLUE

    return out, channel, analyses


def _validate_structure(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BG, BLUE, RED, GREEN}), "unexpected colors"
    h, w = g.shape
    assert w >= 3, "the grid must have room for the blue edge, red markers, and at least one interior column"
    assert np.all(g[:, 0] == BLUE), "column 0 must be a full blue bar"
    assert np.all(g[:, 1][g[:, 1] != RED] == BG), "column 1 may only contain red markers or background"
    assert np.all(g[:, 2:] != BLUE), "blue may only appear on the left edge in the input"
    assert np.all(g[:, 2:] != RED), "red markers may only appear in column 1"

    greens = [(int(r), int(c)) for r, c in np.argwhere(g == GREEN)]
    assert 2 <= len(greens) <= 4, "expected 2 to 4 green markers"
    assert all(c >= 2 for _, c in greens), "green markers must not be in columns 0 or 1"

    red_rows = _red_rows(g)
    assert red_rows, "expected at least one red marker"
    for row in red_rows:
        assert int(g[row, 1]) == RED
    return greens, red_rows


def _semantic_checks(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    greens, red_rows = _validate_structure(g)
    assert 8 <= g.shape[1] <= 10, "generator keeps widths compact so a full-length lane remains possible"
    assert 3 <= len(red_rows) <= g.shape[0] - 2, "generator should include several red markers"
    out, channel, analyses = _simulate(g)

    # Every red marker must be able to take the first step.
    for row in red_rows:
        assert bool(channel[row, 2]), "each red row must have channel access at column 2"
        assert int(g[row, 2]) == BG, "column 2 must be free for the first step"
        assert int(analyses[row]["run"]) >= 1

    finish_rows = [row for row in red_rows if analyses[row]["reason"] == "finish"]
    green_stop_rows = [row for row in red_rows if analyses[row]["reason"] == "green"]
    gap_rows = [
        row
        for row in red_rows
        if analyses[row]["reason"] == "gap" and bool(analyses[row]["later_channel"])
    ]
    assert finish_rows, "need a lane that reaches the far edge and turns blue"
    assert green_stop_rows, "need a lane blocked by a green marker"
    assert gap_rows, "need a lane blocked by a channel gap despite later channel cells"

    final_cols = {int(analyses[row]["final_col"]) for row in red_rows}
    assert len(final_cols) >= 3, "rows must not all collapse to the same stopping column"

    # The output must reflect both systems: without channels, motion would differ.
    shortcut = g.copy()
    for row in red_rows:
        pos = 1
        while pos + 1 < g.shape[1] and int(shortcut[row, pos + 1]) == BG:
            shortcut[row, pos] = BG
            pos += 1
            shortcut[row, pos] = RED
        if pos == g.shape[1] - 1:
            shortcut[row, pos] = BLUE
    assert not np.array_equal(shortcut, out), "channel gating must materially affect the result"

    # The output must also differ from the input.
    assert not np.array_equal(g, out), "markers must actually migrate"

    # Finished lanes become blue at the right edge; unfinished lanes stay red.
    for row in finish_rows:
        assert int(out[row, -1]) == BLUE
    for row in red_rows:
        if row not in finish_rows:
            final_col = int(analyses[row]["final_col"])
            assert int(out[row, final_col]) == RED

    # Green markers are preserved.
    for r, c in greens:
        assert int(out[r, c]) == GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Migrate each red edge marker rightward only along diagonal-ray channel cells, turning a completed lane into blue at the far edge.

    Input structure:
    - The background is black, and column 0 is a full blue vertical bar.
    - Some rows have a single red marker in column 1, with at most one red marker per row.
    - Two to four green marker pixels lie in the interior (never in columns 0 or 1).

    Rule:
    - Through every green marker, activate both full diagonals: the NW-SE line and the NE-SW line, all the way to the grid boundary.
    - Process red markers from top row to bottom row. A red marker advances one column at a time along its own row, but only if the next cell is both on an activated diagonal channel and currently black.
    - The marker stops before the first cell that is off-channel, occupied by a green marker, or outside the grid; if it reaches the last column, that final cell becomes blue instead of remaining red.
    - Green markers and the blue edge bar stay unchanged in the output.

    Key insight:
    - The green diagonals do not paint the result directly; they encode a sparse permission map, and the red markers can migrate only through the horizontally contiguous channel cells that map exposes.
    """
    g = np.asarray(grid, dtype=int)
    _validate_structure(g)
    out, _channel, _analyses = _simulate(g)
    return out


def _choose_greens(rng: np.random.Generator) -> tuple[int, int, list[tuple[int, int]]]:
    width = int(rng.choice(WIDTH_OPTIONS))
    deltas = list(LAYOUT_DELTAS[width])
    if rng.random() < 0.5:
        deltas = [(-dr, c) for dr, c in deltas]

    height = int(rng.integers(*HEIGHT_RANGE))
    min_delta = min(dr for dr, _ in deltas)
    max_delta = max(dr for dr, _ in deltas)
    finish_lo = -min_delta
    finish_hi = height - 1 - max_delta
    finish_row = int(rng.integers(finish_lo, finish_hi + 1))
    greens = [(finish_row + dr, c) for dr, c in deltas]
    return height, width, greens


def _candidate_rows(grid: np.ndarray) -> tuple[list[int], list[int], list[int], dict[int, dict[str, int | bool | str]]]:
    g = np.asarray(grid, dtype=int)
    greens = [(int(r), int(c)) for r, c in np.argwhere(g == GREEN)]
    channel = _channel_mask(g.shape, greens)
    analyses = {row: _analyze_row(g, channel, row) for row in range(g.shape[0])}
    red_candidates = [row for row, info in analyses.items() if int(info["run"]) >= 1]
    finish_rows = [row for row in red_candidates if analyses[row]["reason"] == "finish"]
    green_rows = [row for row in red_candidates if analyses[row]["reason"] == "green"]
    gap_rows = [
        row
        for row in red_candidates
        if analyses[row]["reason"] == "gap" and bool(analyses[row]["later_channel"])
    ]
    return finish_rows, green_rows, gap_rows, analyses


def _choose_red_rows(rng: np.random.Generator, grid: np.ndarray) -> list[int] | None:
    finish_rows, green_rows, gap_rows, analyses = _candidate_rows(grid)
    if not finish_rows or not green_rows or not gap_rows:
        return None

    finish_row = int(rng.choice(finish_rows))
    green_row = int(rng.choice(green_rows))
    green_final = int(analyses[green_row]["final_col"])

    usable_gap_rows = [
        row
        for row in gap_rows
        if row != green_row and int(analyses[row]["final_col"]) != green_final
    ]
    if not usable_gap_rows:
        return None

    gap_row = min(
        usable_gap_rows,
        key=lambda row: (int(analyses[row]["final_col"]), row),
    )

    selected = [finish_row, green_row, gap_row]
    used = set(selected)
    target_count = min(
        int(rng.integers(3, 6)),
        len([row for row, info in analyses.items() if int(info["run"]) >= 1]),
    )

    extra_pool = [
        row
        for row, info in analyses.items()
        if row not in used and int(info["run"]) >= 1
    ]
    rng.shuffle(extra_pool)
    for row in extra_pool:
        if len(selected) >= target_count:
            break
        selected.append(int(row))

    return sorted(selected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height, width, greens = _choose_greens(rng)
    grid = np.zeros((height, width), dtype=int)
    grid[:, 0] = BLUE
    for r, c in greens:
        grid[r, c] = GREEN

    red_rows = _choose_red_rows(rng, grid)
    if red_rows is None:
        return None
    for row in red_rows:
        grid[row, 1] = RED

    try:
        _semantic_checks(grid)
        solved = main(grid)
        oracle, _channel, _analyses = _simulate(grid)
    except Exception:
        return None

    if not np.array_equal(solved, oracle):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
