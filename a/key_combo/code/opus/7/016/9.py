from __future__ import annotations

import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
BLACK = 0

GEN_ATTEMPTS = 500


def _valid_period(grid: np.ndarray, period: int) -> bool:
    h, w = grid.shape
    p = int(period)
    if p <= 0 or 2 * p > w:
        return False
    for r in range(h):
        for residue in range(p):
            seen = {
                int(grid[r, c])
                for c in range(residue, w, p)
                if int(grid[r, c]) != YELLOW
            }
            if len(seen) > 1:
                return False
    return True


def _infer_period(grid: np.ndarray) -> int:
    _, w = grid.shape
    for period in range(1, w):
        if _valid_period(grid, period):
            return int(period)
    raise AssertionError("failed to infer a valid horizontal period")


def _recover_tile(grid: np.ndarray, period: int) -> np.ndarray:
    h, _ = grid.shape
    tile = np.zeros((h, period), dtype=int)
    for r in range(h):
        for residue in range(period):
            values = [
                int(grid[r, c])
                for c in range(residue, grid.shape[1], period)
                if int(grid[r, c]) != YELLOW
            ]
            assert values, "every tile position must have at least one non-yellow observation"
            uniq = set(values)
            assert len(uniq) == 1, "period class must agree after ignoring yellow"
            tile[r, residue] = values[0]
    return tile


def _repeat_to_width(tile: np.ndarray, width: int) -> np.ndarray:
    h, period = tile.shape
    out = np.zeros((h, width), dtype=int)
    for c in range(width):
        out[:, c] = tile[:, c % period]
    return out


def _expected_output_from_tile(tile: np.ndarray, anomalies: list[tuple[int, int]]) -> np.ndarray:
    ideal = _repeat_to_width(tile, 9)
    out = np.where(ideal == BLUE, RED, BLACK).astype(int)
    for r, c in anomalies:
        for cc in range(out.shape[1]):
            if out[r, cc] == BLACK:
                out[r, cc] = GREEN
        for rr in range(out.shape[0]):
            if out[rr, c] == BLACK:
                out[rr, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a horizontal sprite period, complete it to width 9, and broadcast each error location as a green cross.

    Input structure:
    - The grid is a small black-background canvas with width less than 9.
    - Blue pixels form a horizontally periodic sprite pattern.
    - One or two yellow pixels replace the correct blue-or-black value inside one visible repeat fragment.

    Rule:
    - Ignore yellow pixels and infer the smallest horizontal translation period that is consistent with the remaining blue/black cells.
    - Reconstruct the ideal periodic pattern, extend it rightward to width 9, and repaint every blue cell red.
    - Treat the original yellow coordinates as anomaly sites, but verify that only those cells differ from the reconstructed ideal input.
    - For each anomaly site, color the whole row and whole column green only where the completed output is black; red cells stay red.

    Key insight:
    - The yellow pixels are not part of the period itself: once the clean period is reconstructed, the same reconstruction both repairs the corrupted cells and reveals where the green cross guides must be drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30
    assert w < 9, "input width must be smaller than 9"
    allowed = {BLACK, BLUE, YELLOW}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected input colors"

    anomalies = [tuple(map(int, pos)) for pos in np.argwhere(g == YELLOW)]
    assert 1 <= len(anomalies) <= 2, "expected one or two yellow anomalies"

    period = _infer_period(g)
    tile = _recover_tile(g, period)
    ideal_input = _repeat_to_width(tile, w)

    mismatch = ideal_input != g
    assert np.array_equal(mismatch, g == YELLOW), "only yellow cells may differ from the ideal pattern"

    return _expected_output_from_tile(tile, anomalies)


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = start[0] + dr, start[1] + dc
        if 0 <= rr < h and 0 <= cc < w:
            frontier.add((rr, cc))

    while len(cells) < area and frontier:
        nxt = random.choice(tuple(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = nxt[0] + dr, nxt[1] + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                frontier.add((rr, cc))

    mask = np.zeros((h, w), dtype=int)
    for r, c in cells:
        mask[r, c] = BLUE
    return mask


def _generate_tile(h: int, period: int, width: int) -> np.ndarray | None:
    max_area = h * period - period
    min_area = max(period + 1, 3)
    for _ in range(200):
        area = random.randint(min_area, max_area)
        tile = _random_connected_mask(h, period, area)

        column_counts = tile.sum(axis=0)
        if not np.all((column_counts >= 1) & (column_counts <= h - 1)):
            continue
        if not np.any((tile == BLUE).sum(axis=1) >= 2):
            continue
        if not np.any((tile == BLACK).sum(axis=1) >= 2):
            continue

        clean = _repeat_to_width(tile, width)
        if _infer_period(clean) != period:
            continue
        return tile
    return None


def _choose_anomalies(clean: np.ndarray, period: int) -> list[tuple[int, int]] | None:
    tail_cols = list(range(2 * period, clean.shape[1]))
    assert tail_cols, "generator must reserve a partial third repeat"

    blue_positions = []
    black_positions = []
    for r in range(clean.shape[0]):
        for c in tail_cols:
            pos = (r, c)
            if int(clean[r, c]) == BLUE:
                blue_positions.append(pos)
            else:
                black_positions.append(pos)

    if not blue_positions or not black_positions:
        return None

    count = random.choice([1, 2])
    positions: list[tuple[int, int]] = []

    if count == 1:
        source = blue_positions if random.random() < 0.5 else black_positions
        positions.append(random.choice(source))
        return positions

    branch = random.choice(["mixed", "blue", "black"])
    if branch == "mixed":
        positions = [random.choice(blue_positions), random.choice(black_positions)]
    elif branch == "blue":
        if len(blue_positions) < 2:
            return None
        positions = random.sample(blue_positions, 2)
    else:
        if len(black_positions) < 2:
            return None
        positions = random.sample(black_positions, 2)

    if len(set(positions)) != len(positions):
        return None
    return positions


def _sample_input() -> np.ndarray | None:
    h = random.randint(6, 8)
    period = random.choice([2, 3])
    width = random.choice([2 * period + 1, 2 * period + 2])
    if width >= 9:
        return None

    tile = _generate_tile(h, period, width)
    if tile is None:
        return None

    clean = _repeat_to_width(tile, width)
    anomalies = _choose_anomalies(clean, period)
    if anomalies is None:
        return None

    inp = clean.copy()
    for r, c in anomalies:
        inp[r, c] = YELLOW

    if np.any(inp[:, : 2 * period] == YELLOW):
        return None
    if _infer_period(inp) != period:
        return None

    out = main(inp)
    expected = _expected_output_from_tile(tile, anomalies)
    if not np.array_equal(out, expected):
        return None

    for r, c in anomalies:
        ideal = _repeat_to_width(tile, 9)
        if not (
            np.any(ideal[r, :] == BLACK) or np.any(ideal[:, c] == BLACK)
        ):
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _sample_input()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid input")
