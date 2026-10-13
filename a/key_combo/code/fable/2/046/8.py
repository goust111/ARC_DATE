from __future__ import annotations

import random

import numpy as np


SEPARATOR_COLOR = 2
PANEL_H_MAX = 12
PANEL_W_MAX = 12
PERIOD_MIN = 2
PERIOD_MAX = 5
COLOR_POOL = [1, 3, 4, 5, 6, 7, 8, 9]
GENERATION_ATTEMPTS = 400


def _split_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR_COLOR)]
    assert len(separator_cols) == 1, "expected one solid red separator column"
    sep = int(separator_cols[0])

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.size and right.size and left.shape == right.shape, "separator must split two equal panels"
    assert not np.any(left == SEPARATOR_COLOR) and not np.any(right == SEPARATOR_COLOR), (
        "separator color must be unique to the divider"
    )
    return left, right


def _expand_tile(tile: np.ndarray, shape: tuple[int, int], shift: tuple[int, int]) -> np.ndarray:
    p, q = tile.shape
    h, w = shape
    dr, dc = shift
    rows = (np.arange(h) + int(dr)) % p
    cols = (np.arange(w) + int(dc)) % q
    return tile[rows[:, None], cols[None, :]]


def _candidate_tile_from_views(
    left: np.ndarray,
    right: np.ndarray,
    p: int,
    q: int,
    dr: int,
    dc: int,
) -> np.ndarray | None:
    tile = np.full((p, q), -1, dtype=int)

    for panel, row_shift, col_shift in ((left, 0, 0), (right, dr, dc)):
        coords = np.argwhere(panel != 0)
        for r, c in coords:
            rr = (int(r) + int(row_shift)) % p
            cc = (int(c) + int(col_shift)) % q
            value = int(panel[r, c])
            current = int(tile[rr, cc])
            if current == -1:
                tile[rr, cc] = value
            elif current != value:
                return None

    if np.any(tile < 0):
        return None
    return tile


def _joint_candidates(left: np.ndarray, right: np.ndarray) -> list[dict]:
    h, w = left.shape
    visible_budget = int(np.count_nonzero(left) + np.count_nonzero(right))
    candidates: list[dict] = []

    for p in range(1, h + 1):
        for q in range(1, w + 1):
            if p * q > visible_budget:
                continue
            for dr in range(p):
                for dc in range(q):
                    tile = _candidate_tile_from_views(left, right, p, q, dr, dc)
                    if tile is None:
                        continue
                    candidates.append(
                        {
                            "p": int(p),
                            "q": int(q),
                            "dr": int(dr),
                            "dc": int(dc),
                            "tile": tile,
                        }
                    )
    return candidates


def _panel_complete_candidates(panel: np.ndarray) -> list[dict]:
    h, w = panel.shape
    visible_budget = int(np.count_nonzero(panel))
    candidates: list[dict] = []

    for p in range(1, h + 1):
        for q in range(1, w + 1):
            if p * q > visible_budget:
                continue
            tile = _candidate_tile_from_views(panel, np.zeros_like(panel), p, q, 0, 0)
            if tile is None:
                continue
            candidates.append({"p": int(p), "q": int(q), "tile": tile})
    return candidates


def _minimal_candidates(candidates: list[dict]) -> list[dict]:
    minimal: list[dict] = []
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other is cand:
                continue
            if other["p"] <= cand["p"] and other["q"] <= cand["q"] and (
                other["p"] < cand["p"] or other["q"] < cand["q"]
            ):
                dominated = True
                break
        if not dominated:
            minimal.append(cand)
    return minimal


def _solve_views(left: np.ndarray, right: np.ndarray) -> dict:
    candidates = _joint_candidates(left, right)
    assert candidates, "no periodic alignment explains both panels"

    minimal = _minimal_candidates(candidates)
    assert len(minimal) == 1, "expected a unique minimal period and phase offset"
    return minimal[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the left panel by phase-aligning two holey views of one periodic tile.

    Input structure:
    - A solid red separator column splits the input into equal-size left and right panels.
    - Each panel is a crop of the same rectangular periodic color table, but the two
      panels use different phase offsets and both contain black 0 holes.
    - Nonzero cells are trustworthy observations of the hidden periodic tile.

    Rule:
    - Use the visible cells to find the unique smallest tile size `(p, q)` that can
      explain both panels together.
    - Find the unique right-panel phase offset `(dr, dc)` relative to the left-panel
      residue system so that every visible cell in both panels agrees on one `p x q`
      tile.
    - Reconstruct that tile from the aligned evidence and fill every black hole in the
      left panel according to the left panel's own phase.

    Key insight:
    - The two panels cannot be merged at the same coordinates because they are
      mis-phased views of the same repeating tile; only after residue-level alignment do
      their complementary clues reveal the missing tile entries.
    """
    left, right = _split_input(np.asarray(grid, dtype=int))
    solved = _solve_views(left, right)
    full_left = _expand_tile(solved["tile"], left.shape, (0, 0))

    out = left.copy()
    holes = out == 0
    out[holes] = full_left[holes]
    assert np.all((left == 0) | (left == out)), "visible left cells must stay unchanged"
    return out


def _row_period(tile: np.ndarray) -> int:
    h = tile.shape[0]
    for p in range(1, h + 1):
        if np.array_equal(tile, tile[np.arange(h) % p, :]):
            return int(p)
    raise AssertionError("row period search failed")


def _col_period(tile: np.ndarray) -> int:
    w = tile.shape[1]
    for q in range(1, w + 1):
        if np.array_equal(tile, tile[:, np.arange(w) % q]):
            return int(q)
    raise AssertionError("column period search failed")


def _random_tile(p: int, q: int) -> np.ndarray:
    for _ in range(200):
        color_count = random.randint(3, min(len(COLOR_POOL), max(3, p * q)))
        palette = random.sample(COLOR_POOL, color_count)
        tile = np.array([[random.choice(palette) for _ in range(q)] for _ in range(p)], dtype=int)
        if _row_period(tile) == p and _col_period(tile) == q and len(set(map(int, tile.ravel()))) >= 3:
            return tile
    raise AssertionError("failed to sample a tile with the requested minimal period")


def _sample_panel_dim(period: int, limit: int) -> int:
    lo = max(period + 1, 2 * period - 1)
    hi = min(limit, 3 * period + 1)
    assert lo <= hi, "panel size bounds must be feasible"
    return random.randint(lo, hi)


def _sample_shift(p: int, q: int) -> tuple[int, int]:
    mode = random.choice(["row", "col", "both"])
    dr = random.randrange(1, p) if mode in {"row", "both"} else 0
    dc = random.randrange(1, q) if mode in {"col", "both"} else 0
    return int(dr), int(dc)


def _sample_hidden_residues(p: int, q: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    residues = [(r, c) for r in range(p) for c in range(q)]
    random.shuffle(residues)
    count_cap = max(1, len(residues) // 3)
    left_count = random.randint(1, count_cap)
    right_count = random.randint(1, count_cap)
    while left_count + right_count >= len(residues):
        if left_count >= right_count:
            left_count -= 1
        else:
            right_count -= 1
    left_hidden = set(residues[:left_count])
    right_hidden = set(residues[left_count : left_count + right_count])
    assert left_hidden and right_hidden and left_hidden.isdisjoint(right_hidden)
    return left_hidden, right_hidden


def _panel_with_hidden_residues(
    tile: np.ndarray,
    shape: tuple[int, int],
    shift: tuple[int, int],
    hidden: set[tuple[int, int]],
) -> np.ndarray:
    full = _expand_tile(tile, shape, shift)
    p, q = tile.shape
    rows = (np.arange(shape[0]) + int(shift[0])) % p
    cols = (np.arange(shape[1]) + int(shift[1])) % q
    hidden_mask = np.zeros(shape, dtype=bool)
    for rr, cc in hidden:
        hidden_mask |= (rows[:, None] == int(rr)) & (cols[None, :] == int(cc))
    out = full.copy()
    out[hidden_mask] = 0
    return out


def _build_input(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h, w = left.shape
    grid = np.empty((h, 2 * w + 1), dtype=int)
    grid[:, :w] = left
    grid[:, w] = SEPARATOR_COLOR
    grid[:, w + 1 :] = right
    return grid


def _naive_pixelwise_fill(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    naive = left.copy()
    take = (naive == 0) & (right != 0)
    naive[take] = right[take]
    return naive


def _validate_instance(
    tile: np.ndarray,
    shift: tuple[int, int],
    left_hidden: set[tuple[int, int]],
    right_hidden: set[tuple[int, int]],
    left: np.ndarray,
    right: np.ndarray,
) -> bool:
    try:
        solved = _solve_views(left, right)
    except AssertionError:
        return False
    p, q = tile.shape
    dr, dc = shift

    if (solved["p"], solved["q"], solved["dr"], solved["dc"]) != (p, q, dr, dc):
        return False
    if not np.array_equal(solved["tile"], tile):
        return False

    if _panel_complete_candidates(left) or _panel_complete_candidates(right):
        return False

    expected = _expand_tile(tile, left.shape, (0, 0))
    naive = _naive_pixelwise_fill(left, right)
    overlap = (left != 0) & (right != 0)

    if np.array_equal(naive, expected):
        return False
    if not np.any(overlap & (left != right)):
        return False

    left_missing = expected[left == 0]
    if left_missing.size == 0:
        return False
    if not np.any(left == 0):
        return False

    for residue in left_hidden:
        if residue not in right_hidden:
            break
    else:
        return False
    for residue in right_hidden:
        if residue not in left_hidden:
            break
    else:
        return False

    inp = _build_input(left, right)
    try:
        return np.array_equal(main(inp), expected)
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    p = random.randint(PERIOD_MIN, PERIOD_MAX)
    q = random.randint(PERIOD_MIN, PERIOD_MAX)
    tile = _random_tile(p, q)
    shape = (_sample_panel_dim(p, PANEL_H_MAX), _sample_panel_dim(q, PANEL_W_MAX))
    shift = _sample_shift(p, q)
    left_hidden, right_hidden = _sample_hidden_residues(p, q)

    left = _panel_with_hidden_residues(tile, shape, (0, 0), left_hidden)
    right = _panel_with_hidden_residues(tile, shape, shift, right_hidden)

    if not _validate_instance(tile, shift, left_hidden, right_hidden, left, right):
        return None
    return _build_input(left, right)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid dual-view periodic puzzle")
