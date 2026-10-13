from __future__ import annotations

import itertools
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


def _load_arc_utils():
    try:
        from arc_utils import Color, get_bg  # type: ignore
        return Color, get_bg
    except ModuleNotFoundError:
        candidates = [
            Path.cwd() / "solvers",
            Path("/home/lixinyu/snap/new/solvers"),
        ]
        for candidate in candidates:
            if candidate.exists():
                text = str(candidate)
                if text not in sys.path:
                    sys.path.append(text)
        from arc_utils import Color, get_bg  # type: ignore
        return Color, get_bg


Color, get_bg = _load_arc_utils()


GREY = int(Color.GREY)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
BLACK = int(Color.BLACK)

_GEN_RETRIES = 400
_ROW_RANGE = (7, 11)
_COL_RANGE = (8, 13)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each column's gray pixel sequence by parity, recolor it, and compress the two parity groups toward opposite edges.

    Input structure:
    - The grid is a black background containing only scattered gray foreground pixels.
    - Each column is an independent channel with 2 to 4 gray pixels.
    - Within one column, gray pixels are separated by at least one empty row.

    Rule:
    - Process each column independently and enumerate its gray pixels from top to bottom as 0, 1, 2, ...
    - Even-indexed pixels become green and are compressed to the bottom edge of that column, occupying consecutive bottom rows while preserving their top-to-bottom order.
    - Odd-indexed pixels become red and are compressed to the top edge of that column, occupying consecutive top rows while preserving their top-to-bottom order.
    - Every other cell becomes or remains black.

    Key insight:
    - A single hidden signal, the top-to-bottom enumeration parity inside each column, determines both the recolor mapping and the compression direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BLACK
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}

    h, w = g.shape
    out = np.full((h, w), BLACK, dtype=int)

    for c in range(w):
        rows = np.flatnonzero(g[:, c] == GREY)
        if rows.size == 0:
            continue

        even_rows = rows[::2]
        odd_rows = rows[1::2]

        if odd_rows.size:
            out[np.arange(odd_rows.size), c] = RED
        if even_rows.size:
            out[h - even_rows.size : h, c] = GREEN

    return out


@lru_cache(maxsize=None)
def _feasible_row_tuples(h: int, count: int) -> tuple[tuple[int, ...], ...]:
    choices = []
    for rows in itertools.combinations(range(h), count):
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            choices.append(tuple(int(x) for x in rows))
    return tuple(choices)


def _choose_rows(h: int, count: int, profile: str, rng: np.random.Generator) -> tuple[int, ...]:
    combos = _feasible_row_tuples(h, count)
    assert combos

    center = (h - 1) / 2.0
    scores = []
    for rows in combos:
        row_sum = float(sum(rows))
        spread = float(rows[-1] - rows[0])
        middle_dist = float(sum(abs(r - center) for r in rows))
        if profile == "top":
            score = 1.0 + (h * count - row_sum)
        elif profile == "bottom":
            score = 1.0 + row_sum
        elif profile == "spread":
            score = 1.0 + 3.0 * spread + 0.25 * middle_dist
        elif profile == "middle":
            score = 1.0 + 2.0 * (h * count - middle_dist)
        else:
            score = 1.0
        scores.append(score)

    weights = np.asarray(scores, dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(len(combos), p=weights))
    return combos[idx]


def _profile_specs(width: int, rng: np.random.Generator) -> list[tuple[int, str]]:
    base = [
        (2, "top"),
        (3, "bottom"),
        (4, "spread"),
        (2, "bottom"),
        (3, "top"),
        (4, "middle"),
    ]
    extras = []
    while len(base) + len(extras) < width:
        extras.append(
            (
                int(rng.choice([2, 3, 4])),
                str(rng.choice(["top", "bottom", "spread", "middle"])),
            )
        )
    specs = base + extras
    rng.shuffle(specs)
    return specs[:width]


def _reverse_parity_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros_like(g)
    for c in range(w):
        rows = np.flatnonzero(g[:, c] == GREY)
        red_count = (rows.size + 1) // 2
        green_count = rows.size // 2
        if red_count:
            out[:red_count, c] = RED
        if green_count:
            out[h - green_count :, c] = GREEN
    return out


def _half_split_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros_like(g)
    split = h / 2.0
    for c in range(w):
        rows = np.flatnonzero(g[:, c] == GREY)
        top_group = [r for r in rows if r < split]
        bottom_group = [r for r in rows if r >= split]
        if top_group:
            out[: len(top_group), c] = RED
        if bottom_group:
            out[h - len(bottom_group) :, c] = GREEN
    return out


def _validate_structure(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}

    h, w = g.shape
    counts = []
    top_heavy = False
    bottom_heavy = False
    widely_spread = False

    for c in range(w):
        rows = np.flatnonzero(g[:, c] == GREY)
        assert 2 <= rows.size <= 4
        if rows.size > 1:
            assert bool(np.all(np.diff(rows) >= 2))
        counts.append(int(rows.size))
        top_heavy |= float(np.mean(rows)) <= (h - 1) / 2.0 - 0.9
        bottom_heavy |= float(np.mean(rows)) >= (h - 1) / 2.0 + 0.9
        widely_spread |= (rows[-1] - rows[0]) >= max(4, h // 2)

    assert {2, 3, 4}.issubset(set(counts))
    assert top_heavy and bottom_heavy and widely_spread


def _validate_semantics(grid: np.ndarray) -> None:
    _validate_structure(grid)
    out = main(grid)
    h, w = grid.shape

    assert set(map(int, np.unique(out))) <= {BLACK, RED, GREEN}
    assert int(np.count_nonzero(out == RED)) > 0
    assert int(np.count_nonzero(out == GREEN)) > 0

    for c in range(w):
        in_rows = np.flatnonzero(grid[:, c] == GREY)
        red_rows = np.flatnonzero(out[:, c] == RED)
        green_rows = np.flatnonzero(out[:, c] == GREEN)

        assert red_rows.size == in_rows.size // 2
        assert green_rows.size == (in_rows.size + 1) // 2
        assert np.array_equal(red_rows, np.arange(red_rows.size))
        assert np.array_equal(green_rows, np.arange(h - green_rows.size, h))

    assert not np.array_equal(out, _reverse_parity_output(grid))
    assert not np.array_equal(out, _half_split_output(grid))

def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*_ROW_RANGE))
    w = int(rng.integers(*_COL_RANGE))
    grid = np.zeros((h, w), dtype=int)

    for c, (count, profile) in enumerate(_profile_specs(w, rng)):
        rows = _choose_rows(h, count, profile, rng)
        grid[list(rows), c] = GREY

    try:
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-compression sample")
