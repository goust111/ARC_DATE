from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7

SEED_COLORS = (RED, GREEN, YELLOW, MAGENTA, 8, 9)
GEN_ATTEMPTS = 500


def _corner_scores(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    return {
        "tl": int(np.sum(grid[0, :] == ORANGE) + np.sum(grid[:, 0] == ORANGE) - (grid[0, 0] == ORANGE)),
        "tr": int(np.sum(grid[0, :] == ORANGE) + np.sum(grid[:, w - 1] == ORANGE) - (grid[0, w - 1] == ORANGE)),
        "bl": int(np.sum(grid[h - 1, :] == ORANGE) + np.sum(grid[:, 0] == ORANGE) - (grid[h - 1, 0] == ORANGE)),
        "br": int(np.sum(grid[h - 1, :] == ORANGE) + np.sum(grid[:, w - 1] == ORANGE) - (grid[h - 1, w - 1] == ORANGE)),
    }


def _orientation(grid: np.ndarray) -> str:
    scores = _corner_scores(grid)
    best = max(scores.values())
    winners = [name for name, score in scores.items() if score == best]
    assert len(winners) == 1 and best > 0, "expected one orange corner L"
    return winners[0]


def _normalize(grid: np.ndarray, orient: str) -> np.ndarray:
    if orient == "tl":
        return grid.copy()
    if orient == "tr":
        return np.fliplr(grid)
    if orient == "bl":
        return np.flipud(grid)
    if orient == "br":
        return np.flipud(np.fliplr(grid))
    raise ValueError(orient)


def _denormalize(grid: np.ndarray, orient: str) -> np.ndarray:
    return _normalize(grid, orient)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move an orange corner L inward, then expand interior seed colors into repeated down-right diagonals.
    """
    g = np.asarray(grid, dtype=int)
    orient = _orientation(g)
    norm = _normalize(g, orient)
    h, w = norm.shape
    assert h >= 5 and w >= 5, "grid must leave room for an inner diagonal field"

    seeds: list[tuple[int, int, int]] = []
    for r in range(2, h):
        for c in range(2, w):
            color = int(norm[r, c])
            if color not in (BLACK, BLUE, ORANGE):
                seeds.append((r, c, color))
    assert seeds, "expected at least one interior seed"

    out = norm.copy()
    out[:, :] = BLUE
    out[0, :] = BLUE
    out[:, 0] = BLUE
    out[1, 1:] = ORANGE
    out[1:, 1] = ORANGE
    out[2:, 2:] = BLUE

    for sr, sc, color in sorted(seeds, key=lambda item: (item[0], item[1])):
        phase = (sr - sc) % 3
        for r in range(2, h):
            for c in range(2, w):
                if (r - c) % 3 == phase:
                    out[r, c] = color

    return _denormalize(out, orient)


def _make_normalized(height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BLUE, dtype=int)
    grid[0, :] = ORANGE
    grid[:, 0] = ORANGE

    seed_count = random.randint(2, 3)
    colors = random.sample(SEED_COLORS, seed_count)
    phases = random.sample([0, 1, 2], seed_count)
    used: set[tuple[int, int]] = set()
    for color, phase in zip(colors, phases):
        candidates = [
            (r, c)
            for r in range(2, height)
            for c in range(2, width)
            if (r - c) % 3 == phase and (r, c) not in used
        ]
        r, c = random.choice(candidates)
        used.add((r, c))
        grid[r, c] = color
    return grid


def _apply_orientation(grid: np.ndarray, orient: str) -> np.ndarray:
    return _denormalize(grid, orient)


def _valid_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.ndim == 2 and max(out.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(8, 14)
        width = random.randint(8, 14)
        orient = random.choice(("tl", "tr", "bl", "br"))
        grid = _apply_orientation(_make_normalized(height, width), orient)
        if _valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate puzzle 030/7 input")

