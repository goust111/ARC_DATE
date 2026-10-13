from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
COMMAND_COLORS = (BLUE, RED, GREEN)
INPUT_SIZE = 4
MIN_FOREGROUND = 5
MAX_FOREGROUND = 11
GENERATION_ATTEMPTS = 300


def _pose(seed: np.ndarray, color: int) -> np.ndarray:
    if color == BLUE:
        return seed.copy()
    if color == RED:
        return np.fliplr(seed).copy()
    if color == GREEN:
        return np.rot90(seed, 2).copy()
    raise AssertionError("foreground colors must be blue, red, or green")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the seed grid into block copies whose pose is commanded by the seed colors.

    Input structure:
    - The input is a small black-background grid whose foreground pixels use only blue, red, and green.
    - Every foreground pixel belongs to the seed itself; there are no separate legend objects or extra decorations.

    Rule:
    - Treat the whole input grid as the seed sprite to be stamped.
    - For each input cell, reserve one output block the same size as the full seed.
    - Leave black cells' blocks empty; stamp blue cells with the original seed, red cells with the left-right mirror of the seed, and green cells with the 180-degree rotation of the seed.

    Key insight:
    - The seed is self-referential: each colored pixel simultaneously marks where to stamp and selects which pose of the complete seed should appear there.
    """
    seed = np.asarray(grid, dtype=int)
    assert seed.ndim == 2, "expected a 2D grid"

    allowed = {BLACK, BLUE, RED, GREEN}
    colors = set(map(int, np.unique(seed)))
    assert colors <= allowed, "input may only use black, blue, red, and green"

    h, w = map(int, seed.shape)
    out = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            color = int(seed[r, c])
            if color == BLACK:
                continue
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = _pose(seed, color)
    return out


def _covers_all_commands(seed: np.ndarray) -> bool:
    return all(np.any(seed == color) for color in COMMAND_COLORS)


def _uses_enough_space(seed: np.ndarray) -> bool:
    fg = seed != BLACK
    return (
        MIN_FOREGROUND <= int(np.count_nonzero(fg)) <= MAX_FOREGROUND
        and int(np.count_nonzero(np.any(fg, axis=0))) >= 3
        and int(np.count_nonzero(np.any(fg, axis=1))) >= 3
    )


def _poses_are_distinct(seed: np.ndarray) -> bool:
    original = _pose(seed, BLUE)
    mirrored = _pose(seed, RED)
    rotated = _pose(seed, GREEN)
    return (
        not np.array_equal(original, mirrored)
        and not np.array_equal(original, rotated)
        and not np.array_equal(mirrored, rotated)
    )


def _gen_attempt() -> np.ndarray | None:
    seed = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=int)
    foreground_count = random.randint(MIN_FOREGROUND, MAX_FOREGROUND)
    cells = [(r, c) for r in range(INPUT_SIZE) for c in range(INPUT_SIZE)]
    chosen = random.sample(cells, foreground_count)
    random.shuffle(chosen)

    for color, (r, c) in zip(COMMAND_COLORS, chosen[: len(COMMAND_COLORS)]):
        seed[r, c] = color
    for r, c in chosen[len(COMMAND_COLORS) :]:
        seed[r, c] = random.choice(COMMAND_COLORS)

    if not _covers_all_commands(seed):
        return None
    if not _uses_enough_space(seed):
        return None
    if not _poses_are_distinct(seed):
        return None

    out = main(seed)
    if out.shape != (INPUT_SIZE * INPUT_SIZE, INPUT_SIZE * INPUT_SIZE):
        return None
    return seed


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        seed = _gen_attempt()
        if seed is not None:
            return seed
    raise RuntimeError("failed to generate a valid self-stamping seed")
