from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(_ARC_ROOT) not in sys.path:
    sys.path.append(str(_ARC_ROOT))

try:
    from solvers.arc_utils import Color, scale_sprite
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        ALL_COLORS = [BLACK, BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]

    def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
        return np.kron(sprite, np.ones((int(factor), int(factor)), dtype=sprite.dtype))


MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(Color.ALL_COLORS)


def _infer_scale(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    side = int(g.shape[0])
    s = int(round(side ** 0.5))
    assert s * s == side and s in (3, 4), "side length must be 3^2 or 4^2"
    return s


def _extract_blocks(grid: np.ndarray, s: int) -> list[list[np.ndarray]]:
    return [
        [grid[br * s : (br + 1) * s, bc * s : (bc + 1) * s].copy() for bc in range(s)]
        for br in range(s)
    ]


def _unique_blocks(blocks: list[list[np.ndarray]]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for row in blocks:
        for block in row:
            if not any(np.array_equal(block, seen) for seen in unique):
                unique.append(block.copy())
    return unique


def _render_nested(occupancy: np.ndarray, pattern: np.ndarray, fg: int, bg: int) -> np.ndarray:
    s = int(pattern.shape[0])
    active_blocks = scale_sprite(occupancy.astype(int), s).astype(bool)
    tiled_pattern = np.tile(pattern.astype(bool), occupancy.shape)
    return np.where(active_blocks & tiled_pattern, int(fg), int(bg)).astype(int)


def _candidate_repairs(grid: np.ndarray) -> list[dict[str, np.ndarray | int | tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    s = _infer_scale(g)
    blocks = _extract_blocks(g, s)
    unique_blocks = _unique_blocks(blocks)
    colors = sorted(set(map(int, np.unique(g))))
    candidates: list[dict[str, np.ndarray | int | tuple[int, int]]] = []
    seen_repairs: set[tuple[bytes, tuple[int, int]]] = set()

    for bg in colors:
        bg_block = np.full((s, s), int(bg), dtype=int)
        for pattern_block in unique_blocks:
            fg_colors = sorted(set(map(int, np.unique(pattern_block))) - {int(bg)})
            if len(fg_colors) != 1:
                continue
            fg = int(fg_colors[0])
            pattern = (pattern_block == fg)

            occupancy = np.zeros((s, s), dtype=bool)
            valid_partition = True
            for br in range(s):
                for bc in range(s):
                    block = blocks[br][bc]
                    if np.array_equal(block, bg_block):
                        continue
                    if np.array_equal(block, pattern_block):
                        occupancy[br, bc] = True
                        continue
                    valid_partition = False
                    break
                if not valid_partition:
                    break
            if not valid_partition:
                continue

            mismatch = np.argwhere(occupancy != pattern)
            if mismatch.shape != (1, 2):
                continue

            repair = _render_nested(pattern, pattern, fg, bg)
            sig = (repair.tobytes(), repair.shape)
            if sig in seen_repairs:
                continue
            seen_repairs.add(sig)
            candidates.append(
                {
                    "repair": repair,
                    "pattern": pattern.astype(bool),
                    "occupancy": occupancy.astype(bool),
                    "bg": bg,
                    "fg": fg,
                    "error": (int(mismatch[0, 0]), int(mismatch[0, 1])),
                }
            )

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the unique block flip in a self-nested binary pattern.

    Input structure:
    - The grid is a square of side s^2 with s in {3,4}, using exactly one
      background color and one foreground color.
    - It splits evenly into an s x s array of s x s blocks.
    - Every non-empty block is the same s x s binary pattern, while empty blocks
      are solid background.

    Rule:
    - Read the shared non-empty block content as the fine-scale truth pattern.
    - Build the coarse s x s occupancy mask telling which blocks are non-empty.
    - The correct scene satisfies coarse occupancy == fine pattern, but the
      input has exactly one whole block flipped relative to that identity.
    - If that coarse cell should be on, repaint the entire block with the shared
      pattern; if it should be off, clear the whole block to background.

    Key insight:
    - No block is locally distinguishable as wrong; the error is exposed only by
      comparing the coarse block layout against the repeated inner pattern.
    """
    g = np.asarray(grid, dtype=int)
    candidates = _candidate_repairs(g)
    assert len(candidates) == 1, f"expected one unique repair, found {len(candidates)}"
    return np.asarray(candidates[0]["repair"], dtype=int)


def _sample_pattern(rng: np.random.Generator, s: int) -> np.ndarray:
    cell_count = s * s
    while True:
        on_count = int(rng.integers(2, cell_count - 1))
        flat = np.zeros(cell_count, dtype=bool)
        chosen = rng.choice(cell_count, size=on_count, replace=False)
        flat[chosen] = True
        pattern = flat.reshape(s, s)

        if 0 < pattern.sum() < cell_count:
            return pattern


def _sample_case(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    s = int(rng.choice([3, 4]))
    bg, fg = map(int, rng.choice(PALETTE, size=2, replace=False))
    pattern = _sample_pattern(rng, s)
    occupancy = pattern.copy()

    flip_on = bool(rng.integers(0, 2))
    if flip_on:
        choices = np.argwhere(~pattern)
    else:
        choices = np.argwhere(pattern)
    error_r, error_c = map(int, choices[int(rng.integers(len(choices)))])
    occupancy[error_r, error_c] = ~occupancy[error_r, error_c]

    inp = _render_nested(occupancy, pattern, fg, bg)
    out = _render_nested(pattern, pattern, fg, bg)
    return inp, out, pattern, occupancy, fg, bg


def _validate_case(
    inp: np.ndarray,
    out: np.ndarray,
    pattern: np.ndarray,
    occupancy: np.ndarray,
    fg: int,
    bg: int,
) -> None:
    s = int(pattern.shape[0])
    assert inp.shape == out.shape == (s * s, s * s)
    assert inp.ndim == out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert np.count_nonzero(occupancy != pattern) == 1
    assert occupancy.any(), "need at least one observed non-empty block"
    assert (~occupancy).any(), "need at least one observed empty block"

    pattern_block = np.where(pattern, fg, bg).astype(int)
    blocks = _extract_blocks(inp, s)
    for br in range(s):
        for bc in range(s):
            block = blocks[br][bc]
            if occupancy[br, bc]:
                assert np.array_equal(block, pattern_block)
            else:
                assert np.all(block == bg)

    candidates = _candidate_repairs(inp)
    assert len(candidates) == 1, "repair must be uniquely determined"
    assert np.array_equal(candidates[0]["repair"], out)
    assert np.array_equal(main(inp), out)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, out, pattern, occupancy, fg, bg = _sample_case(rng)
        try:
            _validate_case(inp, out, pattern, occupancy, fg, bg)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid self-nested block-flip input")
