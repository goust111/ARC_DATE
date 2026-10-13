from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BORDER = 1
INTERIOR = 2
SPRITE_SIZE = 5
INNER_SIZE = 3
OUTPUT_SIZE = INNER_SIZE * INNER_SIZE
GEN_ATTEMPTS = 1000


def _border_values(patch: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            patch[0, :],
            patch[-1, :],
            patch[1:-1, 0],
            patch[1:-1, -1],
        )
    )


def _find_framed_sprites(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    sprites: list[tuple[int, np.ndarray]] = []
    h, w = grid.shape
    for r in range(h - SPRITE_SIZE + 1):
        for c in range(w - SPRITE_SIZE + 1):
            patch = grid[r : r + SPRITE_SIZE, c : c + SPRITE_SIZE]
            border = _border_values(patch)
            border_color = int(border[0])
            if border_color == BG or np.any(border != border_color):
                continue
            interior = patch[1:-1, 1:-1]
            fg_colors = {int(v) for v in np.unique(interior) if v != BG}
            if len(fg_colors) != 1 or border_color in fg_colors:
                continue
            sprites.append((border_color, (interior != BG).copy()))
    return sprites


def _kron_recolor(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(np.kron(mask.astype(int), mask.astype(int)) == 1, color, BG).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """Select the only unique framed sprite interior, then draw its self-Kronecker mask in the frame color.

    Input structure:
    - Black background with 4-5 separated 5x5 framed sprites.
    - Every sprite has a one-cell non-black border and a 3x3 interior made of one foreground color plus black.
    - Exactly one interior pattern occurs once; all others share the same repeated 3x3 pattern.

    Rule:
    - Detect each 5x5 framed sprite and extract its 3x3 interior.
    - Compare interiors as binary masks where non-black cells are foreground.
    - Pick the single mask that appears exactly once.
    - Output the mask's Kronecker self-product, recolored with that sprite's border color, on black.

    Key insight:
    - The interior color only defines the binary mask; the enlarged output must be painted with the border color instead.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sprites = _find_framed_sprites(g)
    assert len(sprites) >= 4, "expected at least four framed sprites"

    keys = [mask.tobytes() for _, mask in sprites]
    counts = Counter(keys)
    unique_keys = [key for key, count in counts.items() if count == 1]
    assert len(unique_keys) == 1, "expected exactly one unique interior pattern"

    for border_color, mask in sprites:
        if mask.tobytes() == unique_keys[0]:
            return _kron_recolor(mask, border_color)
    raise AssertionError("unique sprite was not found")


def _sample_mask(min_fg: int, max_fg: int) -> np.ndarray:
    fg_count = random.randint(min_fg, max_fg)
    mask = np.zeros(INNER_SIZE * INNER_SIZE, dtype=bool)
    indices = random.sample(range(mask.size), fg_count)
    mask[indices] = True
    return mask.reshape((INNER_SIZE, INNER_SIZE))


def _sample_distinct_masks() -> tuple[np.ndarray, np.ndarray]:
    repeated = _sample_mask(3, 6)
    for _ in range(200):
        unique = _sample_mask(3, 6)
        if np.count_nonzero(repeated != unique) >= 2:
            return repeated, unique
    raise RuntimeError("failed to sample distinct masks")


def _can_place(top_left: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = top_left
    for pr, pc in placed:
        separated_rows = r + SPRITE_SIZE + 1 <= pr or pr + SPRITE_SIZE + 1 <= r
        separated_cols = c + SPRITE_SIZE + 1 <= pc or pc + SPRITE_SIZE + 1 <= c
        if not (separated_rows or separated_cols):
            return False
    return True


def _place_sprites(
    shape: tuple[int, int],
    count: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    candidates = [(r, c) for r in range(h - SPRITE_SIZE + 1) for c in range(w - SPRITE_SIZE + 1)]
    random.shuffle(candidates)
    placed: list[tuple[int, int]] = []
    for candidate in candidates:
        if _can_place(candidate, placed):
            placed.append(candidate)
            if len(placed) == count:
                return placed
    return None


def _draw_sprite(mask: np.ndarray) -> np.ndarray:
    sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    sprite[0, :] = BORDER
    sprite[-1, :] = BORDER
    sprite[:, 0] = BORDER
    sprite[:, -1] = BORDER
    sprite[1:-1, 1:-1][mask] = INTERIOR
    return sprite


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_ATTEMPTS):
        count = random.randint(4, 5)
        height = random.randint(16, 22)
        width = random.randint(16, 22)
        positions = _place_sprites((height, width), count)
        if positions is None:
            continue

        repeated_mask, unique_mask = _sample_distinct_masks()
        grid = np.zeros((height, width), dtype=int)
        unique_index = random.randrange(count)
        repeated_sprite = _draw_sprite(repeated_mask)
        unique_sprite = _draw_sprite(unique_mask)

        for idx, (r, c) in enumerate(positions):
            sprite = unique_sprite if idx == unique_index else repeated_sprite
            grid[r : r + SPRITE_SIZE, c : c + SPRITE_SIZE] = sprite

        expected = _kron_recolor(unique_mask, BORDER)
        found = _find_framed_sprites(grid)
        if len(found) != count:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid, expected
    raise RuntimeError("failed to generate a valid input")


def generate_input() -> np.ndarray:
    return _generate_case()[0]


def _variant_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(grid.shape), grid.tobytes()


def validate_random_cases(trials: int = 100) -> dict[str, int]:
    variants: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(trials):
        grid, expected = _generate_case()
        assert max(grid.shape) <= 30, "input size exceeds limit"
        assert max(expected.shape) <= 30, "output size exceeds limit"
        actual = main(grid)
        assert actual.shape == (OUTPUT_SIZE, OUTPUT_SIZE), "output must be 9x9"
        assert np.array_equal(actual, expected), "solver mismatch on generated case"
        variants.add(_variant_signature(grid))

    assert len(variants) >= 20, "insufficient input diversity"
    return {"trials": trials, "variants": len(variants), "max_input_dim": 30, "max_output_dim": OUTPUT_SIZE}

