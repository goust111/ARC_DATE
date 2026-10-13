"""ARC task 7: infer each quadrant copy's transform from a one-pixel relic."""

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
TRANSFORM_COUNT = 4
MAX_GENERATION_ATTEMPTS = 500
_generation_index = 0


def _transforms(array: np.ndarray) -> tuple[np.ndarray, ...]:
    """Identity, left-right reflection, up-down reflection, and 180 rotation."""
    return (
        array.copy(),
        np.fliplr(array),
        np.flipud(array),
        np.rot90(array, 2),
    )


def _decode(grid: np.ndarray):
    """Validate and decode the quadrant motif and the three relics."""
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width and height % 2 == 0, "expected an even square grid"
    assert height <= 30, "ARC grids may not exceed 30x30"

    q = height // 2
    motif = g[:q, :q]
    motif_mask = motif != BACKGROUND
    motif_colors = set(map(int, np.unique(motif))) - {BACKGROUND}
    assert motif_mask.any(), "the upper-left quadrant must contain a motif"
    assert len(motif_colors) >= 2, "the source motif must be multicolored"

    masks = _transforms(motif_mask)
    assert len({mask.tobytes() for mask in masks}) == TRANSFORM_COUNT, (
        "the source motif must distinguish all four transforms"
    )

    decoded = []
    for quadrant_row, quadrant_col in ((0, 1), (1, 0), (1, 1)):
        r0, c0 = quadrant_row * q, quadrant_col * q
        quadrant = g[r0 : r0 + q, c0 : c0 + q]
        relic_cells = np.argwhere(quadrant != BACKGROUND)
        assert relic_cells.shape == (1, 2), (
            "each incomplete quadrant must contain exactly one relic"
        )
        relic_r, relic_c = map(int, relic_cells[0])
        relic_color = int(quadrant[relic_r, relic_c])
        assert relic_color not in motif_colors, (
            "relic colors must be absent from the source motif"
        )

        hits = [index for index, mask in enumerate(masks) if mask[relic_r, relic_c]]
        assert len(hits) == 1, "the relic position must identify one unique transform"
        decoded.append((r0, c0, relic_r, relic_c, relic_color, hits[0]))

    return g, motif, masks, decoded


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete three quadrants with transformed, relic-colored copies of the upper-left motif.

    Input structure:
    - The grid is an even square split into four equal square quadrants on a black background.
    - The upper-left quadrant contains a nonempty, asymmetric multicolor motif.
    - Each other quadrant contains exactly one colored relic pixel whose color is absent
      from the motif; its position is covered by exactly one of four candidate motif masks.

    Rule:
    - In each incomplete quadrant independently, align the upper-left quadrant to that
      quadrant and test identity, left-right reflection, up-down reflection, and 180-degree
      rotation in the quadrant's own coordinates.
    - Select the unique candidate whose foreground covers the relic position, then paint
      every foreground cell of that transformed copy with the relic's color.
    - Preserve the upper-left quadrant exactly.

    Key insight:
    - A relic has two independent roles: its quadrant-relative position indexes the
      transform, while its color paints the entire recovered copy.
    """
    g, _motif, masks, decoded = _decode(grid)
    out = g.copy()
    q = g.shape[0] // 2

    for r0, c0, _relic_r, _relic_c, relic_color, transform_index in decoded:
        quadrant_out = out[r0 : r0 + q, c0 : c0 + q]
        quadrant_out[:] = BACKGROUND
        quadrant_out[masks[transform_index]] = relic_color

    return out


def _random_connected_mask(q: int) -> np.ndarray:
    """Grow a sparse connected polyomino with deliberately asymmetric margins."""
    target_area = random.randint(max(6, q), min(q * 2 + 3, q * q // 2))
    start = (random.randrange(q), random.randrange(q))
    cells = {start}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < q and 0 <= nc < q and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)

    mask = np.zeros((q, q), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _exclusive_positions(masks: tuple[np.ndarray, ...]) -> list[np.ndarray]:
    coverage = sum(mask.astype(np.int8) for mask in masks)
    return [np.argwhere(mask & (coverage == 1)) for mask in masks]


def _paint_multicolor_motif(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    """Color a connected mask so every selected visible color occurs."""
    motif = np.zeros(mask.shape, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    for index, (r, c) in enumerate(coords):
        motif[r, c] = colors[index % len(colors)]
    return motif


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    chosen: list[int],
    relic_colors: list[int],
) -> bool:
    """Check the stated semantics, including key dependencies and non-degeneracy."""
    try:
        g, motif, masks, decoded = _decode(inp)
        actual = main(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    q = g.shape[0] // 2
    if not np.array_equal(actual, expected):
        return False
    if not np.array_equal(actual[:q, :q], motif):
        return False
    if len(set(chosen)) != 3 or len(set(relic_colors)) != 3:
        return False

    # Position must choose exactly the intended transform, and every alternative
    # transform must change the completed quadrant rather than being a shortcut.
    for item, intended in zip(decoded, chosen):
        r0, c0, relic_r, relic_c, color, found = item
        if found != intended:
            return False
        if sum(bool(mask[relic_r, relic_c]) for mask in masks) != 1:
            return False
        rendered = actual[r0 : r0 + q, c0 : c0 + q]
        if not np.array_equal(rendered != BACKGROUND, masks[intended]):
            return False
        if set(map(int, np.unique(rendered))) != {BACKGROUND, color}:
            return False
        if any(np.array_equal(masks[intended], masks[alternative])
               for alternative in range(TRANSFORM_COUNT) if alternative != intended):
            return False
    return True


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked input satisfying all uniqueness constraints."""
    global _generation_index

    # Omitting a different transform on successive calls guarantees that the first
    # two (and therefore first three) generated examples collectively show all four.
    omitted = _generation_index % TRANSFORM_COUNT
    _generation_index += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        q = random.randint(5, 10)
        mask = _random_connected_mask(q)
        masks = _transforms(mask)
        if len({candidate.tobytes() for candidate in masks}) != TRANSFORM_COUNT:
            continue
        exclusive = _exclusive_positions(masks)
        if any(len(positions) == 0 for positions in exclusive):
            continue

        palette = random.sample(range(1, 10), 6)
        motif_color_count = random.choice((2, 3))
        motif_colors = palette[:motif_color_count]
        relic_colors = palette[3:6]
        motif = _paint_multicolor_motif(mask, motif_colors)

        chosen = [index for index in range(TRANSFORM_COUNT) if index != omitted]
        random.shuffle(chosen)
        random.shuffle(relic_colors)

        inp = np.zeros((2 * q, 2 * q), dtype=int)
        expected = np.zeros_like(inp)
        inp[:q, :q] = motif
        expected[:q, :q] = motif

        quadrants = ((0, q), (q, 0), (q, q))
        for (r0, c0), transform_index, color in zip(quadrants, chosen, relic_colors):
            relic_r, relic_c = map(int, random.choice(exclusive[transform_index]))
            inp[r0 + relic_r, c0 + relic_c] = color
            target = expected[r0 : r0 + q, c0 : c0 + q]
            target[masks[transform_index]] = color

        if _semantic_checks(inp, expected, chosen, relic_colors):
            return inp

    raise RuntimeError("failed to generate a valid relic-indexed quadrant puzzle")
