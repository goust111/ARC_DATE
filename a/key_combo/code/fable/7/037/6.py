from __future__ import annotations

import random

import numpy as np


def _variants(tile: np.ndarray) -> tuple[np.ndarray, ...]:
    """The four shape-preserving mirror postures, in a fixed local frame."""
    return tile, np.fliplr(tile), np.flipud(tile), np.flipud(np.fliplr(tile))


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    winners = colors[counts == counts.max()]
    assert len(winners) == 1, "background color must be uniquely dominant"
    return int(winners[0])


def _infer_missing_posture(labels: list[int | None]) -> tuple[int, int] | None:
    """Return (shortest period, missing label), or None if it is not unique."""
    n = len(labels)
    missing = [i for i, value in enumerate(labels) if value is None]
    if len(missing) != 1:
        return None
    hole = missing[0]

    valid: list[tuple[int, int]] = []
    # A cycle must repeat at least twice across the observed strip.
    for period in range(1, n // 2 + 1):
        residue_values: list[int] = []
        ok = True
        for residue in range(period):
            seen = {
                int(labels[i])
                for i in range(residue, n, period)
                if labels[i] is not None
            }
            if len(seen) != 1:
                ok = False
                break
            residue_values.append(next(iter(seen)))
        if ok:
            valid.append((period, residue_values[hole % period]))

    if not valid:
        return None
    shortest = min(period for period, _ in valid)
    answers = {(period, value) for period, value in valid if period == shortest}
    return next(iter(answers)) if len(answers) == 1 else None


def _candidate_repairs(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, np.ndarray]]:
    """Enumerate unambiguous (block width, hole, period, repaired grid) readings."""
    h, total_w = grid.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []

    for block_w in range(1, total_w + 1):
        if total_w % block_w:
            continue
        count = total_w // block_w
        if count < 5:
            continue

        blocks = [grid[:, i * block_w : (i + 1) * block_w] for i in range(count)]
        blank = [i for i, block in enumerate(blocks) if np.all(block == bg)]
        if len(blank) != 1:
            continue
        hole = blank[0]

        base = next((block for block in blocks if not np.all(block == bg)), None)
        if base is None:
            continue
        postures = _variants(base)
        # Without four visibly different postures, the orientation is not readable.
        if any(np.array_equal(postures[i], postures[j]) for i in range(4) for j in range(i)):
            continue

        labels: list[int | None] = []
        valid = True
        for i, block in enumerate(blocks):
            if i == hole:
                labels.append(None)
                continue
            matches = [k for k, posture in enumerate(postures) if np.array_equal(block, posture)]
            if len(matches) != 1:
                valid = False
                break
            labels.append(matches[0])
        if not valid:
            continue

        inferred = _infer_missing_posture(labels)
        if inferred is None:
            continue
        period, posture = inferred
        repaired = grid.copy()
        repaired[:, hole * block_w : (hole + 1) * block_w] = postures[posture]
        candidates.append((block_w, hole, period, repaired))

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the one erased tile by extrapolating a horizontal cycle of mirror postures.

    Input structure:
    - A uniquely dominant background underlies equal-width horizontal tile blocks.
    - Exactly one whole block is background; every other block is one asymmetric
      base tile in its original, left-right mirrored, top-bottom mirrored, or
      180-degree posture.
    - The block postures follow a uniquely inferable short repeating cycle.

    Rule:
    - Infer the unique block width from the erased block and the common four-way
      tile family, label each visible block in its own block coordinates, and find
      the shortest posture period consistent with every visible position.
    - Extrapolate that cycle at the erased position and paste the correspondingly
      transformed base tile there, leaving all other cells unchanged.

    Key insight:
    - Pixel repetition identifies the tile content, while the independent posture
      cycle identifies its required orientation; both layers are needed to repair
      the missing block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    bg = _dominant_color(g)
    candidates = _candidate_repairs(g, bg)
    assert len(candidates) == 1, "tile width and missing posture must be unique"
    return candidates[0][3]


# Generator state deliberately rotates hard branches while all visual parameters
# remain random. Thus any three consecutive training examples include both cycle
# lengths and both edge/interior erasures.
_generation_index = 0
_PALETTE = tuple(range(10))


def _random_asymmetric_tile(h: int, w: int, bg: int, colors: list[int]) -> np.ndarray:
    for _ in range(300):
        tile = np.full((h, w), bg, dtype=int)
        density = random.uniform(0.36, 0.66)
        mask = np.random.random((h, w)) < density
        # Visible anchors on all sides make the full block frame apparent.
        for cells in (
            [(0, c) for c in range(w)],
            [(h - 1, c) for c in range(w)],
            [(r, 0) for r in range(h)],
            [(r, w - 1) for r in range(h)],
        ):
            r, c = random.choice(cells)
            mask[r, c] = True
        for r, c in np.argwhere(mask):
            tile[int(r), int(c)] = random.choice(colors)

        postures = _variants(tile)
        distinct = all(
            not np.array_equal(postures[i], postures[j])
            for i in range(4)
            for j in range(i)
        )
        if distinct and np.count_nonzero(tile != bg) >= 5:
            return tile
    raise RuntimeError("failed to create an asymmetric tile")


def _build_instance(branch: int) -> tuple[np.ndarray, np.ndarray, int]:
    period = 3 if branch == 0 else 4
    count = random.choice((7, 8)) if period == 3 else random.choice((9, 10))
    max_w = 30 // count
    block_w = random.randint(3, max_w)
    height = random.randint(3, 7)

    bg = random.choice(_PALETTE)
    foreground = random.sample([color for color in _PALETTE if color != bg], random.randint(2, 3))
    tile = _random_asymmetric_tile(height, block_w, bg, foreground)

    # Branches 0/2 erase an edge and branches 1/3 erase an interior position.
    pool = [0, count - 1] if branch % 2 == 0 else list(range(1, count - 1))
    while True:
        cycle = random.sample(range(4), period)
        sequence = [cycle[i % period] for i in range(count)]
        if any(sequence[i] != 0 for i in pool):
            break
    # Replacing the blank with the untransformed generator tile must be wrong.
    choices = [i for i in pool if sequence[i] != 0]
    hole = random.choice(choices)

    postures = _variants(tile)
    expected = np.concatenate([postures[state] for state in sequence], axis=1)
    inp = expected.copy()
    inp[:, hole * block_w : (hole + 1) * block_w] = bg
    return inp, expected, period


def generate_input() -> np.ndarray:
    """Generate a diverse, uniquely decodable missing-tile instance."""
    global _generation_index
    branch = _generation_index % 4
    _generation_index += 1

    for _ in range(400):
        inp, expected, intended_period = _build_instance(branch)
        colors, counts = np.unique(inp, return_counts=True)
        if np.count_nonzero(counts == counts.max()) != 1:
            continue
        bg = int(colors[np.argmax(counts)])
        candidates = _candidate_repairs(inp, bg)
        if len(candidates) != 1:
            continue
        block_w, hole, inferred_period, repaired = candidates[0]
        if inferred_period != intended_period or not np.array_equal(repaired, expected):
            continue

        # Semantic guards: exactly one full block is added, and neither a visible
        # neighbor nor the generator's base posture is a valid shortcut.
        target = expected[:, hole * block_w : (hole + 1) * block_w]
        neighbors = []
        if hole > 0:
            neighbors.append(expected[:, (hole - 1) * block_w : hole * block_w])
        if hole + 1 < expected.shape[1] // block_w:
            neighbors.append(expected[:, (hole + 1) * block_w : (hole + 2) * block_w])
        base = expected[:, :block_w]
        if np.array_equal(target, base) or any(np.array_equal(target, item) for item in neighbors):
            continue
        changed = inp != expected
        expected_changed = np.zeros(inp.shape, dtype=bool)
        expected_changed[:, hole * block_w : (hole + 1) * block_w] = target != bg
        if not np.array_equal(changed, expected_changed):
            continue
        return inp

    raise RuntimeError("failed to generate a unique nondegenerate posture-cycle instance")
