"""ARC task 7: decode defects hidden in a periodic tile field."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Model:
    tile_h: int
    tile_w: int
    template: np.ndarray
    # (tile row, tile col, local row, local col, replacement color)
    defects: tuple[tuple[int, int, int, int, int], ...]


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _is_fundamental(tile: np.ndarray) -> bool:
    """Whether tile has no strictly smaller rectangular translation period."""
    h, w = tile.shape
    for ph in _divisors(h):
        for pw in _divisors(w):
            if ph * pw == h * w:
                continue
            small = tile[:ph, :pw]
            if np.array_equal(np.tile(small, (h // ph, w // pw)), tile):
                return False
    return True


def _analyze_period(g: np.ndarray, ph: int, pw: int) -> _Model | None:
    """Recover and validate the consensus tile for one candidate period."""
    h, w = g.shape
    rows, cols = h // ph, w // pw
    tile_count = rows * cols
    if tile_count < 3:
        return None

    stack = g.reshape(rows, ph, cols, pw).transpose(0, 2, 1, 3)
    template = np.empty((ph, pw), dtype=int)
    for r in range(ph):
        for c in range(pw):
            values, counts = np.unique(stack[:, :, r, c], return_counts=True)
            best = int(np.argmax(counts))
            # A strict majority makes every recovered template cell unambiguous.
            if int(counts[best]) * 2 <= tile_count:
                return None
            template[r, c] = int(values[best])

    if not _is_fundamental(template):
        return None
    palette = set(map(int, np.unique(template)))
    # A defect must change to a different color already visible in the tile.
    if len(palette) < 2:
        return None

    defects: list[tuple[int, int, int, int, int]] = []
    for tr in range(rows):
        for tc in range(cols):
            diff = np.argwhere(stack[tr, tc] != template)
            if len(diff) > 1:
                return None
            if len(diff) == 1:
                lr, lc = map(int, diff[0])
                ink = int(stack[tr, tc, lr, lc])
                if ink not in palette or ink == int(template[lr, lc]):
                    return None
                defects.append((tr, tc, lr, lc, ink))

    # Defects are present but remain a minority of the tile array.
    if not defects or 2 * len(defects) >= tile_count:
        return None
    addresses = [(d[2], d[3]) for d in defects]
    if len(set(addresses)) != len(addresses):
        return None
    return _Model(ph, pw, template, tuple(defects))


def _valid_models(g: np.ndarray) -> list[_Model]:
    models = []
    for ph in _divisors(g.shape[0]):
        for pw in _divisors(g.shape[1]):
            model = _analyze_period(g, ph, pw)
            if model is not None:
                models.append(model)
    return models


def _render(model: _Model) -> np.ndarray:
    out = np.zeros((model.tile_h, model.tile_w), dtype=int)
    for _tr, _tc, lr, lc, ink in model.defects:
        out[lr, lc] = ink
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode same-palette defects in a periodic field into one address tile.

    Input structure:
    - The 2D grid is an integer number of repetitions of one fundamental p x q
      tile. A minority of tiles differ from it at exactly one pixel; every
      replacement color already belongs to the tile palette.

    Rule:
    - Recover the unique p x q tile by strict per-offset majority agreement.
      For every damaged tile, place its replacement color at the changed
      pixel's tile-local (row, column) coordinate in an otherwise black p x q
      output. Damaged tiles have distinct local coordinates.

    Key insight:
    - The repeated texture hides in-palette changes: the defect's local offset,
      not the damaged tile's position in the larger array, is the output address.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    models = _valid_models(g)
    assert len(models) == 1, "the fundamental damaged-tile model must be unique"
    return _render(models[0])


_PALETTE = tuple(range(1, 10))
_MAX_ATTEMPTS = 500


def _random_template(ph: int, pw: int, colors: tuple[int, ...]) -> np.ndarray | None:
    """Make a readable, non-degenerate tile with every color well represented."""
    # Begin with diagonal bands, then perturb a few cells.  The result remains
    # visually patterned while avoiding accidental smaller translation periods.
    rr, cc = np.mgrid[:ph, :pw]
    offset = random.randrange(len(colors))
    tile = np.asarray(colors, dtype=int)[(rr + cc + offset) % len(colors)]
    for _ in range(random.randint(1, max(1, (ph * pw) // 4))):
        r, c = random.randrange(ph), random.randrange(pw)
        choices = [x for x in colors if x != int(tile[r, c])]
        tile[r, c] = random.choice(choices)
    counts = [int(np.count_nonzero(tile == color)) for color in colors]
    if min(counts) < 2 or not _is_fundamental(tile):
        return None
    return tile


def _oracle(template: np.ndarray, defects: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    expected = np.zeros(template.shape, dtype=int)
    for _tr, _tc, lr, lc, ink in defects:
        expected[lr, lc] = ink
    return expected


def _semantic_check(
    grid: np.ndarray,
    template: np.ndarray,
    defects: list[tuple[int, int, int, int, int]],
) -> bool:
    """Check the described rule and reject the tempting tile-position shortcut."""
    ph, pw = template.shape
    models = _valid_models(grid)
    if len(models) != 1:
        return False
    model = models[0]
    if (model.tile_h, model.tile_w) != (ph, pw):
        return False
    if not np.array_equal(model.template, template):
        return False

    expected = _oracle(template, defects)
    if not np.array_equal(_render(model), expected):
        return False

    # Because the tile array is also ph x pw, this wrong interpretation has the
    # right output shape and is a genuine decoy rather than a dimension giveaway.
    decoy = np.zeros((ph, pw), dtype=int)
    for tr, tc, _lr, _lc, ink in defects:
        decoy[tr, tc] = ink
    if np.array_equal(decoy, expected):
        return False

    addresses = {(lr, lc) for _tr, _tc, lr, lc, _ink in defects}
    tile_positions = {(tr, tc) for tr, tc, _lr, _lc, _ink in defects}
    inks = {ink for _tr, _tc, _lr, _lc, ink in defects}
    # Both the local-address step and the replacement-color step visibly matter.
    return addresses != tile_positions and len(inks) >= 2


def _gen_attempt() -> np.ndarray | None:
    ph = random.randint(3, 5)
    pw = random.randint(3, 5)

    nonblack = random.sample(_PALETTE, 2)
    colors = (0, nonblack[0], nonblack[1])
    template = _random_template(ph, pw, colors)
    if template is None:
        return None

    tile_count = ph * pw  # array dimensions intentionally match tile dimensions
    defect_count = random.randint(3, max(3, tile_count // 3))
    tile_positions = random.sample([(r, c) for r in range(ph) for c in range(pw)], defect_count)
    local_positions = random.sample([(r, c) for r in range(ph) for c in range(pw)], defect_count)
    if set(tile_positions) == set(local_positions):
        return None

    # Force at least two visible ink colors; all inks still come from the tile.
    required_inks = [nonblack[0], nonblack[1]]
    defects: list[tuple[int, int, int, int, int]] = []
    for i, ((tr, tc), (lr, lc)) in enumerate(zip(tile_positions, local_positions)):
        allowed = [color for color in nonblack if color != int(template[lr, lc])]
        if not allowed:
            return None
        ink = required_inks[i] if i < 2 else random.choice(allowed)
        if ink == int(template[lr, lc]):
            alternatives = [color for color in allowed if color != ink]
            if not alternatives:
                return None
            ink = random.choice(alternatives)
        defects.append((tr, tc, lr, lc, int(ink)))
    if len({d[4] for d in defects}) < 2:
        return None

    grid = np.tile(template, (ph, pw))
    for tr, tc, lr, lc, ink in defects:
        grid[tr * ph + lr, tc * pw + lc] = ink

    if max(grid.shape) > 30 or not _semantic_check(grid, template, defects):
        return None
    try:
        if not np.array_equal(main(grid), _oracle(template, defects)):
            return None
    except (AssertionError, ValueError):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied valid periodic field with several hidden defects."""
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a unique non-degenerate defect puzzle")
