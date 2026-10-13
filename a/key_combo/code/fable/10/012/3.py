from __future__ import annotations

import math
import random

import numpy as np


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, np.unique(grid))))
    if exclude:
        blocked = set(map(int, exclude))
        colors = [color for color in colors if color not in blocked]
    return colors


def _split_tiles(grid: np.ndarray, k: int) -> list[list[np.ndarray]]:
    return [
        [grid[r * k : (r + 1) * k, c * k : (c + 1) * k] for c in range(k)]
        for r in range(k)
    ]


def _swap_two_colors(tile: np.ndarray) -> np.ndarray:
    colors = _colors_present(tile)
    assert len(colors) == 2, "each tile must be binary"
    a, b = colors
    return np.where(tile == a, b, a)


def _classify_tiles(grid: np.ndarray, k: int, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverted = _swap_two_colors(reference)
    macro = np.full((k, k), -1, dtype=int)
    tiles = _split_tiles(grid, k)
    for r in range(k):
        for c in range(k):
            tile = tiles[r][c]
            if np.array_equal(tile, reference):
                macro[r, c] = 0
            elif np.array_equal(tile, inverted):
                macro[r, c] = 1
            else:
                raise AssertionError("every tile must equal the reference or its inversion")
    return macro, inverted


def _candidate_masks(reference: np.ndarray) -> list[np.ndarray]:
    return [(reference == color).astype(int) for color in _colors_present(reference)]


def _repair_with_reference(grid: np.ndarray, k: int, reference: np.ndarray) -> list[np.ndarray]:
    macro, inverted = _classify_tiles(grid, k, reference)
    solutions: list[np.ndarray] = []
    for mask in _candidate_masks(reference):
        bad = np.argwhere(macro != mask)
        if len(bad) != 1:
            continue
        br, bc = map(int, bad[0])
        patch = inverted if int(mask[br, bc]) == 1 else reference
        out = grid.copy()
        out[br * k : (br + 1) * k, bc * k : (bc + 1) * k] = patch
        repaired_macro, _ = _classify_tiles(out, k, reference)
        if np.array_equal(repaired_macro, mask):
            solutions.append(out)
    return solutions


def _unique_repair(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "grid must be square"
    side = int(g.shape[0])
    k = int(math.isqrt(side))
    assert k * k == side, "grid side must equal k^2"
    assert len(_colors_present(g)) == 2, "grid must use exactly two colors"

    unique: dict[bytes, np.ndarray] = {}
    for row in _split_tiles(g, k):
        for reference in row:
            for candidate in _repair_with_reference(g, k, reference):
                key = candidate.tobytes()
                unique.setdefault(key, candidate)

    assert unique, "no self-referential repair found"
    assert len(unique) == 1, "repair must be unique"
    return next(iter(unique.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the one tile whose inversion state breaks a self-referential tile array.

    Input structure:
    - The square grid side length is k^2, so it splits into a k-by-k array of equal k-by-k tiles.
    - Every tile is the same binary base pattern or its two-color inversion.
    - The two tile versions are reused across the whole array, with exactly one tile using the wrong version.

    Rule:
    - Infer the two tile versions from the repeated tiles.
    - Find the binary interpretation for which the macro pattern of "inverted tile or not"
      matches the reference tile's own pixel pattern except at one macro position.
    - Replace that single mismatching tile with the version required by the self-reference and
      return the repaired full grid.

    Key insight:
    The bitmap drawn inside a tile reappears one scale higher as the map of which tile positions
    should be inverted, so the bad tile is the only macro cell where those two scales disagree.
    """
    return _unique_repair(grid)


# --------------- generator (o5x) ---------------

K_RANGE = (3, 5)
FOREGROUND_COLORS = tuple(range(1, 10))
FOREGROUND_FRACTION_RANGE = (0.30, 0.55)
GENERATION_ATTEMPTS = 500


def _has_both_colors_per_axis(mask: np.ndarray) -> bool:
    return (
        mask.any(axis=0).all()
        and mask.any(axis=1).all()
        and (~mask).any(axis=0).all()
        and (~mask).any(axis=1).all()
    )


def _is_checkerboard(mask: np.ndarray) -> bool:
    rows, cols = np.indices(mask.shape)
    checker = ((rows + cols) % 2).astype(bool)
    return np.array_equal(mask, checker) or np.array_equal(mask, ~checker)


def _enough_structure(mask: np.ndarray) -> bool:
    row_patterns = {tuple(map(int, row)) for row in mask}
    col_patterns = {tuple(map(int, col)) for col in mask.T}
    return len(row_patterns) >= 2 and len(col_patterns) >= 2


def _gen_pattern(k: int, ones: int) -> np.ndarray | None:
    mask = np.zeros((k, k), dtype=bool)

    cols = list(range(k))
    random.shuffle(cols)
    for r, c in enumerate(cols):
        mask[r, c] = True

    row_counts = mask.sum(axis=1).astype(int).tolist()
    col_counts = mask.sum(axis=0).astype(int).tolist()

    for _ in range(ones - k):
        choices = [
            (r, c)
            for r in range(k)
            for c in range(k)
            if not mask[r, c] and row_counts[r] < k - 1 and col_counts[c] < k - 1
        ]
        if not choices:
            return None
        r, c = random.choice(choices)
        mask[r, c] = True
        row_counts[r] += 1
        col_counts[c] += 1

    if not _has_both_colors_per_axis(mask):
        return None
    if _is_checkerboard(mask):
        return None
    if not _enough_structure(mask):
        return None
    return mask


def _tile_from_mask(mask: np.ndarray, fg: int) -> np.ndarray:
    tile = np.zeros(mask.shape, dtype=int)
    tile[mask] = int(fg)
    return tile


def _render_correct_grid(mask: np.ndarray, fg: int) -> np.ndarray:
    k = int(mask.shape[0])
    tile = _tile_from_mask(mask, fg)
    inverted = np.where(tile == 0, fg, 0)
    out = np.zeros((k * k, k * k), dtype=int)
    for r in range(k):
        for c in range(k):
            patch = inverted if bool(mask[r, c]) else tile
            out[r * k : (r + 1) * k, c * k : (c + 1) * k] = patch
    return out


def _count_different_tiles(a: np.ndarray, b: np.ndarray) -> int:
    k = int(math.isqrt(a.shape[0]))
    count = 0
    for r in range(k):
        for c in range(k):
            ra = slice(r * k, (r + 1) * k)
            ca = slice(c * k, (c + 1) * k)
            if not np.array_equal(a[ra, ca], b[ra, ca]):
                count += 1
    return count


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if _count_different_tiles(inp, actual) != 1:
        return False

    unique: dict[bytes, np.ndarray] = {}
    k = int(math.isqrt(inp.shape[0]))
    for row in _split_tiles(inp, k):
        for reference in row:
            for candidate in _repair_with_reference(inp, k, reference):
                unique.setdefault(candidate.tobytes(), candidate)
    if len(unique) != 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(*K_RANGE)
    total = k * k
    low = max(k, 2, math.ceil(FOREGROUND_FRACTION_RANGE[0] * total))
    high = min(total - k, total - 2, math.floor(FOREGROUND_FRACTION_RANGE[1] * total))
    if low > high:
        return None

    ones = random.randint(low, high)
    mask = _gen_pattern(k, ones)
    if mask is None:
        return None

    fg = random.choice(FOREGROUND_COLORS)
    expected = _render_correct_grid(mask, fg)
    inp = expected.copy()

    wrong_r = random.randrange(k)
    wrong_c = random.randrange(k)
    tile = inp[wrong_r * k : (wrong_r + 1) * k, wrong_c * k : (wrong_c + 1) * k].copy()
    inp[wrong_r * k : (wrong_r + 1) * k, wrong_c * k : (wrong_c + 1) * k] = _swap_two_colors(tile)

    tile_versions = {
        inp[r * k : (r + 1) * k, c * k : (c + 1) * k].tobytes()
        for r in range(k)
        for c in range(k)
    }
    if len(tile_versions) != 2:
        return None
    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential inverted-tile input")
