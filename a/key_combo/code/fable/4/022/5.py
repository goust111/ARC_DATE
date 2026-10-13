from __future__ import annotations

import math
import numpy as np


PALETTE = tuple(range(10))
K_CHOICES = (2, 3, 4, 5, 6)
LAYOUT_FILL_RANGE = (0.28, 0.72)
TILE_FILL_RANGE = (0.22, 0.78)
MAX_GENERATION_ATTEMPTS = 400


def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    return np.kron(outer.astype(int), inner.astype(int)).astype(bool)


def _candidate_factorizations(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    colors = [int(c) for c in np.unique(g)]
    assert len(colors) == 2, "expected exactly one foreground color and one background color"

    h, w = map(int, g.shape)
    candidates: list[dict] = []

    for fg in colors:
        bg = colors[0] if colors[1] == fg else colors[1]
        mask = g == fg

        for k in range(2, min(h, w) + 1):
            if h % k != 0 or w % k != 0:
                continue

            outer_h = h // k
            outer_w = w // k
            if outer_h * outer_w <= 1:
                continue

            tile = None
            outer = np.zeros((outer_h, outer_w), dtype=bool)
            valid = True

            for br in range(outer_h):
                if not valid:
                    break
                for bc in range(outer_w):
                    block = mask[br * k : (br + 1) * k, bc * k : (bc + 1) * k]
                    if not block.any():
                        continue
                    if block.all():
                        valid = False
                        break
                    if tile is None:
                        tile = block.copy()
                    elif not np.array_equal(block, tile):
                        valid = False
                        break
                    outer[br, bc] = True

            if not valid or tile is None:
                continue
            if not outer.any() or outer.all():
                continue
            if not tile.any() or tile.all():
                continue

            candidates.append(
                {
                    "bg": int(bg),
                    "fg": int(fg),
                    "k": int(k),
                    "outer": outer,
                    "tile": tile,
                }
            )

    return candidates


def _unique_factorization(grid: np.ndarray) -> dict:
    candidates = _candidate_factorizations(grid)
    assert len(candidates) == 1, f"expected a unique nontrivial factorization, found {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden block layout and hidden tile, then swap their roles in the Kronecker product.

    Input structure:
    - The grid uses exactly one foreground color and one background color.
    - It can be partitioned into equal k x k blocks; each block is either all background
      or exactly the same mixed foreground/background tile.
    - The nontrivial decomposition with both a mixed tile and a mixed outer layout is unique.

    Rule:
    - Invert the product to obtain the outer layout and the shared k x k tile.
    - Swap the factors: treat the tile as the new outer layout and the original layout as
      the new tile.
    - For each foreground cell of the tile, stamp one copy of the original layout; tile
      background cells become all-background blocks. Keep the original two colors.

    Key insight:
    - The visible picture is a Kronecker product with hidden factors, so the solution must
      first recover that unique factorization and only then rebuild the grid with the factors exchanged.
    """
    factorization = _unique_factorization(grid)
    swapped_mask = _compose(factorization["tile"], factorization["outer"])
    out = np.full(swapped_mask.shape, factorization["bg"], dtype=int)
    out[swapped_mask] = factorization["fg"]
    assert out.shape == np.asarray(grid).shape
    return out


def _sample_mask(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    fill_range: tuple[float, float],
) -> np.ndarray:
    area = rows * cols
    lo = max(2, math.ceil(area * fill_range[0]))
    hi = min(area - 2, math.floor(area * fill_range[1]))
    assert lo <= hi, "fill range leaves no mixed mask"

    for _ in range(200):
        count = int(rng.integers(lo, hi + 1))
        mask = np.zeros((rows, cols), dtype=bool)
        indices = rng.choice(area, size=count, replace=False)
        mask.flat[indices] = True
        if mask.any() and not mask.all():
            return mask
    raise RuntimeError("failed to sample a mixed binary mask")


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    k = int(rng.choice(K_CHOICES))
    max_outer = min(8, 30 // k)
    if max_outer < 2:
        return None

    outer_h = int(rng.integers(2, max_outer + 1))
    outer_w = int(rng.integers(2, max_outer + 1))

    tile = _sample_mask(rng, k, k, TILE_FILL_RANGE)
    outer = _sample_mask(rng, outer_h, outer_w, LAYOUT_FILL_RANGE)

    bg, fg = map(int, rng.choice(PALETTE, size=2, replace=False))
    input_mask = _compose(outer, tile)
    output_mask = _compose(tile, outer)

    if np.array_equal(input_mask, output_mask):
        return None

    grid = np.full(input_mask.shape, bg, dtype=int)
    grid[input_mask] = fg

    candidates = _candidate_factorizations(grid)
    if len(candidates) != 1:
        return None

    candidate = candidates[0]
    if candidate["fg"] != fg or candidate["bg"] != bg or candidate["k"] != k:
        return None
    if not np.array_equal(candidate["outer"], outer):
        return None
    if not np.array_equal(candidate["tile"], tile):
        return None

    expected = np.full(output_mask.shape, bg, dtype=int)
    expected[output_mask] = fg
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
