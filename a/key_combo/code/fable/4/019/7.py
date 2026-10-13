from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import random
import sys

import numpy as np


# Make arc_utils importable when loaded via SourceFileLoader.
_current = Path(__file__).resolve()
for parent in (_current.parent, *_current.parents):
    candidate = parent / "new" / "solvers" / "arc_utils.py"
    if candidate.exists():
        arc_utils_dir = str(candidate.parent)
        if arc_utils_dir not in sys.path:
            sys.path.append(arc_utils_dir)
        break
else:
    raise ImportError("arc_utils.py not found in project hierarchy")

from arc_utils import Color, get_bg, find_components  # noqa: E402


BG = int(Color.BLACK)


def _band_height(g: np.ndarray, bg: int) -> int:
    """Length of the top non-background full-width band."""
    rows = g.shape[0]
    full = np.all(g != bg, axis=1)
    if not full.size or not bool(full[0]):
        raise AssertionError("expected top band row 0 to be fully non-background")

    end = 0
    while end < rows and bool(full[end]):
        end += 1
    return end


def _find_square_period(g: np.ndarray, bg: int, band_rows: int) -> int:
    """Find the isolated solid square whose color never appears in the top band."""
    band_colors: set[int] = set(map(int, np.unique(g[:band_rows, :])))
    squares: list[tuple[int, int]] = []  # (side, color)

    for comp in find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        if int(comp["area"]) == 0:
            continue

        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h != w:
            continue
        if int(comp["area"]) != h * w:
            continue
        if int(r0) < band_rows:
            continue

        color = int(comp["color"])
        if color in band_colors:
            continue

        if h == 1:
            continue
        squares.append((h, color))

    if len(squares) != 1:
        raise AssertionError(f"expected one valid parameter square, found {len(squares)}")

    side, color = squares[0]
    if side <= 0:
        raise AssertionError("invalid square side")
    return side


def _tile_vertical(g: np.ndarray, p: int) -> np.ndarray:
    """Repeat top `p` rows down the full height."""
    if p <= 0:
        raise ValueError("period must be positive")
    H, W = map(int, g.shape)
    if p > H:
        raise AssertionError("period exceeds grid height")

    seed = np.asarray(g[:p, :], dtype=int)
    out = np.empty((H, W), dtype=int)
    for r in range(H):
        out[r, :] = seed[r % p, :]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the true top-band period downward to fill the full grid.

    Input structure:
    - Top rows form a full-width color band where row patterns are periodic.
    - A separate isolated solid square elsewhere has color not appearing in the band.
    - The square side length is the true vertical period of the band.

    Rule:
    - Detect the square side length `p` and the top full-width band height.
    - Use the first `p` input rows as the period seed.
    - Replace the entire grid by vertical repetition of that seed every `p` rows.

    Key insight:
    - The visible band can be compatible with multiple periods, but the hidden parameter
      square fixes the true period by its side length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    band_rows = _band_height(g, bg)
    p = _find_square_period(g, bg, band_rows)
    if band_rows < p:
        raise AssertionError(
            f"band rows ({band_rows}) shorter than true period ({p})"
        )

    return _tile_vertical(g, p)


def _rand_band_row(width: int, colors: Iterable[int]) -> np.ndarray:
    """Build one visually structured full-width row for the band."""
    cols = list(colors)
    seg_count = random.randint(2, min(6, width))
    cuts = sorted(random.sample(range(1, width), seg_count - 1))
    cut_points = [0, *cuts, width]

    row = np.empty(width, dtype=int)
    for lo, hi in zip(cut_points[:-1], cut_points[1:]):
        row[lo:hi] = random.choice(cols)
    return row


def _proper_divisors(n: int) -> list[int]:
    return [d for d in range(2, n) if n % d == 0]


def _make_band_rows(p: int, width: int, palette: list[int]) -> list[np.ndarray]:
    """Construct one `p`-row block with optional decoy period."""
    if p <= 0:
        raise ValueError("p must be positive")

    use_decoy = random.random() < 0.75
    if use_decoy:
        divs = _proper_divisors(p)
        if divs:
            d = random.choice(divs)
        else:
            d = p
            use_decoy = False
    else:
        d = p

    base = [_rand_band_row(width, palette) for _ in range(d)]
    return [base[i % d].copy() for i in range(p)], use_decoy


def _gen_attempt() -> np.ndarray | None:
    """One randomized construction attempt. Returns None on failure."""
    H = random.randint(18, 30)
    W = random.randint(12, 30)

    p = random.randint(2, max(2, H // 3))
    if H - 2 * p < p:
        return None

    band_rows = random.randint(p, H - 2 * p)

    band_palette = random.sample([c for c in Color.NOT_BLACK if c != BG], k=random.randint(2, 4))
    rows, used_decoy = _make_band_rows(p, W, band_palette)

    grid = np.full((H, W), BG, dtype=int)
    for r in range(band_rows):
        grid[r, :] = rows[r % p]

    square_color_pool = [c for c in Color.NOT_BLACK if c != BG and c not in band_palette]
    if not square_color_pool:
        return None
    square_color = random.choice(square_color_pool)

    sr = random.randint(band_rows, H - p)
    sc = random.randint(0, W - p)
    grid[sr : sr + p, sc : sc + p] = square_color

    try:
        expected = main(grid)
    except AssertionError:
        return None

    if expected.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Create random ARC-style inputs that satisfy the main-rule assumptions."""
    for _ in range(800):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid 7 input")
